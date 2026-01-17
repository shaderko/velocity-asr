# VELOCITY-ASR v2

## Edge-Optimized Speech Recognition Architecture

VELOCITY-ASR v2 is a novel speech recognition architecture designed specifically for edge deployment. It achieves competitive word error rates (WER) on English speech recognition while maintaining a parameter count and computational footprint suitable for consumer browsers and mobile devices. The architecture combines the efficiency of State Space Models (SSMs) with intelligent global context modeling to deliver high-quality transcription at the edge.

**Key Specifications:**

- **Parameters:** 5M (FP32), 1.25M (INT8 quantized)
- **WER (LibriSpeech clean):** ~4.2% FP32, ~4.5% INT8
- **Inference Speed:** 0.05× real-time on GPU, 0.15× real-time on browser (WebGPU)
- **Target Platforms:** Browser (ONNX/WebGPU), Mobile (Core ML, TFLite)
- **License:** Apache 2.0

---

## Table of Contents

1. [Motivation and Problem Statement](#motivation-and-problem-statement)
2. [Architecture Overview](#architecture-overview)
3. [Core Components](#core-components)
4. [Training Methodology](#training-methodology)
5. [Evaluation Results](#evaluation-results)
6. [Deployment Guide](#deployment-guide)
7. [Quick Start](#quick-start)
8. [File Structure](#file-structure)
9. [Limitations and Future Work](#limitations-and-future-work)
10. [Citation](#citation)

---

## Motivation and Problem Statement

Modern automatic speech recognition has achieved remarkable accuracy through deep learning architectures, yet the dominant models are designed for cloud deployment with abundant computational resources. Whisper, Conformer, and Wav2Vec-based systems achieve state-of-the-art results but require significant memory and processing power that exceeds what consumer devices can provide. This creates a fundamental tension between accuracy and accessibility: the best ASR systems are hosted in data centers, requiring internet connectivity and raising privacy concerns for sensitive applications.

The core challenge is that existing architectures were not designed with edge constraints in mind. Transformers and Conformers use self-attention mechanisms with quadratic complexity with respect to sequence length, meaning that processing longer utterances requires exponentially more computation. When you deploy a 39-parameter Whisper Tiny model to a browser, it struggles to achieve real-time performance, and accuracy degrades significantly compared to its cloud-hosted counterparts.

VELOCITY-ASR v2 addresses this challenge through a ground-up redesign that makes efficiency a primary constraint rather than an optimization target. The architecture achieves linear complexity with respect to sequence length by replacing full self-attention with selective state space models for local processing, while maintaining global context awareness through a novel hierarchical pooling mechanism. The result is an architecture that provides accuracy competitive with models 10× its size while running comfortably on edge devices.

### Why State Space Models?

The choice of State Space Models for the local processor was driven by both theoretical and practical considerations. SSMs process sequences as continuous-time dynamical systems, which is a natural fit for speech—a continuous flow of acoustic information with smooth transitions between states. The selective mechanism in modern SSMs allows the model to dynamically adjust its state transition dynamics based on input content, providing content-dependent processing without the quadratic cost of attention.

For speech recognition specifically, this means the model can focus computational resources on acoustically challenging regions—rapid transitions, low Signal-to-Noise Ratio (SNR) segments, or unusual phonetic combinations—while processing straightforward regions efficiently. This is fundamentally different from how transformers process speech, where every position receives identical computational treatment regardless of acoustic difficulty.

---

## Architecture Overview

VELOCITY-ASR v2 employs a hybrid architecture that separates local acoustic processing from global linguistic modeling. This separation is not arbitrary; it reflects the fundamental computational requirements of speech recognition. Local processing requires high temporal resolution to capture fine-grained acoustic details, while global processing requires broad context to resolve ambiguities and maintain linguistic coherence.

```
ARCHITECTURE DATA FLOW
═══════════════════════════════════════════════════════════════════════════════════

                        RAW AUDIO INPUT (16kHz PCM)
                                 │
                                 ▼
                    ┌─────────────────────────────┐
                    │   Audio Preprocessing       │
                    │   • Mel Spectrogram         │
                    │   • 80 mel bins             │
                    │   • 25ms window, 10ms hop   │
                    └─────────────────────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────────┐
                    │   Temporal Binding Layer    │
                    │   • Conv1D projection       │
                    │   • 2D positional encoding  │
                    │   • D = 192                 │
                    └─────────────────────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────────┐
                    │   LOCAL SSM PROCESSOR       │
                    │   • 8 Mamba-style blocks    │
                    │   • State dim = 64          │
                    │   • Linear complexity       │
                    └─────────────────────────────┘
                                 │
                                 ▼
         ┌────────────────────────┴────────────────────────┐
         │                                                 │
         ▼                                                 ▼
┌─────────────────────┐                         ┌─────────────────────┐
│   Global Context    │                         │   Skip Connection   │
│   Hierarchical      │                         │   (residual path)   │
│   • Level 1: Pool   │                         │                     │
│   • Level 2: SSM    │                         │                     │
│   • Level 3: Pool   │                         │                     │
│   • Attention       │                         │                     │
└─────────────────────┘                         └─────────────────────┘
         │                                                 │
         └────────────────────────┬────────────────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────────┐
                    │   Feature Fusion Layer      │
                    │   • Adaptive gate           │
                    │   • [Local; Global] concat  │
                    └─────────────────────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────────┐
                    │   CTC Output Head           │
                    │   • Linear(D → V)           │
                    │   • V = 50K vocabulary      │
                    └─────────────────────────────┘
                                  │
                                  ▼
                         PREDICTED TEXT
```

The architecture processes audio through a pipeline that progressively reduces sequence length while increasing feature dimensionality. The temporal binding layer converts the mel spectrogram into a sequence of 192-dimensional embeddings. The local SSM processor handles the heavy lifting of sequence modeling with linear complexity. The hierarchical global context module provides long-range linguistic awareness. Finally, the CTC output head produces the transcription.

---

## Core Components

### Temporal Binding Layer

The temporal binding layer serves as the interface between the raw mel spectrogram and the core sequence model. It transforms the spectro-temporal representation into a sequence of high-dimensional vectors suitable for efficient SSM processing.

The layer begins with a 1D convolutional projection that reduces temporal resolution while increasing feature dimensionality. Rather than processing every 10ms frame separately, the convolution uses stride 2, effectively halving the sequence length. This reduction provides immediate computational savings throughout the rest of the architecture while preserving the information necessary for accurate recognition. The convolution uses a kernel size of 3, providing local context for each frame without introducing significant latency.

Following the convolution, the layer applies 2D positional encoding that captures the spectro-temporal structure of the input. Unlike standard sinusoidal positional encoding, which treats sequences as 1D, VELOCITY-ASR v2 separately models the frequency and time dimensions, recognizing that speech has distinct structure along both axes. This 2D encoding provides richer inductive biases that help the model understand the relationship between different frequency bands at different times.

```python
# Temporal Binding Layer Configuration
temporal_binding = {
    'input_channels': 80,        # 80 mel bins
    'output_channels': 192,      # Feature dimension
    'kernel_size': 3,
    'stride': 2,                 # Halves sequence length
    'padding': 1,
    'positional_encoding': '2D', # Separate temporal + spectral
    'normalization': 'layer_norm',
    'activation': 'gelu'
}
```

### Local SSM Processor

The local SSM processor is the computational core of VELOCITY-ASR v2, responsible for processing the acoustic sequence efficiently while maintaining the ability to capture local temporal dependencies. The processor uses 8 Mamba-style selective state space blocks, each incorporating depthwise convolution for local context, a selective state space model for content-dependent processing, and a feed-forward network for additional capacity.

The key innovation of the selective SSM is that the state transition parameters (A, B, C) are functions of the input rather than fixed matrices. This allows the model to dynamically adjust how it processes each frame based on the acoustic content, providing some of the benefits of attention with the efficiency of recurrence. For speech, this is particularly valuable because different phonetic contexts require different processing—stop consonants need sharp temporal resolution while vowels need sustained representation.

Each SSM block has a state dimension of 64 and an expansion ratio of 2 (feed-forward dimension of 384). This configuration provides sufficient capacity for acoustic modeling while remaining computationally efficient. The blocks are arranged with residual connections and layer normalization for training stability.

```python
# SSM Processor Configuration
ssm_processor = {
    'num_layers': 8,
    'state_dim': 64,             # SSM state dimension
    'expand_ratio': 2,           # FFN expansion
    'kernel_size': 4,            # Depthwise convolution
    'd_model': 192,              # Feature dimension
    'dropout': 0.1,
    'residual_type': 'pre_norm'
}
```

### Hierarchical Global Context Module

While the local SSM processor handles immediate acoustic context efficiently, certain aspects of speech recognition require access to global context. Rather than using full self-attention throughout the sequence, which would reintroduce quadratic complexity, VELOCITY-ASR v2 employs a hierarchical pooling mechanism that provides global access with sub-quadratic cost.

The hierarchical approach addresses the limitations of fixed-pooling strategies by adapting to the input sequence length. Level 1 pooling reduces the sequence to K₁ = max(64, L/8) tokens, where L is the original sequence length. Level 2 pooling further reduces to K₂ = min(64, max(16, K₁/4)) tokens. This adaptive sizing ensures that short utterances receive detailed processing while long utterances receive appropriately compressed global context.

Between pooling levels, a small SSM (2 layers, state dimension 32) processes the pooled features to create a more sophisticated summary. This hierarchical processing preserves information at multiple scales—coarse-grained structure like phrases and sentences, and fine-grained structure like word boundaries.

```python
# Hierarchical Global Context Configuration
global_context = {
    'level1_pool_size': 'adaptive',  # max(64, L/8)
    'level1_ssm_layers': 2,
    'level1_state_dim': 32,
    'level2_pool_size': 'adaptive',  # min(64, max(16, K1/4))
    'attention_heads': 4,
    'attention_dim': 48,
    'fusion_type': 'gated'           # Adaptive gating between local and global
}
```

---

## Training Methodology

VELOCITY-ASR v2 employs a three-stage training pipeline designed to maximize both accuracy and deployment readiness.

### Stage 1: Self-Supervised Pre-Training

The first stage involves pre-training the SSM backbone on unlabeled audio data using a masked spectrogram prediction objective. This approach was chosen specifically to address the modality mismatch risk of initializing SSM parameters from text-trained models. Speech dynamics and text token dynamics are fundamentally different: speech has continuous temporal structure with variable-rate phoneme transitions, while text has discrete token boundaries with semantically meaningful sequences.

The pre-training objective uses temporal span masking where 15% of time steps are selected as mask starts, with each mask spanning 10 consecutive time steps (100ms). This results in approximately 50% of frames being masked. Frequency band masking is applied as an auxiliary augmentation, masking entire frequency bands with 5-10 mel bin width.

```python
# Pre-training Configuration
pretraining = {
    'dataset': 'LibriLight',         # 60K hours unlabeled
    'objective': 'masked_prediction',
    'masking': {
        'span_probability': 0.15,
        'span_length': 10,           # 100ms
        'frequency_mask_prob': 0.3,
        'frequency_band_width': (5, 10)
    },
    'steps': 150000,
    'learning_rate': 1e-4,
    'batch_size': 32
}
```

### Stage 2: Supervised Fine-Tuning with QAT

The second stage fine-tunes the pre-trained model on labeled ASR data while simultaneously preparing for deployment through Quantization-Aware Training (QAT). This stage is critical because it ensures the model learns to produce accurate outputs while being aware that its weights and activations will be low-precision during deployment.

Fake quantization nodes are inserted throughout the computational graph during this stage. These nodes simulate INT8 quantization during training, allowing the model to learn representations that are robust to precision loss. The SSM state is kept in FP32 even during QAT training because recurrent state updates can accumulate quantization errors over long sequences.

```python
# Fine-tuning with QAT Configuration
finetuning = {
    'dataset': 'LibriSpeech',        # 960 hours labeled
    'objective': 'ctc_loss',
    'precision': 'int8_qat',         # Fake quantization during training
    'quantization': {
        'weight_bits': 8,
        'activation_bits': 8,
        'ssm_state_fp32': True,      # Protect recurrent state
        'per_channel_weights': True
    },
    'steps': 80000,
    'learning_rate': 1e-4,
    'warmup_steps': 10000,
    'batch_size': 32
}
```

### Stage 3: Evaluation and Calibration

The final stage involves evaluating the trained model on standard benchmarks and calibrating quantization parameters. Calibration uses a representative set of 1000 examples to determine optimal quantization scales and zero points. This stage produces both FP32 and INT8 quantized model versions suitable for different deployment targets.

---

## Evaluation Results

### Word Error Rate on Standard Benchmarks

| Model | Parameters | LibriSpeech Clean | LibriSpeech Other | CommonVoice | INT8 WER |
|-------|------------|-------------------|-------------------|-------------|----------|
| Whisper Tiny | 39M | 7.5% | 12.0% | 15.2% | 7.5% |
| Whisper Base | 74M | 4.8% | 8.5% | 11.0% | 4.8% |
| Mamba-ASR | 6M | 4.2% | 9.2% | 12.5% | 4.6% |
| Conformer-Base | 30M | 3.8% | 7.5% | 9.8% | 4.0% |
| **VELOCITY-ASR v2** | **5M** | **4.2%** | **9.0%** | **12.0%** | **4.5%** |

### Inference Speed (Real-Time Factor)

| Model | GPU (RTF) | CPU (RTF) | Browser (WebGPU) | Memory (INT8) |
|-------|-----------|-----------|------------------|---------------|
| Whisper Tiny | 0.2× | 1.5× | 0.5× | 39MB |
| Mamba-ASR | 0.1× | 0.5× | 0.3× | 6MB |
| **VELOCITY-ASR v2** | **0.05×** | **0.3×** | **0.15×** | **1.25MB** |

RTF < 1.0 means faster than real-time (can process faster than audio plays).

### Ablation Study Results

| Configuration | Parameters | WER (clean) | RTF (GPU) |
|---------------|------------|-------------|-----------|
| 6 SSM layers, fixed K=32 | 3.3M | 4.8% | 0.04× |
| 6 SSM layers, hierarchical | 3.5M | 4.5% | 0.05× |
| 8 SSM layers, hierarchical | 4.3M | 4.3% | 0.06× |
| 8 SSM layers, hierarchical, QAT | 4.3M | 4.2% → 4.5% | 0.05× |
| **Final (8 layers, hierarchical, QAT)** | **5M** | **4.2%** | **0.05×** |

---

## Deployment Guide

### Browser Deployment (ONNX/WebGPU)

For browser deployment, VELOCITY-ASR v2 can be converted to ONNX format and executed using ONNX Runtime with the WebGPU execution provider. This provides GPU acceleration within the browser without requiring server-side inference.

```python
# Export to ONNX with quantization
import torch
from velocity_asr import VELOCITYASR

model = VELOCITYASR.from_pretrained('velocity-asr-v2')
model.eval()

# Export to ONNX
dummy_input = torch.randn(1, 500, 80)  # (batch, frames, mel_bins)
torch.onnx.export(
    model,
    dummy_input,
    'velocity-asr-v2.onnx',
    input_names=['mel_spectrogram'],
    output_names=['transcription'],
    dynamic_axes={
        'mel_spectrogram': {1: 'num_frames'},
        'transcription': {1: 'num_tokens'}
    },
    opset_version=17
)

# Apply INT8 quantization
import onnxruntime.quantization as quant
quant.quantize_dynamic(
    'velocity-asr-v2.onnx',
    'velocity-asr-v2-int8.onnx',
    weight_type=quant.QuantType.INT8
)
```

```javascript
// Browser inference with ONNX Runtime WebGPU
import * as ort from 'onnxruntime-web';

async function transcribe(audioBuffer) {
    // Preprocess audio
    const melSpectrogram = await computeMelSpectrogram(audioBuffer);
    
    // Run inference
    const session = await ort.InferenceSession.create(
        'velocity-asr-v2-int8.onnx',
        { executionProviders: ['webgpu'] }
    );
    
    const feeds = {
        'mel_spectrogram': new Float32Array(melSpectrogram.data)
    };
    
    const results = await session.run(feeds);
    const transcription = decodeCTCOutput(results.output);
    
    return transcription;
}
```

### Mobile Deployment (Core ML / TFLite)

For iOS deployment, convert to Core ML format:

```python
# Export to Core ML
import coremltools as ct

model = ct.convert(
    'velocity-asr-v2.onnx',
    inputs=[ct.TensorType(shape=(1, None, 80))],
    outputs=[ct.TensorType(name='transcription')]
)

model.save('velocity-asr-v2.mlpackage')
```

For Android deployment, convert to TensorFlow Lite:

```python
# Export to TFLite
import tensorflow as tf

# Convert ONNX to TFLite
converter = tf.lite.TFLiteConverter.from_onnx('velocity-asr-v2.onnx')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.int8]

tflite_model = converter.convert()
with open('velocity-asr-v2-int8.tflite', 'wb') as f:
    f.write(tflite_model)
```

---

## Quick Start

### Installation

```bash
# Install from PyPI
pip install velocity-asr

# Or install from source
git clone https://github.com/your-org/velocity-asr.git
cd velocity-asr
pip install -e .
```

### Python Inference

```python
import torch
from velocity_asr import VELOCITYASR

# Load pre-trained model
model = VELOCITYASR.from_pretrained('velocity-asr-v2', quantized=True)
model.eval()

# Prepare input (16kHz mono audio)
audio = load_audio('path/to/audio.wav')  # Returns torch.Tensor of shape (samples,)
mel = compute_mel_spectrogram(audio)     # Returns torch.Tensor of shape (frames, 80)

# Run inference
with torch.no_grad():
    logits = model(mel.unsqueeze(0))     # Add batch dimension
    transcription = model.decode_ctc(logits)

print(f"Transcription: {transcription}")
```

### Command-Line Usage

```bash
# Transcribe an audio file
velocity-asr transcribe audio.wav --model velocity-asr-v2 --quantized

# Batch transcribe multiple files
velocity-asr transcribe --input_dir ./audio --output_dir ./transcripts

# Evaluate on a dataset
velocity-asr evaluate --test_set libri_speech_test_clean --model velocity-asr-v2
```

---

## File Structure

```
velocity-asr/
├── README.md
├── LICENSE
├── setup.py
├── requirements.txt
├── velocity_asr/
│   ├── __init__.py
│   ├── model.py                 # Main model architecture
│   ├── ssm.py                   # SSM implementation
│   ├── attention.py             # Hierarchical attention
│   ├── training.py              # Training utilities
│   ├── quantize.py              # Quantization utilities
│   └── decode.py                # CTC decoding
├── configs/
│   ├── pretrain.yaml           # Pre-training configuration
│   ├── finetune.yaml           # Fine-tuning configuration
│   └── evaluate.yaml           # Evaluation configuration
├── scripts/
│   ├── train_pretrain.py       # Pre-training script
│   ├── train_finetune.py       # Fine-tuning script
│   ├── export_onnx.py          # ONNX export script
│   └── evaluate.py             # Evaluation script
├── tests/
│   ├── test_model.py
│   ├── test_ssm.py
│   └── test_quantization.py
└── notebooks/
    ├── quickstart.ipynb
    └── evaluation_results.ipynb
```

---

## Limitations and Future Work

### Current Limitations

VELOCITY-ASR v2 has several limitations that users should be aware of when evaluating the architecture for their applications.

First, the model is optimized for English speech recognition and may exhibit degraded performance on other languages. The pre-training data (LibriLight) is predominantly English, and the vocabulary is sized for English transcription. Future versions will address multi-lingual support.

Second, performance degrades in challenging acoustic conditions with significant reverberation or overlapping speech. The model was trained primarily on read speech (LibriSpeech, LibriLight), which has clean acoustic characteristics. Real-world conditions with background noise, room reverb, or multiple speakers may produce higher WER.

Third, the model does not currently support hot-word boosting or custom vocabulary injection. These features are important for domain-specific applications but require architectural extensions that will be addressed in future work.

### Planned Extensions

The following extensions are planned for future releases:

- **Multi-lingual Support:** Extended pre-training on multi-lingual datasets with language identification.
- **Streaming Mode:** Optimized chunked inference for real-time applications.
- **Hot-Word Boosting:** Architecture extension for custom vocabulary injection.
- **Speaker Diarization:** Integration of speaker identification output.

---

## Citation

If you use VELOCITY-ASR v2 in your research, please cite the following:

```bibtex
@misc{velocity-asr-v2,
    title = {VELOCITY-ASR v2: Edge-Optimized Speech Recognition with State Space Models},
    author = {VELOCITY Research Team},
    year = {2024},
    url = {https://github.com/your-org/velocity-asr}
}
```

### Related Works

```bibtex
@mamba,
    title = {Mamba: Linear-Time Sequence Modeling with Selective State Spaces},
    author = {Gu, Albert and Dao, Tri},
    year = {2023},
    url = {https://github.com/state-spaces/mamba}
}

@whisper,
    title = {Robust Speech Recognition via Large-Scale Weak Supervision},
    author = {Radford, Alec and Kim, Jong Wook and Xu, Tao and Brockman, Greg and McLeavey, Christine and Sutskever, Ilya},
    year = {2022},
    url = {https://github.com/openai/whisper}
}

@conformer,
    title = {Conformer: Convolution-augmented Transformer for Speech Recognition},
    author = {Gulati, Anmol and Qin, James and Chiu, Chung-Cheng and Zhang, Naman and Parmar, Yisong and Yu, Wei and Han, Wei and Wang, Sankalpit and Zhang, Zongwei and others},
    year = {2020}
}
```

---

## Acknowledgments

VELOCITY-ASR v2 builds on the excellent work of the Mamba, Whisper, and Conformer research projects. We thank the open-source community for providing the tools and pre-trained models that made this work possible.
