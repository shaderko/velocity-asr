import torch
import sys

# Test imports
try:
    from velocity_asr import VELOCITYASR, VelocityASRConfig, ctc_greedy_decode
    print("✓ Imports successful")
except Exception as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

# Test model creation
try:
    config = VelocityASRConfig()
    model = VELOCITYASR(config)
    num_params = model.count_parameters()
    print(f"✓ Model created: {num_params:,} parameters")
except Exception as e:
    print(f"✗ Model creation error: {e}")
    sys.exit(1)

# Test forward pass
try:
    batch_size = 2
    num_frames = 500
    mel_bins = 80

    x = torch.randn(batch_size, num_frames, mel_bins)
    model.eval()

    with torch.no_grad():
        logits = model(x)

    expected_output_len = (num_frames + 1) // 2  # stride 2

    print(f"✓ Forward pass: input {x.shape} -> output {logits.shape}")
    print(f"  Expected output length: ~{expected_output_len}, got: {logits.shape[1]}")

except Exception as e:
    print(f"✗ Forward pass error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test decoding
try:
    decoded = ctc_greedy_decode(logits)
    print(f"✓ CTC decoding: {len(decoded)} sequences decoded")
except Exception as e:
    print(f"✗ Decoding error: {e}")
    sys.exit(1)

print()
print("All tests passed!")
