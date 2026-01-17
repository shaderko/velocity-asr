#!/usr/bin/env python3
"""
Setup script for VELOCITY-ASR v2.

Installation:
    pip install -e .                    # Development install
    pip install -e ".[dev]"             # With development dependencies
    pip install velocity-asr            # From PyPI (when published)
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
if requirements_path.exists():
    requirements = [
        line.strip()
        for line in requirements_path.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
else:
    requirements = [
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "numpy>=1.21.0",
        "onnx>=1.14.0",
        "onnxruntime>=1.15.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
    ]

# Development dependencies
dev_requirements = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "black>=23.0.0",
    "isort>=5.12.0",
    "flake8>=6.0.0",
    "mypy>=1.0.0",
]

setup(
    name="velocity-asr",
    version="2.0.0",
    author="VELOCITY Research Team",
    author_email="research@velocity.ai",
    description="Edge-Optimized Speech Recognition with State Space Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/velocity-ai/velocity-asr",
    project_urls={
        "Bug Tracker": "https://github.com/velocity-ai/velocity-asr/issues",
        "Documentation": "https://github.com/velocity-ai/velocity-asr#readme",
        "Source Code": "https://github.com/velocity-ai/velocity-asr",
    },
    packages=find_packages(exclude=["tests", "tests.*", "scripts", "notebooks"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": dev_requirements,
        "all": requirements + dev_requirements,
    },
    entry_points={
        "console_scripts": [
            "velocity-asr=scripts.transcribe:main",
        ],
    },
    include_package_data=True,
    package_data={
        "velocity_asr": ["*.yaml", "*.json"],
    },
    zip_safe=False,
    keywords=[
        "speech recognition",
        "asr",
        "automatic speech recognition",
        "state space models",
        "ssm",
        "mamba",
        "edge ai",
        "deep learning",
        "pytorch",
        "onnx",
    ],
)
