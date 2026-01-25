#!/bin/bash
# Download ONNX models for Auralux voice pipeline
# Run this if models are missing or need updating

set -e
cd "$(dirname "$0")"

echo "Downloading Auralux ONNX models from HuggingFace..."

# Silero VAD v5 (2.2MB) - Voice Activity Detection
echo "[1/3] Silero VAD..."
curl -sL "https://huggingface.co/onnx-community/silero-vad/resolve/main/onnx/model.onnx" \
    -o silero_vad_v5.onnx

# HuBERT Base (92MB) - Self-Supervised Speech Encoder
echo "[2/3] HuBERT (quantized)..."
curl -sL "https://huggingface.co/Xenova/hubert-base-ls960/resolve/main/onnx/model_quantized.onnx" \
    -o hubert-soft-quantized.onnx

# Vocos (52MB) - Neural Vocoder
echo "[3/3] Vocos..."
curl -sL "https://huggingface.co/wetdog/vocos-mel-24khz-onnx/resolve/main/mel_spec_24khz.onnx" \
    -o vocos_q8.onnx

echo ""
echo "Done. Model sizes:"
ls -lh *.onnx
