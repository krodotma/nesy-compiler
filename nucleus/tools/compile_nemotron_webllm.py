#!/usr/bin/env python3
"""
Compile Nemotron models for WebLLM/MLC

This script compiles NVIDIA Nemotron models for browser-based inference via WebLLM.
Nemotron-Mini-4B uses standard transformer architecture (GQA + RoPE) compatible with MLC.

Requirements:
  - mlc_llm (pip install mlc-llm)
  - TVM compiler (follow https://llm.mlc.ai/docs/install/tvm.html)
  - WASM build environment for WebGPU target

Usage:
  python compile_nemotron_webllm.py --model nvidia/Nemotron-Mini-4B-Instruct --quant q4f16_1
  python compile_nemotron_webllm.py --model nvidia/Nemotron-Mini-4B-Instruct --quant q4f32_1 --upload

Models:
  - nvidia/Nemotron-Mini-4B-Instruct (4B params, standard transformer)
  - Note: Nemotron-3-Nano uses Mamba-Transformer hybrid - NOT YET SUPPORTED by MLC

References:
  - https://llm.mlc.ai/docs/compilation/compile_models.html
  - https://huggingface.co/nvidia/Nemotron-Mini-4B-Instruct
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Nemotron models with their MLC-compatible conversation templates
NEMOTRON_MODELS = {
    "nvidia/Nemotron-Mini-4B-Instruct": {
        "conv_template": "chatml",  # Nemotron uses ChatML format
        "context_window": 4096,
        "prefill_chunk_size": 1024,
        "architecture": "nemotron",  # Maps to Llama-style in MLC
    },
    # Future: Add more Nemotron variants as MLC adds support
}

QUANTIZATION_OPTIONS = [
    "q0f16",    # No quantization, FP16
    "q0f32",    # No quantization, FP32
    "q4f16_1",  # 4-bit quantization, FP16 (recommended for WebGPU)
    "q4f32_1",  # 4-bit quantization, FP32
    "q3f16_1",  # 3-bit quantization, FP16 (smaller but lower quality)
]


def check_prerequisites() -> bool:
    """Check if MLC LLM tools are installed."""
    try:
        result = subprocess.run(
            ["mlc_llm", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            print("ERROR: mlc_llm not found. Install with: pip install mlc-llm")
            return False
    except FileNotFoundError:
        print("ERROR: mlc_llm not found. Install with: pip install mlc-llm")
        return False
    except Exception as e:
        print(f"ERROR: Failed to check mlc_llm: {e}")
        return False

    # Check TVM
    try:
        import tvm
        print(f"✓ TVM found: {tvm.__file__}")
    except ImportError:
        print("ERROR: TVM not found. Follow: https://llm.mlc.ai/docs/install/tvm.html")
        return False

    return True


def clone_model(model_id: str, output_dir: Path) -> Path:
    """Clone model from HuggingFace."""
    model_name = model_id.split("/")[-1]
    model_path = output_dir / model_name

    if model_path.exists():
        print(f"✓ Model already exists: {model_path}")
        return model_path

    print(f"Cloning {model_id} from HuggingFace...")
    try:
        subprocess.run(
            ["git", "lfs", "install"],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "clone", f"https://huggingface.co/{model_id}", str(model_path)],
            check=True,
        )
        print(f"✓ Cloned to {model_path}")
        return model_path
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Failed to clone model: {e}")
        sys.exit(1)


def convert_weights(
    model_path: Path,
    output_dir: Path,
    model_name: str,
    quant: str,
) -> Path:
    """Convert and quantize model weights."""
    output_path = output_dir / f"{model_name}-{quant}-MLC"

    if output_path.exists():
        print(f"✓ Weights already converted: {output_path}")
        return output_path

    print(f"Converting weights with {quant} quantization...")
    try:
        subprocess.run(
            [
                "mlc_llm", "convert_weight",
                str(model_path),
                "--quantization", quant,
                "-o", str(output_path),
            ],
            check=True,
        )
        print(f"✓ Weights converted to {output_path}")
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Weight conversion failed: {e}")
        sys.exit(1)


def generate_config(
    model_path: Path,
    output_path: Path,
    model_config: dict,
    quant: str,
) -> None:
    """Generate MLC chat configuration."""
    config_file = output_path / "mlc-chat-config.json"

    if config_file.exists():
        print(f"✓ Config already exists: {config_file}")
        return

    print("Generating MLC configuration...")
    cmd = [
        "mlc_llm", "gen_config",
        str(model_path),
        "--quantization", quant,
        "--conv-template", model_config["conv_template"],
        "--context-window-size", str(model_config["context_window"]),
        "--prefill-chunk-size", str(model_config["prefill_chunk_size"]),
        "-o", str(output_path),
    ]

    try:
        subprocess.run(cmd, check=True)
        print(f"✓ Config generated: {config_file}")
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Config generation failed: {e}")
        sys.exit(1)


def compile_webgpu(output_path: Path, model_name: str, quant: str) -> Path:
    """Compile model for WebGPU target."""
    config_file = output_path / "mlc-chat-config.json"
    wasm_file = output_path.parent / "libs" / f"{model_name}-{quant}-webgpu.wasm"

    if wasm_file.exists():
        print(f"✓ WASM already compiled: {wasm_file}")
        return wasm_file

    wasm_file.parent.mkdir(parents=True, exist_ok=True)

    print("Compiling for WebGPU (this may take a while)...")
    try:
        subprocess.run(
            [
                "mlc_llm", "compile",
                str(config_file),
                "--device", "webgpu",
                "-o", str(wasm_file),
            ],
            check=True,
        )
        print(f"✓ WebGPU WASM compiled: {wasm_file}")
        return wasm_file
    except subprocess.CalledProcessError as e:
        print(f"ERROR: WebGPU compilation failed: {e}")
        print("Note: WebGPU compilation requires WASM build environment.")
        print("See: https://llm.mlc.ai/docs/install/tvm.html#install-wasm-build-environment")
        sys.exit(1)


def generate_webllm_config(
    model_name: str,
    quant: str,
    output_path: Path,
    hf_repo: str | None = None,
) -> dict:
    """Generate WebLLM model config for registration."""
    model_id = f"{model_name}-{quant}-MLC"

    config = {
        "model_id": model_id,
        "model_lib": f"https://huggingface.co/{hf_repo or 'pluribus/webllm-models'}/resolve/main/{model_id}-webgpu.wasm",
        "vram_required_MB": 2500 if "4B" in model_name else 1500,
        "low_resource_required": False,
        "overrides": {
            "context_window_size": 4096,
        },
    }

    config_path = output_path / "webllm-model-config.json"
    import json
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"✓ WebLLM config written: {config_path}")
    return config


def main():
    parser = argparse.ArgumentParser(
        description="Compile Nemotron models for WebLLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model",
        default="nvidia/Nemotron-Mini-4B-Instruct",
        choices=list(NEMOTRON_MODELS.keys()),
        help="Nemotron model to compile",
    )
    parser.add_argument(
        "--quant",
        default="q4f16_1",
        choices=QUANTIZATION_OPTIONS,
        help="Quantization method (default: q4f16_1)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./mlc-nemotron"),
        help="Output directory for compiled model",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload to HuggingFace after compilation",
    )
    parser.add_argument(
        "--hf-repo",
        default=None,
        help="HuggingFace repo for upload (default: pluribus/webllm-models)",
    )
    parser.add_argument(
        "--skip-compile",
        action="store_true",
        help="Skip WebGPU compilation (useful if WASM env not set up)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Nemotron WebLLM Compiler")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Quantization: {args.quant}")
    print(f"Output: {args.output_dir}")
    print("=" * 60)

    if not check_prerequisites():
        sys.exit(1)

    model_config = NEMOTRON_MODELS[args.model]
    model_name = args.model.split("/")[-1]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Clone model
    model_path = clone_model(args.model, args.output_dir)

    # Step 2: Convert weights
    output_path = convert_weights(model_path, args.output_dir, model_name, args.quant)

    # Step 3: Generate config
    generate_config(model_path, output_path, model_config, args.quant)

    # Step 4: Compile for WebGPU
    if not args.skip_compile:
        wasm_file = compile_webgpu(output_path, model_name, args.quant)

    # Step 5: Generate WebLLM registration config
    webllm_config = generate_webllm_config(model_name, args.quant, output_path, args.hf_repo)

    print("\n" + "=" * 60)
    print("COMPILATION COMPLETE")
    print("=" * 60)
    print(f"\nTo use in WebLLM, add this to your model config:")
    print(f"  Model ID: {webllm_config['model_id']}")
    print(f"\nNext steps:")
    print(f"  1. Upload {output_path} to HuggingFace")
    print(f"  2. Upload WASM to GitHub releases or HuggingFace")
    print(f"  3. Update webllm-enhanced.ts MODEL_REGISTRY with new model")

    if args.upload:
        print("\n[Upload not yet implemented - manual upload required]")


if __name__ == "__main__":
    main()
