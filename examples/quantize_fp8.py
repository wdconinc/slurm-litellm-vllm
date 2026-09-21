"""
Utility script to permanently quantize a BF16/FP16 model to FP8 on disk.
This prevents vLLM from having to perform dynamic quantization on-the-fly during every startup.

Prerequisites:
    pip install llmcompressor
"""

import argparse
from llmcompressor.transformers import SparseAutoModelForCausalLM, SparseAutoTokenizer
from llmcompressor.transformers import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

def main():
    parser = argparse.ArgumentParser(description="Quantize a model to FP8 and save to disk.")
    parser.add_argument("--model-id", type=str, required=True, help="Hugging Face Model ID (e.g., sahilchachra/Leanstral-1.5-119B-A6B-BF16)")
    parser.add_argument("--save-path", type=str, required=True, help="Local directory to save the FP8 checkpoint")
    args = parser.parse_args()

    print(f"Loading {args.model_id}...")
    # Load model and tokenizer
    tokenizer = SparseAutoTokenizer.from_pretrained(args.model_id)
    model = SparseAutoModelForCausalLM.from_pretrained(
        args.model_id,
        device_map="auto",
        torch_dtype="auto",
    )

    print("Configuring FP8 QuantizationModifier...")
    # Configure the simple FP8 quantization modifier (No calibration data needed for dynamic weight-only or basic W8A8)
    recipe = QuantizationModifier(
        targets="Linear",
        scheme="FP8_DYNAMIC",
        ignore=["lm_head"]
    )

    print("Applying quantization...")
    # Apply the quantization recipe to the model
    oneshot(
        model=model,
        recipe=recipe,
    )

    print(f"Saving quantized model to {args.save_path}...")
    # Save the quantized model and tokenizer
    model.save_pretrained(args.save_path)
    tokenizer.save_pretrained(args.save_path)
    
    print("Done! You can now update config/models.yaml 'fullname' to point to this local directory.")

if __name__ == "__main__":
    main()
