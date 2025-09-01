"""
Quantize Sheikh-1B-Instruct model for CPU inference.
This script creates INT8 and INT4 versions of the model for efficient CPU deployment.
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import HfApi
import shutil
from tqdm import tqdm

# Model paths
BASE_PATH = "/workspaces/sheikh-1B-Instruct/model"
TMP_PATH = "/tmp/sheikh-quantize"
FINAL_PATH = "/workspaces/sheikh-1B-Instruct/quantized"

# Temporary paths for quantization
TMP_INT8_PATH = os.path.join(TMP_PATH, "sheikh-1B-instruct-int8")
TMP_INT4_PATH = os.path.join(TMP_PATH, "sheikh-1B-instruct-int4")

# Final paths for storage
INT8_PATH = os.path.join(FINAL_PATH, "sheikh-1B-instruct-int8")
INT4_PATH = os.path.join(FINAL_PATH, "sheikh-1B-instruct-int4")

def check_space():
    """Check if we have enough disk space in /tmp and /workspaces"""
    tmp_space = shutil.disk_usage("/tmp").free / (1024**3)  # GB
    work_space = shutil.disk_usage("/workspaces").free / (1024**3)  # GB
    
    print(f"\nFree space in /tmp: {tmp_space:.1f}GB")
    print(f"Free space in /workspaces: {work_space:.1f}GB")
    
    # Need 30GB in /tmp for intermediate files, 10GB in /workspaces for final models
    return tmp_space > 30 and work_space > 10

def quantize_model(model_path, tmp_path, final_path, bits=8):
    """Quantize model to specified bit width using temporary storage"""
    print(f"\n🔧 Quantizing model to INT{bits}...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Load and quantize model
    load_kwargs = {
        "load_in_8bit": bits == 8,
        "load_in_4bit": bits == 4,
        "device_map": "cpu",
        "torch_dtype": torch.float32,
        "low_cpu_mem_usage": True
    }
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        **load_kwargs
    )
    
    # Save to temporary location first
    print(f"📦 Saving quantized model to temporary location...")
    model.save_pretrained(tmp_path)
    tokenizer.save_pretrained(tmp_path)
    
    # Clear memory
    del model
    torch.cuda.empty_cache()
    
    # Move to final location
    print(f"📦 Moving to final location: {final_path}")
    if os.path.exists(final_path):
        shutil.rmtree(final_path)
    shutil.copytree(tmp_path, final_path)
    shutil.rmtree(tmp_path)  # Clean up temp files

def create_model_card(repo_path):
    """Create a model card README.md"""
    readme = """---
language:
- en
license: apache-2.0
tags:
- text-generation
- instruct
- cpu-inference
pipeline_tag: text-generation
library_name: transformers
base_model: sheikh-1B-instruct
---

# Sheikh-1B-Instruct

**Sheikh-1B-Instruct** is a 1.1B parameter instruction-tuned language model optimized for CPU inference.

## Model Details

- **Architecture**: LLaMA-style with 32 layers
- **Parameters**: 1.1B
- **Context Length**: 32,768 tokens
- **Training Format**: Native BF16 with instruction tuning
- **Quantized Variants**: INT8, INT4 for efficient CPU deployment
- **Languages**: English

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# For CPU inference, use INT8 version
model = AutoModelForCausalLM.from_pretrained(
    "startprooo/sheikh-1B-instruct-int8",
    device_map="cpu",
    low_cpu_mem_usage=True
)
tokenizer = AutoTokenizer.from_pretrained("startprooo/sheikh-1B-instruct-int8")

# Simple generation
prompt = "Explain quantum computing to a 5 year old:"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# Chat completion
messages = [{"role": "user", "content": "What is the capital of France?"}]
chat_input = tokenizer.apply_chat_template(messages, return_tensors="pt")
chat_output = model.generate(chat_input, max_new_tokens=200)
print(tokenizer.decode(chat_output[0], skip_special_tokens=True))
```

## Quantized Versions

- **INT8**: ~600MB memory usage
- **INT4**: ~350MB memory usage

## Limitations

- No sliding window attention (full attention only)
- May struggle with complex reasoning
- English-focused training
- Not suitable for production/critical applications

## License

Apache 2.0
"""
    
    with open(os.path.join(repo_path, "README.md"), "w") as f:
        f.write(readme)

def main():
    """Main quantization pipeline"""
    if not check_space():
        print("❌ Not enough disk space! Need 30GB in /tmp and 10GB in /workspaces")
        return
    
    # Create directories
    os.makedirs(TMP_PATH, exist_ok=True)
    os.makedirs(FINAL_PATH, exist_ok=True)
    
    # Clean up any previous temporary files
    if os.path.exists(TMP_INT8_PATH):
        shutil.rmtree(TMP_INT8_PATH)
    if os.path.exists(TMP_INT4_PATH):
        shutil.rmtree(TMP_INT4_PATH)
    
    try:
        # Quantize to INT8
        quantize_model(BASE_PATH, TMP_INT8_PATH, INT8_PATH, bits=8)
        
        # Quantize to INT4
        quantize_model(BASE_PATH, TMP_INT4_PATH, INT4_PATH, bits=4)
    finally:
        # Clean up temporary files
        print("\n🧹 Cleaning up temporary files...")
        if os.path.exists(TMP_PATH):
            shutil.rmtree(TMP_PATH)
    
    # Create model cards
    create_model_card(INT8_PATH)
    create_model_card(INT4_PATH)
    
    print("\n✅ Quantization complete! Next steps:")
    print("1. Test the quantized models")
    print("2. Upload to Hugging Face Hub")
    print("3. Update documentation with benchmarks")

if __name__ == "__main__":
    main()
