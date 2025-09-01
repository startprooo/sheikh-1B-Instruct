import gradio as gr
from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer
import torch
import os
import psutil
import shutil
from tqdm import tqdm
from accelerate import cpu_offload
from torch.nn import functional as F

def check_system_resources():
    """Check available system resources"""
    mem = psutil.virtual_memory()
    disk = shutil.disk_usage("/workspaces")
    
    print("\n=== System Resources ===")
    print(f"RAM: {mem.available/1024/1024/1024:.1f}GB available out of {mem.total/1024/1024/1024:.1f}GB")
    print(f"Disk: {disk.free/1024/1024/1024:.1f}GB free out of {disk.total/1024/1024/1024:.1f}GB")
    print(f"CPU Cores: {psutil.cpu_count()}")
    print(f"GPU: {'Available' if torch.cuda.is_available() else 'Not available'}")
    
    if mem.available < 8 * 1024 * 1024 * 1024:  # 8GB
        print("\n⚠️ Warning: Less than 8GB RAM available. Model loading may be slow.")
    if disk.free < 20 * 1024 * 1024 * 1024:  # 20GB
        print("\n⚠️ Warning: Less than 20GB disk space available.")
    print("=====================\n")

def load_model(model_path):
    """Load the model and tokenizer from the specified path with optimizations for CPU"""
    check_system_resources()
    print(f"Loading model from {model_path}...")
    
    # Load the LLaMA config
    config_path = os.path.join(model_path, "sheikh_config.json")
    config = LlamaConfig.from_pretrained(config_path)
    
    # Check if we have pretrained weights
    if os.path.exists(os.path.join(model_path, "pytorch_model.bin")) or \
       any(f.endswith(".safetensors") for f in os.listdir(model_path)):
        # Load with 8-bit quantization for CPU
        model = LlamaForCausalLM.from_pretrained(
            model_path,
            config=config,
            load_in_8bit=True,  # Enable 8-bit quantization
            device_map="auto",
            torch_dtype=torch.float32,  # Use float32 for CPU
            low_cpu_mem_usage=True
        )
    else:
        print("No pretrained weights found. Initializing fresh model...")
        model = LlamaForCausalLM(config)
        model = model.to(dtype=torch.float32)
    
    # Initialize tokenizer
    try:
        tokenizer = LlamaTokenizer.from_pretrained(model_path)
    except:
        print("Tokenizer not found in model path. Using default LLaMA tokenizer...")
        tokenizer = LlamaTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
        tokenizer.resize_token_embeddings(config.vocab_size)
    
    return model, tokenizer

def generate_response(message, model, tokenizer):
    """Generate a response from the model using chat template"""
    messages = [{"role": "user", "content": message}]
    
    # Apply chat template with instruction formatting
    inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True
    )
    
    # Move inputs to the same device as the model
    inputs = inputs.to(model.device)
    
    # Generate response using model's generation config
    outputs = model.generate(
        inputs,
        max_new_tokens=512,  # Increased for longer responses
        pad_token_id=tokenizer.pad_token_id,
        # Other parameters will come from generation_config.json
    )
    
    # Decode and clean up the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the assistant's response after the last [/INST] tag
    response = response.split("[/INST]")[-1].strip()
    return response

def create_demo(model_path):
    """Create and launch the Gradio demo"""
    print("Loading model (this may take a few minutes on CPU)...")
    model, tokenizer = load_model(model_path)
    print("Model loaded successfully!")

    # Keep conversation history
    conversation_history = []

    def predict(message, history):
        # Update history with user message
        history = history or []
        
        # Generate response with progress bar
        with tqdm(total=1, desc="Generating response") as pbar:
            response = generate_response(message, model, tokenizer)
            pbar.update(1)
        
        history.append((message, response))
        return history, history
    
    # Create a chat interface
    demo = gr.ChatInterface(
        fn=predict,
        title="Sheikh-1B-Instruct Demo 🤖",
        description="""Chat with Sheikh-1B-Instruct - A lightweight instruction-tuned LLM
        \nRunning in CPU mode with 8-bit quantization for optimal performance.
        \nNote: First response may take 15-30 seconds on CPU.""",
        examples=[
            ["Explain the difference between AI and Machine Learning"],
            ["Write a poem about sunset"],
            ["How does photosynthesis work?"],
            ["Can you help me debug this Python code? def factorial(n): return n * factorial(n-1)"],
            ["What is the capital of France and what's special about it?"]
        ],
        retry_btn="Regenerate Response 🔄",
        undo_btn="Undo Last Message ↩️",
        clear_btn="Clear Chat 🗑️"
    )

    # Launch the app
    demo.launch(server_name="0.0.0.0", share=True)

if __name__ == "__main__":
    MODEL_PATH = "/workspaces/sheikh-1B-Instruct/model"
    create_demo(MODEL_PATH)
