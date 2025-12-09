import os
import requests
from huggingface_hub import hf_hub_download
import torch
from transformers import LlamaTokenizer, LlamaForCausalLM
import json

def check_system_requirements():
    """Check if system can handle LLaMA model"""
    import psutil
    
    total_ram = psutil.virtual_memory().total / (1024**3)  # GB
    available_ram = psutil.virtual_memory().available / (1024**3)  # GB
    
    print(f"Total RAM: {total_ram:.1f} GB")
    print(f"Available RAM: {available_ram:.1f} GB")
    
    if available_ram < 12:
        print("WARNING: LLaMA 7B model requires at least 12GB RAM")
        return False
    return True

def download_llama_model(model_size="7B"):
    """Download LLaMA model from HuggingFace"""
    model_name = f"meta-llama/Llama-2-{model_size.lower()}-chat-hf"
    
    print(f"Downloading {model_name}...")
    print("Note: You need HuggingFace access token for LLaMA models")
    
    # You'll need to get access token from HuggingFace
    # Instructions will be provided in setup guide
    
    try:
        tokenizer = LlamaTokenizer.from_pretrained(model_name)
        model = LlamaForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        
        # Save locally
        model_path = f"../models/llama-2-{model_size.lower()}-chat"
        tokenizer.save_pretrained(model_path)
        model.save_pretrained(model_path)
        
        print(f"Model saved to {model_path}")
        return model_path
        
    except Exception as e:
        print(f"Error downloading model: {e}")
        print("Falling back to alternative setup...")
        return None

def setup_llama_cpp():
    """Alternative setup using llama-cpp-python for efficiency"""
    print("Setting up llama-cpp-python for efficient inference...")
    
    # This will be installed via pip
    install_commands = [
        "pip install llama-cpp-python",
        "pip install huggingface-hub"
    ]
    
    for cmd in install_commands:
        print(f"Run: {cmd}")
    
    # Download GGML model (quantized version)
    model_url = "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML"
    print(f"Download quantized model from: {model_url}")
    
    return True

if __name__ == "__main__":
    print("LLaMA Setup Guide")
    print("================")
    
    # Check requirements
    if not check_system_requirements():
        print("Consider using smaller model or cloud computing")
    
    # Setup options
    print("\nSetup Options:")
    print("1. Full LLaMA model (requires HF token)")
    print("2. Quantized model with llama-cpp-python (recommended)")
    
    setup_llama_cpp()