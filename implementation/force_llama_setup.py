#!/usr/bin/env python3

import subprocess
import sys
import os
import json
import time
from pathlib import Path

def install_dependencies():
    dependencies = ['transformers', 'torch', 'huggingface_hub', 'sentencepiece', 'accelerate']
    
    for dep in dependencies:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', dep])
        except subprocess.CalledProcessError:
            pass
    return True

def download_llama_model():
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "huggingface_hub"])
        from huggingface_hub import hf_hub_download
    
    model_repo = "NousResearch/Llama-2-7b-chat-hf"
    model_dir = Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    config_path = hf_hub_download(
        repo_id=model_repo,
        filename="config.json",
        cache_dir=str(model_dir)
    )
    
    tokenizer_path = hf_hub_download(
        repo_id=model_repo,
        filename="tokenizer.model",
        cache_dir=str(model_dir)
    )
    
    return {
        "model_name": model_repo,
        "config_path": config_path,
        "tokenizer_path": tokenizer_path
    }

def test_llama_model(model_info):
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        
        model_name = model_info["model_name"]
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        
        prompt = "Customer support ticket: My billing shows duplicate charges. Category:"
        inputs = tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                inputs['input_ids'],
                max_length=inputs['input_ids'].shape[1] + 20,
                temperature=0.7,
                do_sample=True
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return True, "transformers"
        
    except Exception as e:
        return False, "failed"

def save_config(model_info, working, method):
    config_dir = Path("outputs")
    config_dir.mkdir(exist_ok=True)
    
    config = {
        "model_name": model_info["model_name"],
        "config_path": model_info["config_path"],
        "tokenizer_path": model_info["tokenizer_path"],
        "force_llama": True,
        "setup_complete": True,
        "test_success": working,
        "recommended_mode": method,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(config_dir / "llama_setup_config.json", "w") as f:
        json.dump(config, f, indent=2)

def main():
    install_dependencies()
    
    try:
        model_info = download_llama_model()
    except Exception as e:
        return
    
    working, method = test_llama_model(model_info)
    
    save_config(model_info, working, method)
    
    if working:
        print("LLaMA setup complete")
    else:
        print("Setup completed with issues")

if __name__ == "__main__":
    main()