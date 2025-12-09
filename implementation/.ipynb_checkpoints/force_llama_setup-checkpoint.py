#!/usr/bin/env python3
"""
FORCE LLaMA Setup Script
Alternative to the Jupyter notebook if you have issues opening it
"""

import subprocess
import sys
import os
import json
import time
from pathlib import Path

def install_llama_cpp():
    """Try to install llama-cpp-python"""
    print("🔧 Force installing llama-cpp-python...")
    
    methods = [
        "pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu",
        "pip install llama-cpp-python --force-reinstall --no-cache-dir",
        "pip install llama-cpp-python>=0.2.11"
    ]
    
    for i, method in enumerate(methods, 1):
        print(f"\n🔧 Trying method {i}: {method}")
        try:
            subprocess.run(method, shell=True, check=True)
            print(f"✅ Method {i} succeeded!")
            return True
        except subprocess.CalledProcessError:
            print(f"❌ Method {i} failed")
            continue
    
    print("⚠️  All installation methods failed")
    return False

def download_llama_model():
    """Download LLaMA 7B model"""
    print("\n🦙 FORCE DOWNLOADING LLaMA 7B MODEL")
    print("=" * 60)
    
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Installing huggingface_hub...")
        subprocess.run([sys.executable, "-m", "pip", "install", "huggingface_hub"])
        from huggingface_hub import hf_hub_download
    
    model_config = {
        "repo_id": "TheBloke/Llama-2-7B-Chat-GGUF",
        "filename": "llama-2-7b-chat.Q4_K_M.gguf",
        "size_gb": 3.8
    }
    
    model_dir = Path("models/llama")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = model_dir / model_config["filename"]
    
    if model_path.exists():
        print(f"✅ LLaMA model already exists: {model_path}")
        print(f"📊 File size: {model_path.stat().st_size / (1024**3):.1f} GB")
        return str(model_path)
    
    print(f"⬇️  Downloading LLaMA 7B model...")
    print(f"   Repository: {model_config['repo_id']}")
    print(f"   File: {model_config['filename']}")
    print(f"   Size: ~{model_config['size_gb']} GB")
    print(f"   This will take 10-30 minutes...")
    
    start_time = time.time()
    
    try:
        downloaded_path = hf_hub_download(
            repo_id=model_config["repo_id"],
            filename=model_config["filename"],
            local_dir=str(model_dir),
            local_dir_use_symlinks=False
        )
        
        elapsed = time.time() - start_time
        print(f"\n✅ Download completed in {elapsed/60:.1f} minutes!")
        print(f"   Model saved to: {downloaded_path}")
        
        return downloaded_path
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        raise

def test_llama_model(model_path):
    """Test LLaMA model"""
    print("\n🧪 Testing LLaMA model...")
    
    # Try llama-cpp-python first
    try:
        from llama_cpp import Llama
        
        print("⚡ Loading LLaMA model...")
        llama = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_batch=512,
            verbose=False
        )
        
        print("✅ LLaMA model loaded successfully!")
        
        # Test prompt
        test_prompt = """<s>[INST] <<SYS>>
You are an AI assistant specialized in customer support ticket analysis. 
Analyze the support ticket and provide a JSON response with category, priority, and sentiment.
<</SYS>>

Ticket: My billing statement shows duplicate charges

Respond with JSON: [/INST]"""
        
        response = llama(
            test_prompt,
            max_tokens=100,
            temperature=0.1,
            echo=False
        )
        
        generated_text = response['choices'][0]['text'].strip()
        
        print("\n🎉 LLaMA Response:")
        print("=" * 40)
        print(generated_text)
        print("=" * 40)
        
        del llama
        
        print("\n✅ LLaMA test successful!")
        return True, "llama-cpp"
        
    except ImportError:
        print("❌ llama-cpp-python not available")
        return False, "not_available"
    except Exception as e:
        print(f"❌ LLaMA test failed: {e}")
        return False, "failed"

def save_config(model_path, working, method):
    """Save configuration"""
    config_dir = Path("outputs")
    config_dir.mkdir(exist_ok=True)
    
    config = {
        "forced_llama": True,
        "llama_model_path": model_path,
        "llama_working": working,
        "test_method": method,
        "model_size": "7B",
        "download_forced": True,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(config_dir / "forced_llama_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"💾 Configuration saved to: {config_dir}/forced_llama_config.json")

def main():
    """Main setup function"""
    print("🦙 FORCE LLaMA Setup Script")
    print("=" * 50)
    
    # Step 1: Try to install llama-cpp-python
    llama_cpp_installed = install_llama_cpp()
    
    # Step 2: Download LLaMA model
    try:
        model_path = download_llama_model()
    except Exception as e:
        print(f"❌ Failed to download model: {e}")
        return
    
    # Step 3: Test LLaMA model
    working, method = test_llama_model(model_path)
    
    # Step 4: Save configuration
    save_config(model_path, working, method)
    
    # Final status
    if working:
        print("\n🎉 SUCCESS: FORCE LLaMA setup complete!")
        print("✅ LLaMA 7B model downloaded and tested")
        print("✅ Ready for customer support processing")
        print("\n🚀 Next steps:")
        print("1. Open Jupyter: jupyter notebook")
        print("2. Run: 02_model_setup_and_configuration.ipynb")
    else:
        print("\n⚠️  LLaMA model downloaded but testing had issues")
        print("✅ Model ready for use in notebooks")
        print("The system will attempt to use it anyway")
    
    print("\n🦙 FORCE LLaMA: No fallbacks, LLaMA only!")

if __name__ == "__main__":
    main()