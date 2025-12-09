import os
import json
import torch
import warnings
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from sentence_transformers import SentenceTransformer

warnings.filterwarnings('ignore')

@dataclass
class LlamaConfig:
    max_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 0.95
    repeat_penalty: float = 1.2
    top_k: int = 50
    echo: bool = False

class LlamaCustomerSupportModel:
    
    def __init__(self, model_type: str = "transformers", model_path: Optional[str] = None, force_llama: bool = True):
        self.model_type = model_type
        self.model_path = model_path
        self.force_llama = force_llama
        self.config = LlamaConfig()
        self.model = None
        self.tokenizer = None
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        self.categories = ["billing", "technical", "general_inquiry", "complaint", "compliment", "account"]
        self.priority_levels = ["high", "medium", "low"]
        self.sentiment_types = ["positive", "negative", "neutral"]
        
    def setup_model(self):
        config_path = "../outputs/llama_setup_config.json"
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            if config.get("model_name"):
                self.model_path = config["model_name"]
                self.model_type = config.get("recommended_mode", "transformers")
                print(f"Using LLaMA model from config: {self.model_path}")
        else:
            print("LLaMA configuration not found!")
            print("Please run LLAMA_SETUP.ipynb first to download and configure LLaMA for your system")
            raise FileNotFoundError("Run LLAMA_SETUP.ipynb first to set up LLaMA")
        
        self._setup_transformers()
    
    def _setup_transformers(self):
        import gc
        
        model_name = self.model_path
        print(f"Loading LLaMA model: {model_name}")
        
        # Force CPU for Intel integrated graphics systems
        device = "cpu"
        torch_dtype = torch.float32
        
        # Clean memory before loading
        gc.collect()
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load model with memory optimization for 12GB systems
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True
        )
        self.model = self.model.to(device)
        self.model.eval()
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"LLaMA model loaded on {device} for 12GB system")
    
    def create_system_message(self, task_type: str) -> str:
        
        if task_type == "classification":
            return """You are an AI assistant specialized in customer support ticket analysis. 
Analyze the support ticket and provide a JSON response with the following fields:
- category: one of [billing, technical, general_inquiry, complaint, compliment, account]
- priority: one of [high, medium, low] 
- estimated_hours: float between 0.5 and 48.0
- tags: list of 2-4 relevant keywords
- sentiment: one of [positive, negative, neutral]

Respond only with valid JSON format."""

        elif task_type == "response_generation":
            return """You are a professional customer support agent. Generate a helpful, empathetic, and solution-oriented response to the customer ticket. 
The response should be:
- Professional and courteous
- Specific to the customer's issue
- Include next steps or solutions
- Maintain appropriate tone based on sentiment"""
        
        return "You are a helpful AI assistant for customer support."
    
    def generate_classification(self, ticket_text: str) -> Dict:
        system_msg = self.create_system_message("classification")
        
        prompt = f"""<s>[INST] <<SYS>>
{system_msg}
<</SYS>>

Ticket: {ticket_text}

Analyze this ticket and respond with JSON: [/INST]"""

        return self._generate_with_transformers(prompt, "classification")
    
    def generate_response(self, ticket_text: str, sentiment: str = "neutral") -> str:
        system_msg = self.create_system_message("response_generation")
        
        prompt = f"""<s>[INST] <<SYS>>
{system_msg}
<</SYS>>

Customer Ticket: {ticket_text}
Customer Sentiment: {sentiment}

Generate a professional response: [/INST]"""

        return self._generate_with_transformers(prompt, "response")
    
    def _generate_with_transformers(self, prompt: str, task_type: str) -> Union[Dict, str]:
        
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("LLaMA model not loaded! Run setup_model() first.")
        
        device = next(self.model.parameters()).device
        inputs = self.tokenizer.encode(prompt, return_tensors='pt', truncation=True, max_length=1024)
        inputs = inputs.to(device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                do_sample=True if self.config.temperature > 0 else False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_text = generated_text[len(prompt):].strip()
        
        if task_type == "classification":
            return self._parse_classification_response(generated_text)
        else:
            return generated_text
    
    def _parse_classification_response(self, response_text: str) -> Dict:
        
        try:
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = response_text[start_idx:end_idx]
                parsed = json.loads(json_str)
                
                return self._validate_classification(parsed)
                
        except json.JSONDecodeError:
            pass
        except Exception:
            pass
        
        return {
            "category": "general_inquiry",
            "priority": "medium", 
            "estimated_hours": 2.0,
            "tags": ["support", "inquiry"],
            "sentiment": "neutral"
        }
    
    def _validate_classification(self, classification: Dict) -> Dict:
        
        validated = {}
        
        category = classification.get('category', 'general_inquiry').lower()
        validated['category'] = category if category in self.categories else 'general_inquiry'
        
        priority = classification.get('priority', 'medium').lower()
        validated['priority'] = priority if priority in self.priority_levels else 'medium'
        
        estimated_hours = classification.get('estimated_hours', 2.0)
        try:
            estimated_hours = float(estimated_hours)
            validated['estimated_hours'] = max(0.5, min(48.0, estimated_hours))
        except:
            validated['estimated_hours'] = 2.0
        
        tags = classification.get('tags', ['support', 'inquiry'])
        if isinstance(tags, list):
            validated['tags'] = tags[:4]
        else:
            validated['tags'] = ['support', 'inquiry']
        
        sentiment = classification.get('sentiment', 'neutral').lower()
        validated['sentiment'] = sentiment if sentiment in self.sentiment_types else 'neutral'
        
        return validated
    
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        return self.embedder.encode(texts)
    
    def process_single_ticket(self, ticket_text: str) -> Dict:
        
        classification = self.generate_classification(ticket_text)
        
        response = self.generate_response(ticket_text, classification['sentiment'])
        
        embedding = self.get_embeddings([ticket_text])[0]
        
        return {
            'ticket_text': ticket_text,
            'classification': classification,
            'generated_response': response,
            'embedding': embedding.tolist(),
            'model_type': self.model_type
        }
    
    def process_dataframe(self, df: pd.DataFrame, text_column: str = 'ticket_text') -> pd.DataFrame:
        
        results = []
        
        for idx, row in df.iterrows():
            ticket_text = row[text_column]
            result = self.process_single_ticket(ticket_text)
            results.append(result)
            
            if (idx + 1) % 10 == 0:
                print(f"Processed {idx + 1}/{len(df)} tickets")
        
        result_df = df.copy()
        
        classifications = [r['classification'] for r in results]
        result_df['predicted_category'] = [c['category'] for c in classifications]
        result_df['predicted_priority'] = [c['priority'] for c in classifications]
        result_df['predicted_eta_hours'] = [c['estimated_hours'] for c in classifications]
        result_df['predicted_sentiment'] = [c['sentiment'] for c in classifications]
        result_df['generated_tags'] = [c['tags'] for c in classifications]
        
        result_df['generated_response'] = [r['generated_response'] for r in results]
        
        return result_df

def download_llama_model(model_name: str = "NousResearch/Llama-2-7b-chat-hf") -> str:
    
    from huggingface_hub import hf_hub_download
    
    try:
        model_path = hf_hub_download(
            repo_id=model_name,
            filename="config.json",
            local_dir="../models",
            local_dir_use_symlinks=False
        )
        
        return model_path
        
    except Exception as e:
        raise

if __name__ == "__main__":
    model = LlamaCustomerSupportModel()
    model.setup_model()
    
    test_ticket = "My billing statement shows duplicate charges for this month"
    result = model.process_single_ticket(test_ticket)
    
    print(json.dumps(result, indent=2, default=str))