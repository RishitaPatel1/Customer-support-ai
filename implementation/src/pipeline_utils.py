import pandas as pd
import numpy as np
import json
import joblib
from datetime import datetime
from sentence_transformers import SentenceTransformer
from textblob import TextBlob
import re
import warnings
warnings.filterwarnings('ignore')

class ModelLoader:
    def __init__(self, models_path='../models'):
        self.models_path = models_path
        
    def load_complete_pipeline(self):
        try:
            return joblib.load(f'{self.models_path}/complete_pipeline.joblib')
        except FileNotFoundError:
            return None
    
    def load_individual_components(self):
        components = {}
        try:
            components['classifier'] = joblib.load(f'{self.models_path}/ticket_classifier.joblib')
            components['eta_predictor'] = joblib.load(f'{self.models_path}/eta_predictor.joblib')
            components['sentiment_analyzer'] = joblib.load(f'{self.models_path}/sentiment_analyzer.joblib')
            components['response_generator'] = joblib.load(f'{self.models_path}/response_generator.joblib')
            return components
        except FileNotFoundError:
            return {}

class DataValidator:
    @staticmethod
    def validate_input_csv(file_path):
        try:
            df = pd.read_csv(file_path)
            required_columns = ['text']
            
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                return False, f"Missing required columns: {missing_columns}"
            
            empty_texts = df['text'].isna().sum()
            if empty_texts > 0:
                return False, f"Found {empty_texts} empty text entries"
            
            return True, "Input validation successful"
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    @staticmethod
    def validate_output_quality(results_df):
        quality_checks = {}
        
        quality_checks['total_processed'] = len(results_df)
        quality_checks['successful_predictions'] = len(results_df[results_df['processing_status'] == 'success'])
        quality_checks['error_rate'] = (quality_checks['total_processed'] - quality_checks['successful_predictions']) / quality_checks['total_processed']
        
        if 'category_confidence' in results_df.columns:
            quality_checks['avg_category_confidence'] = results_df['category_confidence'].mean()
        
        if 'predicted_eta_hours' in results_df.columns:
            quality_checks['avg_eta_hours'] = results_df['predicted_eta_hours'].mean()
            quality_checks['eta_range'] = [results_df['predicted_eta_hours'].min(), results_df['predicted_eta_hours'].max()]
        
        return quality_checks

class PerformanceMonitor:
    def __init__(self):
        self.metrics = {}
        self.start_time = None
    
    def start_monitoring(self):
        self.start_time = datetime.now()
    
    def record_metric(self, metric_name, value):
        self.metrics[metric_name] = value
    
    def get_processing_summary(self, tickets_processed):
        if self.start_time:
            total_time = (datetime.now() - self.start_time).total_seconds()
            self.metrics['total_processing_time_seconds'] = total_time
            self.metrics['tickets_per_second'] = tickets_processed / total_time if total_time > 0 else 0
            self.metrics['average_time_per_ticket'] = total_time / tickets_processed if tickets_processed > 0 else 0
        
        return self.metrics

class ResultsAnalyzer:
    @staticmethod
    def generate_summary_report(results_df):
        report = {
            'processing_summary': {
                'total_tickets': len(results_df),
                'successful_processing': len(results_df[results_df['processing_status'] == 'success']),
                'processing_success_rate': len(results_df[results_df['processing_status'] == 'success']) / len(results_df)
            }
        }
        
        if 'predicted_category' in results_df.columns:
            report['category_distribution'] = results_df['predicted_category'].value_counts().to_dict()
        
        if 'adjusted_priority' in results_df.columns:
            report['priority_distribution'] = results_df['adjusted_priority'].value_counts().to_dict()
        
        if 'predicted_sentiment' in results_df.columns:
            report['sentiment_distribution'] = results_df['predicted_sentiment'].value_counts().to_dict()
        
        if 'eta_category' in results_df.columns:
            report['eta_distribution'] = results_df['eta_category'].value_counts().to_dict()
        
        if 'predicted_eta_hours' in results_df.columns:
            report['eta_statistics'] = {
                'mean_eta_hours': float(results_df['predicted_eta_hours'].mean()),
                'median_eta_hours': float(results_df['predicted_eta_hours'].median()),
                'std_eta_hours': float(results_df['predicted_eta_hours'].std())
            }
        
        return report
    
    @staticmethod
    def export_results(results_df, output_path, include_summary=True):
        results_df.to_csv(output_path, index=False)
        
        if include_summary:
            summary = ResultsAnalyzer.generate_summary_report(results_df)
            summary_path = output_path.replace('.csv', '_summary.json')
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
        
        return output_path

class ConfigurationManager:
    DEFAULT_CONFIG = {
        'processing': {
            'batch_size': 32,
            'max_text_length': 512,
            'enable_progress_tracking': True
        },
        'models': {
            'embedding_model': 'all-MiniLM-L6-v2',
            'classification_threshold': 0.5,
            'eta_bounds': [0.1, 72.0]
        },
        'output': {
            'save_intermediate_results': True,
            'include_confidence_scores': True,
            'generate_summary_reports': True
        }
    }
    
    def __init__(self, config_path=None):
        self.config = self.DEFAULT_CONFIG.copy()
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
    
    def load_config(self, config_path):
        with open(config_path, 'r') as f:
            custom_config = json.load(f)
            self.update_config(custom_config)
    
    def update_config(self, updates):
        for section, settings in updates.items():
            if section in self.config:
                self.config[section].update(settings)
            else:
                self.config[section] = settings
    
    def save_config(self, config_path):
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def get(self, section, key, default=None):
        return self.config.get(section, {}).get(key, default)

class ErrorHandler:
    @staticmethod
    def handle_processing_error(ticket_id, error, context=""):
        error_info = {
            'ticket_id': ticket_id,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context,
            'timestamp': datetime.now().isoformat()
        }
        return error_info
    
    @staticmethod
    def create_fallback_result(ticket_id, original_text, error_info):
        return {
            'ticket_id': ticket_id,
            'original_text': original_text,
            'predicted_category': 'general_inquiry',
            'predicted_priority': 'medium',
            'adjusted_priority': 'medium',
            'predicted_sentiment': 'neutral',
            'predicted_eta_hours': 4.0,
            'eta_category': 'same_day',
            'generated_response': 'Thank you for contacting us. We will review your request and respond shortly.',
            'generated_tags': ['general', 'inquiry'],
            'suggested_actions': ['review_manually'],
            'processing_status': 'error_with_fallback',
            'error_details': error_info,
            'processing_timestamp': datetime.now().isoformat()
        }

def setup_logging():
    import logging
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('../outputs/pipeline.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def create_sample_input_csv(output_path, num_samples=10):
    sample_data = {
        'ticket_id': [f'SAMPLE_{i:03d}' for i in range(1, num_samples + 1)],
        'text': [
            'My billing statement shows duplicate charges for this month',
            'The application crashes every time I try to upload a file',
            'I love your new feature update, great work!',
            'How do I change my password?',
            'This service is completely unreliable, I want my money back',
            'Can you help me update my account information?',
            'The website is loading very slowly today',
            'Thank you for the excellent customer service',
            'I need urgent assistance with my account suspension',
            'When will the new features be available?'
        ][:num_samples]
    }
    
    df = pd.DataFrame(sample_data)
    df.to_csv(output_path, index=False)
    return output_path