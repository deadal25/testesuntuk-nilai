import torch
from sentence_transformers import SentenceTransformer
from transformers import MarianMTModel, MarianTokenizer

def load_model():
    """Memuat model Sentence-BERT."""
    return SentenceTransformer('all-MiniLM-L6-v2')

def load_translation_model():
    """Memuat model translasi Indonesia -> Inggris secara lokal."""
    model_name = "Helsinki-NLP/opus-mt-id-en"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name).to(device)
    model.eval()
    return tokenizer, model