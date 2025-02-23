from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import os

app = FastAPI()

# Load model manually from Hugging Face Hub
MODEL_NAME = "ai4bharat/indictrans2-indic-en-1B"
MODEL_PATH = "./indictrans2_model"

# Download model files manually if not already present
if not os.path.exists(MODEL_PATH):
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=MODEL_NAME, local_dir=MODEL_PATH)

# Load tokenizer and model from local directory
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)

class TranslationRequest(BaseModel):
    text: str
    source_lang: str
    target_lang: str

@app.post("/translate")
def translate_text(request: TranslationRequest):
    inputs = tokenizer(request.text, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(**inputs)
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return {"translated_text": translated_text}

@app.get("/")
def home():
    return {"message": "IndicTrans2 API is running!"}
