import os
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

app = FastAPI()

# ✅ Model details
MODEL_NAME = "ai4bharat/indictrans2-indic-en-1B"
MODEL_PATH = "./indictrans2_model"

# ✅ Download model if not present
if not os.path.exists(MODEL_PATH):
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=MODEL_NAME, local_dir=MODEL_PATH)

# ✅ Load model & tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)

# ✅ Request format
class TranslationRequest(BaseModel):
    text: str
    source_lang: str
    target_lang: str

@app.post("/translate")
def translate_text(request: TranslationRequest):
    """Translate text from source to target language"""
    inputs = tokenizer(request.text, return_tensors="pt", padding=True, truncation=True)
    translated_ids = model.generate(inputs["input_ids"])
    translated_text = tokenizer.decode(translated_ids[0], skip_special_tokens=True)
    
    return {"translated_text": translated_text}

@app.get("/")
def home():
    return {"message": "IndicTrans2 API is running!"}

# ✅ Use Render's dynamic PORT
if __name__ == "__main__":
    PORT = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=PORT)
