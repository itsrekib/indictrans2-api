from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import os

app = FastAPI()

# Load the summarizer model manually
MODEL_NAME = "facebook/bart-large-cnn"
MODEL_PATH = "./summarizer_model"

# Download model files manually if not already present
if not os.path.exists(MODEL_PATH):
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=MODEL_NAME, local_dir=MODEL_PATH)

# Load tokenizer and model from local directory
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)

class SummarizationRequest(BaseModel):
    text: str
    max_length: int = 100
    min_length: int = 30

@app.post("/summarize")
def summarize_text(request: SummarizationRequest):
    inputs = tokenizer(request.text, return_tensors="pt", padding=True, truncation=True)
    summary_ids = model.generate(inputs["input_ids"], max_length=request.max_length, min_length=request.min_length, length_penalty=2.0)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    return {"summary": summary}

@app.get("/")
def home():
    return {"message": "Summarization API is running!"}

# Use Render's assigned port
if __name__ == "__main__":
    import uvicorn
    PORT = int(os.environ.get("PORT", 8000))  # Get Render's assigned port
    uvicorn.run("app:app", host="0.0.0.0", port=PORT, reload=True)
