from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import os

app = FastAPI()

# Define the request body schema using Pydantic
class TranslationRequest(BaseModel):
    text: str
    target_lang: str

# Load the model and tokenizer
model_name = "ai4bharat/indictrans2-indic-en-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True)

@app.post("/translate/")
async def translate(request: TranslationRequest):
    try:
        inputs = tokenizer(request.text, return_tensors="pt", max_length=512, truncation=True)
        with torch.no_grad():
            translated_tokens = model.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id[request.target_lang])
        translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
        return {"translated_text": translated_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))  # Use Render's PORT environment variable
    uvicorn.run(app, host="0.0.0.0", port=port)