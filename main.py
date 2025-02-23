from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import os

# Initialize FastAPI app
app = FastAPI()

# Define the request body schema using Pydantic
class TranslationRequest(BaseModel):
    text: str
    target_lang: str

# Load the model and tokenizer
model_name = "ai4bharat/indictrans2-indic-en-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True)

# Language token mapping for IndicTrans2
lang_code_map = {
    "asm_Beng": "<2asm_Beng>",  # Assamese
    "ben_Beng": "<2ben_Beng>",  # Bengali
    "brx_Deva": "<2brx_Deva>",  # Bodo
    "doi_Deva": "<2doi_Deva>",  # Dogri
    "eng_Latn": "<2eng_Latn>",  # English
    "gom_Deva": "<2gom_Deva>",  # Konkani
    "guj_Gujr": "<2guj_Gujr>",  # Gujarati
    "hin_Deva": "<2hin_Deva>",  # Hindi
    "kan_Knda": "<2kan_Knda>",  # Kannada
    "kas_Arab": "<2kas_Arab>",  # Kashmiri (Arabic)
    "kas_Deva": "<2kas_Deva>",  # Kashmiri (Devanagari)
    "mai_Deva": "<2mai_Deva>",  # Maithili
    "mal_Mlym": "<2mal_Mlym>",  # Malayalam
    "mar_Deva": "<2mar_Deva>",  # Marathi
    "mni_Beng": "<2mni_Beng>",  # Manipuri (Bengali)
    "mni_Mtei": "<2mni_Mtei>",  # Manipuri (Meitei)
    "npi_Deva": "<2npi_Deva>",  # Nepali
    "ory_Orya": "<2ory_Orya>",  # Odia
    "pan_Guru": "<2pan_Guru>",  # Punjabi
    "san_Deva": "<2san_Deva>",  # Sanskrit
    "sat_Olck": "<2sat_Olck>",  # Santali
    "snd_Arab": "<2snd_Arab>",  # Sindhi (Arabic)
    "snd_Deva": "<2snd_Deva>",  # Sindhi (Devanagari)
    "tam_Taml": "<2tam_Taml>",  # Tamil
    "tel_Telu": "<2tel_Telu>",  # Telugu
    "urd_Arab": "<2urd_Arab>",  # Urdu
}

# Root endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to the IndicTrans2 Translation API!"}

# Translation endpoint
@app.post("/translate/")
async def translate(request: TranslationRequest):
    try:
        # Check if the target language is supported
        if request.target_lang not in lang_code_map:
            raise HTTPException(status_code=400, detail="Unsupported target language")

        # Prepend language token to input text
        input_text = lang_code_map[request.target_lang] + " " + request.text
        inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)

        # Generate translation
        with torch.no_grad():
            translated_tokens = model.generate(**inputs)
        
        # Decode the translated tokens
        translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
        return {"translated_text": translated_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the app
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))  # Use Render's PORT environment variable
    uvicorn.run(app, host="0.0.0.0", port=port)