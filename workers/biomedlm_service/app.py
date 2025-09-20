import os
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

app = FastAPI()
MODEL_DIR = '/models/biomedlm'

class GenReq(BaseModel):
    prompt: str
    max_new_tokens: int = 256

@app.on_event('startup')
def load_model():
    global tok, model
    try:
        # load in 8-bit if available
        tok = AutoTokenizer.from_pretrained(MODEL_DIR)
        model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, load_in_8bit=True, device_map='auto')
        logging.info('BioMedLM loaded from ' + MODEL_DIR)
    except Exception as e:
        logging.exception('Failed to load BioMedLM: %s', e)
        tok = None
        model = None

@app.post('/generate')
def generate(req: GenReq):
    if model is None:
        return {'status':'error','detail':'Model not loaded. Ensure weights are present in /models/biomedlm and have been downloaded.'}
    inputs = tok(req.prompt, return_tensors='pt').to(model.device)
    out = model.generate(**inputs, max_new_tokens=req.max_new_tokens, do_sample=True, temperature=0.8)
    text = tok.decode(out[0], skip_special_tokens=True)
    return {'text': text}
