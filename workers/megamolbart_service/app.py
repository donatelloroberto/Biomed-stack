from fastapi import FastAPI
from pydantic import BaseModel
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

app = FastAPI()
MODEL_DIR = '/models/megamolbart'

class GenReq(BaseModel):
    prompt: str
    num: int = 5

@app.on_event('startup')
def load():
    global tok, model
    try:
        if os.path.exists(MODEL_DIR):
            tok = AutoTokenizer.from_pretrained(MODEL_DIR)
            model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR, device_map='auto')
        else:
            tok = AutoTokenizer.from_pretrained('nvidia/megamolbart')
            model = AutoModelForSeq2SeqLM.from_pretrained('nvidia/megamolbart', device_map='auto')
    except Exception as e:
        tok = None
        model = None
        print('MegaMolBART load failed:', e)

@app.post('/generate')
def generate(req: GenReq):
    if model is None:
        return {'status':'error','detail':'Model not loaded. Place weights in /models/megamolbart or check logs.'}
    inputs = tok(req.prompt, return_tensors='pt').to(model.device)
    outs = model.generate(**inputs, max_length=256, num_return_sequences=req.num)
    decoded = [tok.decode(o, skip_special_tokens=True) for o in outs]
    return {'candidates': decoded}
