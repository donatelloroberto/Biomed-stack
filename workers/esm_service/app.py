from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import os

app = FastAPI()
MODEL_DIR = '/models/esm'

class SeqReq(BaseModel):
    seq: str

@app.on_event('startup')
def load():
    global model
    try:
        # sentence-transformers wrapper will auto-download if MODEL_DIR isn't present
        if os.path.exists(MODEL_DIR):
            model = SentenceTransformer(MODEL_DIR)
        else:
            model = SentenceTransformer('facebook/esm2_t33_650M_UR50S')
    except Exception as e:
        model = None
        print('ESM load failed:', e)

@app.post('/embed')
def embed_seq(req: SeqReq):
    if model is None:
        return {'status':'error','detail':'Model not loaded'}
    emb = model.encode(req.seq)
    return {'embedding': emb.tolist()}
