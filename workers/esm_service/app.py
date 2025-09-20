from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

app = FastAPI()
model = SentenceTransformer('facebook/esm2_t33_650M_UR50S')  # placeholder HF id

class SeqReq(BaseModel):
    seq: str

@app.post('/embed')
def embed_seq(req: SeqReq):
    emb = model.encode(req.seq)
    return {'embedding': emb.tolist()}
