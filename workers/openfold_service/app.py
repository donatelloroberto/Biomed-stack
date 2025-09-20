from fastapi import FastAPI
from pydantic import BaseModel
app = FastAPI()

class SeqReq(BaseModel):
    seq: str

@app.post('/predict_structure')
def predict(req: SeqReq):
    # Stub: production must call OpenFold/AlphaFold with MSAs and model weights
    return {'status': 'stub', 'note': 'Replace with OpenFold/AlphaFold inference container mounted with databases.'}
