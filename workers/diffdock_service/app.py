from fastapi import FastAPI
from pydantic import BaseModel
app = FastAPI()

class DockReq(BaseModel):
    protein_pdb: str
    ligand_smiles: str

@app.post('/dock')
def dock(req: DockReq):
    # Stub: production replace with DiffDock repo and weights
    return {'status':'stub', 'note':'Replace with DiffDock inference service.'}
