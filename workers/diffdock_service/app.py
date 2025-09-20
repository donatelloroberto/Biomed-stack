from fastapi import FastAPI
from pydantic import BaseModel
import os, subprocess
app = FastAPI()
DIFFDOCK_SCRIPT = '/models/diffdock/run_diffdock.sh'

class DockReq(BaseModel):
    protein_pdb: str
    ligand_smiles: str

@app.post('/dock')
def dock(req: DockReq):
    if os.path.exists(DIFFDOCK_SCRIPT):
        try:
            cmd = [DIFFDOCK_SCRIPT, req.protein_pdb, req.ligand_smiles]
            proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return {'status':'ok','output':proc.stdout}
        except subprocess.CalledProcessError as e:
            return {'status':'error','stderr': e.stderr}
    return {'status':'stub','note':'DiffDock script not found at /models/diffdock. Mount DiffDock repo and add a run script.'}
