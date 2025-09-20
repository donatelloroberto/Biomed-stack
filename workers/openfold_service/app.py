import os, subprocess
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
OPENFOLD_SCRIPT = '/models/openfold/run_openfold_inference.sh'

class SeqReq(BaseModel):
    seq: str
    outname: str = 'pred'

@app.post('/predict_structure')
def predict(req: SeqReq):
    if os.path.exists(OPENFOLD_SCRIPT):
        # call the mounted OpenFold inference script
        try:
            cmd = [OPENFOLD_SCRIPT, req.seq, req.outname]
            proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return {'status':'ok','stdout':proc.stdout}
        except subprocess.CalledProcessError as e:
            return {'status':'error','stderr': e.stderr}
    return {'status':'stub','note':'OpenFold inference script not found at /models/openfold. Mount OpenFold repo or add a wrapper script.'}
