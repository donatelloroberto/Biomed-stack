from fastapi import FastAPI, Depends, HTTPException, Request
from pydantic import BaseModel
import os, time, logging, requests
from loguru import logger
from orchestrator.auth import require_role
from orchestrator.models import load_embed, load_llm
from orchestrator.utils import redact_procedural
from pymilvus import connections, Collection

app = FastAPI(title='BioMed Orchestrator')

connections.connect(host=os.environ.get('MILVUS_HOST','milvus'), port=os.environ.get('MILVUS_PORT','19530'))
embed = load_embed()
# LLM delegation: call biomedlm_worker via REST (safer to isolate GPU workloads)
BIOMEDLM_URL = os.environ.get('BIOMEDLM_URL', 'http://biomedlm_worker:8110')

class IdeationReq(BaseModel):
    query: str
    max_tokens: int = 256

@app.post('/ideate')
async def ideate(req: IdeationReq, user=Depends(lambda request=None: require_role(request, 'researcher'))):
    col = Collection('pubmed')
    qvec = embed.encode([req.query])[0]
    hits = col.search([qvec], 'embedding', param={'anns_field':'embedding','topk':5,'metric_type':'L2'})
    evidence = ''
    for hit in hits[0]:
        evidence += f"[doc_id:{hit.id}] {hit.entity.get('text')}\n\n"

    prompt = (
        "You are a biomedical research assistant. Use the evidence excerpts below to propose conceptual hypotheses and rationale. "
        "Do NOT provide step-by-step wet-lab protocols. Attach citations.\n\n"
        f"EVIDENCE:\n{evidence}\nQUESTION: {req.query}\n\nHYPOTHESIS:")

    # delegate generation to the GPU-backed BioMedLM worker
    try:
        resp = requests.post(f"{BIOMEDLM_URL}/generate", json={"prompt":prompt, "max_new_tokens": req.max_tokens}, timeout=120)
        out = resp.json().get('text','')
    except Exception as e:
        logger.exception('BioMedLM worker error')
        raise HTTPException(status_code=500, detail='LLM worker error')

    redacted, flagged = redact_procedural(out)
    logger.info(f"IDEATE user=? q={req.query[:120]} flagged={flagged}")
    if flagged:
        return { 'hypothesis': redacted, 'status': 'flagged_for_review' }
    return { 'hypothesis': redacted, 'status': 'ok' }
