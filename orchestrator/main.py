from fastapi import FastAPI, Depends, HTTPException, Request
from pydantic import BaseModel
import os, time, logging
from loguru import logger
from orchestrator.auth import require_role
from orchestrator.models import load_llm, load_embed
from orchestrator.utils import redact_procedural
from pymilvus import connections, Collection

app = FastAPI(title='BioMed Orchestrator')

# connect Milvus
connections.connect(host=os.environ.get('MILVUS_HOST','milvus'), port=os.environ.get('MILVUS_PORT','19530'))
# load models (in dev this is OK; in prod run LLM on separate GPU-backed service)
embed = load_embed()
llm = load_llm(device=-1)  # device auto; change to GPU device id when available

class IdeationReq(BaseModel):
    query: str
    max_tokens: int = 256

@app.post('/ideate')
async def ideate(req: IdeationReq, user=Depends(lambda request=None: require_role(request, 'researcher'))):
    # 1) retrieve top-k passages from Milvus (collection 'pubmed')
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
    out = llm(prompt, max_new_tokens=req.max_tokens, do_sample=True, temperature=0.8)[0]['generated_text']
    redacted, flagged = redact_procedural(out)
    # audit log
    logger.info(f"IDEATE user={user.get('preferred_username','?')} q={req.query[:120]} flagged={flagged}")

    if flagged:
        # store in DB / queue for reviewer - simplified here
        return { 'hypothesis': redacted, 'status': 'flagged_for_review' }
    return { 'hypothesis': redacted, 'status': 'ok' }
