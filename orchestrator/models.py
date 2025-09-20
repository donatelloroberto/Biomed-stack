# small abstraction for integration points (RAG, LLM)
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer

# LLM: replace with stanford-crfm/BioMedLM-2.7b or other HF ID
LLM_HF_ID = 'stanford-crfm/BioMedLM-2.7b'
EMBED_MODEL_ID = 'sentence-transformers/all-mpnet-base-v2'

def load_llm(device=0):
    tokenizer = AutoTokenizer.from_pretrained(LLM_HF_ID)
    model = AutoModelForCausalLM.from_pretrained(LLM_HF_ID, device_map='auto')
    gen = pipeline('text-generation', model=model, tokenizer=tokenizer, device=device)
    return gen

def load_embed():
    return SentenceTransformer(EMBED_MODEL_ID)
