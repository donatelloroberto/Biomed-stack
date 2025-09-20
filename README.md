# biomed-stack

**Quick**: This repo is a research-oriented stack for computational disease research and drug discovery using open-source models. It is not for wet-lab or procedural instruction generation. Use responsibly.

## Components
- **orchestrator**: FastAPI service that provides RBAC, audit logging, RAG prompt building, and LLM orchestration (BioMedLM recommended).
- **workers**: lightweight GPU workers that host ESM-2 embeddings, structure inference (OpenFold/AlphaFold), molecule generation (MegaMolBART), and docking (DiffDock). They expose small REST endpoints.
- **Milvus**: vector DB for retrieval (RAG). Included in docker-compose.
- **Keycloak**: identity + role-based access control (dev-mode included in docker-compose).
- **Classifier**: small procedural-detector to flag outputs that might contain wet-lab steps; flagged outputs are routed to a reviewer queue.

## Quickstart (dev)
1. Edit `env.sample` → `.env` (add `HF_TOKEN`, etc.).
2. `bash scripts/setup_env.sh` (installs CLI tools; does not download heavy models).
3. `docker compose up --build` (will start Postgres, Redis, Milvus, Keycloak, Orchestrator). Worker images are stubs — configure GPUs and model weights before production.
4. Build your RAG index: `python scripts/build_milvus_index.py --pubmed ./data/pmc_paragraphs.jsonl`.
5. Use Keycloak to create users/roles. Use `POST /ideate` to generate conceptual hypotheses.

## Important notes
- **Safety**: outputs containing procedural language are flagged and not returned directly; a reviewer must approve them.
- **Licenses**: check each model's license before production/commercial use. See `README_FILES/NOTE_LICENSES.md`.
- **Compute**: large models require GPUs. For inference, use quantization (bitsandbytes) and accelerate where possible.

## Next steps
- Replace worker stubs with full repo clones for OpenFold, MegaMolBART, DiffDock and mount model data to `/models` inside each worker container.
- Train or tune the procedural classifier on your internal red-team dataset for production.
