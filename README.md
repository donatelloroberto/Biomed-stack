# biomed-stack (Full workers added)

This repository expands the starter skeleton with **GPU-ready worker Dockerfiles and FastAPI wrappers**
for ESM-2 embeddings, BioMedLM inference (8-bit via bitsandbytes), OpenFold/AlphaFold inference (wrapper),
MegaMolBART molecule generation, and DiffDock docking.

IMPORTANT: This repo **does not** include model weights or large biological databases. You **must** download
model checkpoints yourself (commands provided in `scripts/download_models.sh`) and place them under `/models` or
use the helper script which uses `huggingface_hub.snapshot_download`. You must accept each model's license.

Safety: the orchestrator includes a procedural detector and HITL gating logic; do not remove. This stack is for
computational research only — not for generating wet-lab protocols.

## Quick steps (high level)
1. Install Docker + NVIDIA Container Toolkit (for GPU runtime). See README_FILES for links.
2. `cp env.sample .env` and fill `HF_TOKEN`.
3. Run `bash scripts/download_models.sh` **locally** to download model weights into `./models` (requires HF login).
4. `docker compose up --build` (compose will include GPU-enabled workers; ensure Docker has GPU access).
5. Follow `README_FILES/DEPLOY_NOTES.md` for AlphaFold DB / MSA setup and exact resource sizing.

See `scripts/` for helper download commands and `workers/` for GPU-ready worker containers.
