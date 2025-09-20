# Deployment notes & dataset sizing

## AlphaFold / OpenFold databases
- UniRef90, MGnify, PDB70 and other MSA databases are large. Expect **~1 TB** for the full set.
- Option: use AlphaFold DB (precomputed structures) to avoid large MSA builds; this is much smaller but not universal.
- Mount the databases read-only into `/data/alphafold_db` inside the openfold_worker container.

## Model sizes (approx):
- BioMedLM 2.7B: tens of GB (weights vary with format & quantization). Using 8-bit/bitsandbytes reduces memory footprint.
- ESM-2 (650M): ~2-4 GB; larger ESM models can be 15B (do not include those in default workflow).
- MegaMolBART: model size depends on checkpoint; expect several GB.
- DiffDock: model weights and dependencies may be several GB.

## Hardware recommendations
- For development: 1x A40/80GB or A100/40GB recommended for BioMedLM. ESM and MegaMolBART can run on smaller GPUs for inference.
- AlphaFold/OpenFold inference: GPU with >= 16GB recommended for single-sequence inference; heavy MSA generation & full pipelines benefit from more memory.

## NVIDIA Container Toolkit
- Install NVIDIA Container Toolkit so Docker containers with `runtime: nvidia` can access GPUs.
- https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

