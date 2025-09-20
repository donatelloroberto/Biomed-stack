    #!/usr/bin/env bash
    set -e
    if [ -z "${HF_TOKEN}" ]; then
      echo "Set HF_TOKEN in your environment or in .env before running this script."
      exit 1
    fi

    python3 - <<'PY'
from huggingface_hub import snapshot_download
import os
os.environ['HF_TOKEN'] = os.environ.get('HF_TOKEN')
models = {
    'biomedlm': 'stanford-crfm/BioMedLM-2.7b',
    'esm': 'facebook/esm2_t33_650M_UR50S',
    'megamolbart': 'nvidia/megamolbart',
    'diffdock': 'diffdock/diffdock',
    # OpenFold/AlphaFold weights typically hosted separately; see DEPLOY_NOTES.md
}
for name, repo in models.items():
    out = os.path.join('models', name)
    print(f"Downloading {repo} into {out} (this may be large)...")
    snapshot_download(repo_id=repo, cache_dir=out, repo_type='model')
    print('Done', name)
PY
    echo "NOTE: OpenFold/AlphaFold databases are large and are not downloaded here. See README_FILES/DEPLOY_NOTES.md"
