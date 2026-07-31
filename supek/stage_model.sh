#!/bin/bash
# Download the model into a Lustre-backed HF_HOME. Run on a LOGIN NODE.
#
#     ./supek/stage_model.sh                      # the configured default model
#     MODEL_REPO=Qwen/Qwen3-14B-AWQ ./supek/stage_model.sh
#
# Worker nodes have no internet, so jobs run with HF_HUB_OFFLINE=1 and can only
# use what this script leaves behind.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=config.sh
source "$HERE/config.sh"

mkdir -p "$MODEL_DIR"

echo "Staging $MODEL_REPO -> $MODEL_DIR"
# Unset offline mode for this step only: config.sh exports it for job use.
HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 \
    huggingface-cli download "$MODEL_REPO" --local-dir "$MODEL_DIR"

# The optillm router's classifier is also a HuggingFace download, and it is
# fetched lazily on the first routed request - which happens on a worker node
# with no internet. Stage it here too, into the same HF_HOME, or every `router`
# request fails over to the fallback path.
echo
echo "Staging the optillm router classifier ..."
HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 \
    huggingface-cli download codelion/optillm-modernbert-large
HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 \
    huggingface-cli download answerdotai/ModernBERT-large

echo
echo "Staged into $HF_HOME:"
du -sh "$MODEL_DIR"
echo "  files: $(find "$MODEL_DIR" -type f | wc -l)"
echo
echo "Weights present:"
find "$MODEL_DIR" -name '*.safetensors' -o -name '*.bin' | sed 's/^/  /'

# Prove the staged copy loads with no network at all, so a job does not
# discover a missing shard an hour into the queue.
echo
echo "Verifying it loads fully offline ..."
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 apptainer exec "$SIF" python3 -c "
from transformers import AutoConfig, AutoTokenizer
config = AutoConfig.from_pretrained('$MODEL_DIR')
AutoTokenizer.from_pretrained('$MODEL_DIR')
print('offline load OK:', config.model_type, 'hidden', config.hidden_size)
"

echo
if [ -f "$OPTILLM_DIR/optillm.py" ]; then
    echo "optillm checkout present at $OPTILLM_DIR"
else
    cat >&2 <<EOF

WARNING: no optillm checkout at $OPTILLM_DIR.

Jobs run optillm from source, not from the image, because this project's copy
carries local changes (configurable router device; the optillm_router_approach
response field). Copy it up from the laptop:

    scp -r ./optillm ${USER}@${SUPEK_LOGIN}:${OPTILLM_DIR}

Without it, serve.sh will refuse to start rather than silently serving a
released optillm whose router reports nothing.
EOF
fi

echo
echo "Done. Jobs can now run with HF_HUB_OFFLINE=1."
