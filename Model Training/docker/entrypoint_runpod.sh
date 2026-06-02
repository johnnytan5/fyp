#!/bin/bash
set -e

python - <<'PYEOF'
import torch, sys
if not torch.cuda.is_available():
    print("ERROR: CUDA not available — aborting RunPod container.")
    sys.exit(1)
print(f"GPU:  {torch.cuda.get_device_name(0)}")
print(f"CUDA: {torch.version.cuda}")
PYEOF

exec "$@"
