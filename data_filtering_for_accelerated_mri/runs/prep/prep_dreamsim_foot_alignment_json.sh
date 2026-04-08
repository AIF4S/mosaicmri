#!/bin/bash
set -euo pipefail

PYTHON_BIN="/home/paula/miniconda3/envs/mri_varnet/bin/python"
FILTER_REPO="/home/paula/msk_mri_dataset/data_filtering_for_accelerated_mri"
RUN_DIR="/home/paula/ankle_spine_dreamsim"
DATA_ROOT="/data/datasets"
KNN=1
ANATOMY="FOOT"
VAL_NAME="msk_mri_foot_val"
TRAIN_JSON_FILTER="${FILTER_REPO}/datasets_custom/train/msk_mri.json"
TRAIN_EMB_FILTER="${FILTER_REPO}/embeddings/dreamsim_ensemble_emb_p128_msk_mri.pt"

mkdir -p "${RUN_DIR}/datasets_custom/evals" "${RUN_DIR}/datasets_custom/train" "${RUN_DIR}/embeddings"

GPU="$($PYTHON_BIN - <<'PY'
import subprocess
out=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits'], text=True)
rows=[]
for line in out.strip().splitlines():
    i,u,mu,mt=[x.strip() for x in line.split(',')]
    i=int(i);u=float(u);mu=float(mu);mt=float(mt)
    rows.append((i,u,mt-mu))
cand=[r for r in rows if r[2] >= 20000]
best=sorted(cand if cand else rows, key=lambda r:(r[1], -r[2]))[0]
print(best[0])
PY
)"

$PYTHON_BIN - <<PY
import json, os
from importlib.machinery import SourceFileLoader
run_dir = "${RUN_DIR}"
filter_repo = "${FILTER_REPO}"
data_root = "${DATA_ROOT}"
script = os.path.join(filter_repo, 'create_dataset_anatomy_json_custom.py')
mod = SourceFileLoader('create_dataset_anatomy_json_custom', script).load_module()
with open(os.path.join(filter_repo, 'file_category_mapping.json')) as f:
    mapping = json.load(f)
data_dir = os.path.join(data_root, 'msk_mri_h5', 'varnet_msk_dataset2', 'multicoil_val')
os.chdir(run_dir)
mod.create_json(data_dir, "${VAL_NAME}", False, 'test', mapping, "${ANATOMY}")
PY

CUDA_VISIBLE_DEVICES=${GPU} $PYTHON_BIN "${FILTER_REPO}/compute_embeddings_on_the_fly.py" \
  --dataset_json "${RUN_DIR}/datasets_custom/evals/${VAL_NAME}.json" \
  --save_path "${RUN_DIR}/embeddings" \
  --device cuda:0

pushd "${RUN_DIR}" >/dev/null
CUDA_VISIBLE_DEVICES=${GPU} $PYTHON_BIN "${FILTER_REPO}/alignment_filter.py" \
  --dataset_json_filter "${TRAIN_JSON_FILTER}" \
  --path_to_embeddings_filter "${TRAIN_EMB_FILTER}" \
  --path_to_embeddings_ref "${RUN_DIR}/embeddings/dreamsim_ensemble_emb_p128_${VAL_NAME}.pt" \
  --knn "${KNN}" \
  --device cuda:0 \
  --custom_suffix "${VAL_NAME}_filter"
popd >/dev/null

echo "[DONE] ${VAL_NAME}"
