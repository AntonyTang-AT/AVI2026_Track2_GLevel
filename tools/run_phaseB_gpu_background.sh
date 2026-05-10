#!/usr/bin/env bash
# 使用 GPU（默认 conda env magnus）顺序跑完整 Phase B 四组实验；日志与 PID 写入 experiments/glevel_improvement_plan/
set -eu
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_ROOT="$(cd "${_SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT:-$_ROOT}"

# 勿继承宿主机上错误的 PYTHON（如 CPU torch）；除非显式传入 PHASE_B_PYTHON
_MAGNUS_PY="${PHASE_B_PYTHON:-/home/emo/anaconda3/envs/magnus/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHON="${_MAGNUS_PY}"
export PHASE_B_RUN="${PHASE_B_RUN:-1}"
export PHASE_B_DRY_RUN="${PHASE_B_DRY_RUN:-0}"

LOG="${_ROOT}/experiments/glevel_improvement_plan/phaseB_gpu_run.log"
PID_FILE="${_ROOT}/experiments/glevel_improvement_plan/phaseB_gpu_run.pid"
mkdir -p "$(dirname "${LOG}")"

echo "[run_phaseB_gpu_background] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} PYTHON=${PYTHON}" >>"${LOG}"
echo "[run_phaseB_gpu_background] start $(date -Iseconds)" >>"${LOG}"

nohup env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" PYTHON="${PYTHON}" PHASE_B_RUN="${PHASE_B_RUN}" PHASE_B_DRY_RUN="${PHASE_B_DRY_RUN}" \
  PHASE_B_NUM_EPOCHS="${PHASE_B_NUM_EPOCHS:-200}" PHASE_B_JOBS="${PHASE_B_JOBS:-coral,ce,weight,select}" \
  bash "${_ROOT}/tools/glevel_phaseB_train_grid.sh" >>"${LOG}" 2>&1 &
echo $! >"${PID_FILE}"
echo "[run_phaseB_gpu_background] PID=$(cat "${PID_FILE}") log=${LOG}"
