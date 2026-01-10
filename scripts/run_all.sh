#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PY_BIN=${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}
if [ ! -x "$PY_BIN" ]; then
  PY_BIN=$(command -v python3)
fi

echo "Using python: $PY_BIN"

MIT_DATA=$ROOT_DIR/data/mitbih
INCART_DATA=$ROOT_DIR/data/incart
MIT_PROCESSED=$ROOT_DIR/processed/mitbih
INCART_PROCESSED=$ROOT_DIR/processed/incart
TEACHER_OUT=$ROOT_DIR/outputs/teacher
STUDENT_OUT=$ROOT_DIR/outputs/student
STUDENT_AUG_OUT=$ROOT_DIR/outputs/student_aug

mkdir -p "$MIT_PROCESSED" "$INCART_PROCESSED" "$TEACHER_OUT" "$STUDENT_OUT" "$STUDENT_AUG_OUT"

set -x
$PY_BIN -m src.tools.env_check
$PY_BIN -m src.tools.dataset_check --dataset mitbih --data_dir "$MIT_DATA"
$PY_BIN -m src.tools.dataset_check --dataset incart --data_dir "$INCART_DATA"
$PY_BIN -m src.data.prepare_mitbih --data_dir "$MIT_DATA" --out_dir "$MIT_PROCESSED" --target_fs 250 --window_len 256
$PY_BIN -m src.data.prepare_incart --data_dir "$INCART_DATA" --out_dir "$INCART_PROCESSED" --target_fs 250 --window_len 256
$PY_BIN -m src.train.train_erm --model teacher --train_npz "$MIT_PROCESSED/train.npz" --val_npz "$MIT_PROCESSED/val.npz" --epochs ${TEACHER_EPOCHS:-25} --batch 192 --lr ${TEACHER_LR:-5e-4} --weight_decay 1e-4 --num_workers ${NUM_WORKERS:-2} --class_weights_with_sampler --out_dir "$TEACHER_OUT"
$PY_BIN -m src.train.train_erm --model student --train_npz "$MIT_PROCESSED/train.npz" --val_npz "$MIT_PROCESSED/val.npz" --epochs ${STUDENT_EPOCHS:-40} --batch 256 --lr ${STUDENT_LR:-8e-4} --weight_decay 1e-4 --num_workers ${NUM_WORKERS:-2} --no_class_weights --out_dir "$STUDENT_OUT"
$PY_BIN -m src.train.train_erm --model student_aug --train_npz "$MIT_PROCESSED/train.npz" --val_npz "$MIT_PROCESSED/val.npz" --epochs ${STUDENT_AUG_EPOCHS:-50} --batch 256 --lr ${STUDENT_LR:-8e-4} --weight_decay 1e-4 --num_workers ${NUM_WORKERS:-2} --augment --no_class_weights --out_dir "$STUDENT_AUG_OUT"
$PY_BIN -m src.eval.evaluate --model teacher --ckpt "$TEACHER_OUT/best.pt" --npz "$MIT_PROCESSED/test.npz" --out_json "$TEACHER_OUT/metrics_test.json"
$PY_BIN -m src.eval.evaluate --model student --ckpt "$STUDENT_OUT/best.pt" --npz "$MIT_PROCESSED/test.npz" --out_json "$STUDENT_OUT/metrics_mit_test.json"
$PY_BIN -m src.eval.evaluate --model student --ckpt "$STUDENT_OUT/best.pt" --npz "$INCART_PROCESSED/test.npz" --out_json "$STUDENT_OUT/metrics_incart_test.json" --adapt_npz "$INCART_PROCESSED/adapt.npz"
$PY_BIN -m src.eval.evaluate --model student_aug --ckpt "$STUDENT_AUG_OUT/best.pt" --npz "$MIT_PROCESSED/test.npz" --out_json "$STUDENT_AUG_OUT/metrics_mit_test.json"
$PY_BIN -m src.eval.evaluate --model student_aug --ckpt "$STUDENT_AUG_OUT/best.pt" --npz "$INCART_PROCESSED/test.npz" --out_json "$STUDENT_AUG_OUT/metrics_incart_test.json" --adapt_npz "$INCART_PROCESSED/adapt.npz"
$PY_BIN -m src.tools.edge_report --model student --input_len 256
set +x

echo "Pipeline complete. Metrics saved under outputs/."
