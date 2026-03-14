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
ADAPT_NPZ=$INCART_PROCESSED/adapt.npz
INCART_ADAPT_TRAIN=$INCART_PROCESSED/adapt_train.npz
INCART_ADAPT_VAL=$INCART_PROCESSED/adapt_val.npz
STUDENT_FT_OUT=$ROOT_DIR/outputs/student_finetune
MIT_CONTEXT_BEATS=${MIT_CONTEXT_BEATS:-3}
INCART_CONTEXT_BEATS=${INCART_CONTEXT_BEATS:-3}
MIT_DSC_TRAIN="101,106,108,109,112,114,115,116,118,119,122,124,201,203,205,207,208,209,215,220,221,223,230"
MIT_DSC_TEST="100,103,105,111,113,117,121,123,200,202,210,212,213,214,219,222,228,231,232,233,234"

mkdir -p "$MIT_PROCESSED" "$INCART_PROCESSED" "$TEACHER_OUT" "$STUDENT_OUT" "$STUDENT_AUG_OUT" "$STUDENT_FT_OUT"

set -x
$PY_BIN -m src.tools.env_check
$PY_BIN -m src.tools.dataset_check --dataset mitbih --data_dir "$MIT_DATA"
$PY_BIN -m src.tools.dataset_check --dataset incart --data_dir "$INCART_DATA"
$PY_BIN -m src.data.prepare_mitbih --data_dir "$MIT_DATA" --out_dir "$MIT_PROCESSED" --target_fs 250 --window_len 256 --context_beats $MIT_CONTEXT_BEATS --minority_classes "${MIT_MINORITY_CLASSES:-3,4}" --minority_factor ${MIT_MINORITY_FACTOR:-4} --minority_noise 0.02 --val_focus_classes "${MIT_VAL_CLASSES:-3,4}" --train_records "${MIT_TRAIN_RECORDS:-$MIT_DSC_TRAIN}" --test_records "${MIT_TEST_RECORDS:-$MIT_DSC_TEST}"
$PY_BIN -m src.data.prepare_incart --data_dir "$INCART_DATA" --out_dir "$INCART_PROCESSED" --target_fs 250 --window_len 256 --context_beats $INCART_CONTEXT_BEATS --minority_classes "${INCART_MINORITY_CLASSES:-3,4}" --minority_factor ${INCART_MINORITY_FACTOR:-6} --minority_noise 0.02
$PY_BIN -m src.train.train_erm --model teacher --train_npz "$MIT_PROCESSED/train.npz" --val_npz "$MIT_PROCESSED/val.npz" --epochs ${TEACHER_EPOCHS:-3} --batch 128 --out_dir "$TEACHER_OUT"
$PY_BIN -m src.train.train_erm --model student --train_npz "$MIT_PROCESSED/train.npz" --val_npz "$MIT_PROCESSED/val.npz" --epochs ${STUDENT_EPOCHS:-5} --batch 256 --class_weight none --boost_classes 3,4 --boost_factor ${MIT_BOOST_FACTOR:-3} --no_sampler --out_dir "$STUDENT_OUT"
$PY_BIN -m src.train.train_erm --model student_aug --train_npz "$MIT_PROCESSED/train.npz" --val_npz "$MIT_PROCESSED/val.npz" --epochs ${STUDENT_EPOCHS:-5} --batch 256 --class_weight none --boost_classes 3,4 --boost_factor ${MIT_BOOST_FACTOR:-3} --no_sampler --augment --out_dir "$STUDENT_AUG_OUT"
$PY_BIN -m src.eval.evaluate --model teacher --ckpt "$TEACHER_OUT/best.pt" --npz "$MIT_PROCESSED/test.npz" --out_json "$TEACHER_OUT/metrics_test.json"
$PY_BIN -m src.eval.evaluate --model student --ckpt "$STUDENT_OUT/best.pt" --npz "$MIT_PROCESSED/test.npz" --out_json "$STUDENT_OUT/metrics_mit_test.json"
$PY_BIN -m src.eval.evaluate --model student --ckpt "$STUDENT_OUT/best.pt" --npz "$INCART_PROCESSED/test.npz" --out_json "$STUDENT_OUT/metrics_incart_test.json"
$PY_BIN -m src.eval.evaluate --model student_aug --ckpt "$STUDENT_AUG_OUT/best.pt" --npz "$MIT_PROCESSED/test.npz" --out_json "$STUDENT_AUG_OUT/metrics_mit_test.json"
$PY_BIN -m src.eval.evaluate --model student_aug --ckpt "$STUDENT_AUG_OUT/best.pt" --npz "$INCART_PROCESSED/test.npz" --out_json "$STUDENT_AUG_OUT/metrics_incart_test.json"
$PY_BIN -m src.train.adapt_bn --model student --ckpt "$STUDENT_OUT/best.pt" --npz "$ADAPT_NPZ" --out_ckpt "$STUDENT_OUT/bn_adapt.pt"
$PY_BIN -m src.eval.evaluate --model student --ckpt "$STUDENT_OUT/bn_adapt.pt" --npz "$INCART_PROCESSED/test.npz" --out_json "$STUDENT_OUT/metrics_incart_test_adapt.json"
$PY_BIN -m src.train.adapt_bn --model student_aug --ckpt "$STUDENT_AUG_OUT/best.pt" --npz "$ADAPT_NPZ" --out_ckpt "$STUDENT_AUG_OUT/bn_adapt.pt"
$PY_BIN -m src.eval.evaluate --model student_aug --ckpt "$STUDENT_AUG_OUT/bn_adapt.pt" --npz "$INCART_PROCESSED/test.npz" --out_json "$STUDENT_AUG_OUT/metrics_incart_test_adapt.json"
$PY_BIN -m src.train.train_erm --model student --train_npz "$INCART_ADAPT_TRAIN" --val_npz "$INCART_ADAPT_VAL" --epochs ${FT_EPOCHS:-5} --batch 512 --lr ${FT_LR:-5e-4} --class_weight none --boost_classes 3,4 --boost_factor ${INCART_BOOST_FACTOR:-4} --no_sampler --init_ckpt "$STUDENT_OUT/best.pt" --out_dir "$STUDENT_FT_OUT"
$PY_BIN -m src.eval.evaluate --model student --ckpt "$STUDENT_FT_OUT/best.pt" --npz "$INCART_PROCESSED/test.npz" --out_json "$STUDENT_FT_OUT/metrics_incart_test.json"
$PY_BIN -m src.eval.evaluate --model student --ckpt "$STUDENT_FT_OUT/best.pt" --npz "$MIT_PROCESSED/test.npz" --out_json "$STUDENT_FT_OUT/metrics_mit_test.json"
$PY_BIN -m src.tools.edge_report --model student --input_len 256
set +x

echo "Pipeline complete. Metrics saved under outputs/."
