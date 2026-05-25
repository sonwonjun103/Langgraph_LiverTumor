#!/usr/bin/env bash
# Train nnUNetv2 folds sequentially for one dataset/config.
#
# Usage:
#   ./scripts/train_nnunetv2_all_folds.sh                  # 001 3d_fullres "0 1 2 3 4"
#   ./scripts/train_nnunetv2_all_folds.sh 002              # 002 3d_fullres "0 1 2 3 4"
#   ./scripts/train_nnunetv2_all_folds.sh 001 3d_fullres "0 2 4"
#   ./scripts/train_nnunetv2_all_folds.sh 001 3d_fullres "0 1 2 3 4" --c
#
# Any positional arguments after the third are forwarded to `nnUNetv2_train`.
# Per-fold stdout/stderr is teed into ./Results/nnunet_train_logs/.
# Override the log directory by setting NNUNET_LOG_DIR.

set -u

DATASET_ID="${1:-001}"
CONFIG="${2:-3d_fullres}"
FOLDS="${3:-0 1 2 3 4}"
shift $(( $# < 3 ? $# : 3 ))
EXTRA_ARGS=("$@")

LOG_DIR="${NNUNET_LOG_DIR:-./Results/nnunet_train_logs}"
mkdir -p "$LOG_DIR"

failed=()
for fold in $FOLDS; do
    log_file="$LOG_DIR/${DATASET_ID}_${CONFIG}_fold${fold}.log"
    echo "============================================================"
    echo "Training  Dataset=$DATASET_ID  Config=$CONFIG  Fold=$fold"
    echo "Log file  $log_file"
    echo "Extra     ${EXTRA_ARGS[*]:-(none)}"
    echo "============================================================"
    if nnUNetv2_train "$DATASET_ID" "$CONFIG" "$fold" "${EXTRA_ARGS[@]}" 2>&1 | tee "$log_file"; then
        echo "Fold $fold completed."
    else
        echo "Fold $fold failed. See $log_file"
        failed+=("$fold")
    fi
done

echo
echo "============================================================"
if [ ${#failed[@]} -eq 0 ]; then
    echo "All folds finished for Dataset=$DATASET_ID Config=$CONFIG"
else
    echo "Failed folds: ${failed[*]}"
    exit 1
fi
