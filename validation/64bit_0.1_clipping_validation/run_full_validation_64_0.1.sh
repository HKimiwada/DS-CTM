#!/bin/bash
# Run all 16 experiments (8 baseline + 8 lagged) on 8 GPUs simultaneously
# Each GPU runs 1 baseline + 1 lagged

set -e

EXPERIMENT_NAME="ctm_validation_v3"
WANDB_PROJECT="baseline-lagged-ctm-validation"
NUM_SEEDS=8
START_SEED=42

echo "=============================================="
echo "CTM Validation: 8 baseline + 8 lagged on 8 GPUs"
echo "=============================================="

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="$REPO_ROOT"
cd "$REPO_ROOT"

for i in $(seq 0 $((NUM_SEEDS - 1))); do
    SEED=$((START_SEED + i))
    GPU=$((i % 8))
    
    echo "GPU $GPU: baseline seed=$SEED + lagged seed=$SEED"
    
    # Baseline
    CUDA_VISIBLE_DEVICES=$GPU python -m validation.train_validation \
        --model_type baseline --seed $SEED \
        --experiment_name $EXPERIMENT_NAME --wandb_project $WANDB_PROJECT \
        --parity_sequence_length 64 --d_model 1024 --d_input 512 \
        --iterations 75 --memory_length 25 --n_synch_out 32 --n_synch_action 32 \
        --heads 8 --synapse_depth 1 --batch_size 64 --batch_size_test 256 \
        --lr 0.0001 --training_iterations 50001 --warmup_steps 500 \
        --gradient_clipping 0.1 --eval_every 1000 --neuron_select_type random --deep_memory &
    
    sleep 2
    
    # Lagged on same GPU
    CUDA_VISIBLE_DEVICES=$GPU python -m validation.train_validation \
        --model_type lagged --seed $SEED --lags 0 1 2 3 4 \
        --experiment_name $EXPERIMENT_NAME --wandb_project $WANDB_PROJECT \
        --parity_sequence_length 64 --d_model 1024 --d_input 512 \
        --iterations 75 --memory_length 25 --n_synch_out 32 --n_synch_action 32 \
        --heads 8 --synapse_depth 1 --batch_size 64 --batch_size_test 256 \
        --lr 0.0001 --training_iterations 50001 --warmup_steps 500 \
        --gradient_clipping 0.1 --eval_every 1000 --neuron_select_type random --deep_memory &
    
    sleep 2
done

echo "All 16 jobs launched. Waiting..."
wait
echo "Complete! Run: python -m validation.analyze_results"