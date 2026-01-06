#!/bin/bash
# Run 8 seeds across 8 GPUs sequentially per card
# Total 16 experiments (8 baseline + 8 lagged)
# Hard Regime: Gradient Clipping @ 0.9, Parity @ 128 bits

set -e

EXPERIMENT_NAME="ctm_hard_regime_v2"
WANDB_PROJECT="hard-lagged-ctm-validation_v2"
NUM_SEEDS=8
START_SEED=42

echo "=========================================================="
echo "CTM Hard Regime: 128-bit Parity with 0.9 Clipping"
echo "Running 1 active process per GPU (Baseline then Lagged)"
echo "=========================================================="

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="$REPO_ROOT"
cd "$REPO_ROOT"

for i in $(seq 0 $((NUM_SEEDS - 1))); do
    SEED=$((START_SEED + i))
    GPU=$((i % 8))
    
    # Run baseline and lagged sequentially on a single GPU background process
    (
        echo "GPU $GPU: Starting Baseline seed $SEED..."
        CUDA_VISIBLE_DEVICES=$GPU python -m validation.train_validation \
            --model_type baseline --seed $SEED \
            --experiment_name $EXPERIMENT_NAME --wandb_project $WANDB_PROJECT \
            --parity_sequence_length 128 --d_model 1024 --d_input 512 \
            --iterations 75 --memory_length 25 --n_synch_out 32 --n_synch_action 32 \
            --heads 8 --synapse_depth 1 --batch_size 64 --batch_size_test 256 \
            --lr 0.0001 --training_iterations 50001 --warmup_steps 500 \
            --gradient_clipping 0.9 --eval_every 1000 --neuron_select_type random --deep_memory
        
        echo "GPU $GPU: Baseline seed $SEED finished. Starting Lagged seed $SEED..."
        CUDA_VISIBLE_DEVICES=$GPU python -m validation.train_validation \
            --model_type lagged --seed $SEED --lags 0 1 2 3 4 \
            --experiment_name $EXPERIMENT_NAME --wandb_project $WANDB_PROJECT \
            --parity_sequence_length 128 --d_model 1024 --d_input 512 \
            --iterations 75 --memory_length 25 --n_synch_out 32 --n_synch_action 32 \
            --heads 8 --synapse_depth 1 --batch_size 64 --batch_size_test 256 \
            --lr 0.0001 --training_iterations 50001 --warmup_steps 500 \
            --gradient_clipping 0.9 --eval_every 1000 --neuron_select_type random --deep_memory
        
        echo "GPU $GPU: Both experiments for seed $SEED complete."
    ) &
    
    # Slight stagger to prevent simultaneous WandB initialization conflicts
    sleep 5
done

echo "All GPU workers launched. Monitoring progress..."
wait
echo "Complete! Analyze results with: python -m validation.analyze_results"