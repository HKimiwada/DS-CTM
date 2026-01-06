#!/bin/bash
# Run 8 baseline CTM seeds in parallel, one per GPU
# Usage: bash validation/run_baseline_validation.sh

EXPERIMENT_NAME="ctm_validation_v1"
WANDB_PROJECT="lagged-ctm-validation"
NUM_SEEDS=8
START_SEED=42

echo "Starting Baseline CTM validation runs..."
echo "Experiment: $EXPERIMENT_NAME"
echo "Seeds: $START_SEED to $((START_SEED + NUM_SEEDS - 1))"

for i in $(seq 0 $((NUM_SEEDS - 1))); do
    SEED=$((START_SEED + i))
    GPU=$((i % 8))  # Distribute across available GPUs
    
    echo "Launching baseline seed=$SEED on GPU=$GPU"
    
    CUDA_VISIBLE_DEVICES=$GPU python -m validation.train_validation \
        --model_type baseline \
        --seed $SEED \
        --experiment_name $EXPERIMENT_NAME \
        --wandb_project $WANDB_PROJECT \
        --parity_sequence_length 64 \
        --d_model 1024 \
        --d_input 512 \
        --iterations 75 \
        --memory_length 25 \
        --n_synch_out 32 \
        --n_synch_action 32 \
        --heads 8 \
        --synapse_depth 1 \
        --batch_size 64 \
        --batch_size_test 256 \
        --lr 0.0001 \
        --training_iterations 50001 \
        --warmup_steps 500 \
        --gradient_clipping 0.9 \
        --eval_every 1000 \
        --save_every 2000 \
        --neuron_select_type random \
        --deep_memory &
    
    # Small delay to avoid race conditions in wandb init
    sleep 5
done

echo "All baseline jobs launched. Waiting for completion..."
wait
echo "All baseline validation runs complete!"