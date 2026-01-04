#!/bin/bash
# Master script for full validation experiment
# Runs both baseline and lagged CTM with 8 seeds each
#
# Usage: 
#   bash validation/run_full_validation.sh          # Run sequentially
#   bash validation/run_full_validation.sh --parallel  # Run both simultaneously (needs 16 GPUs)

set -e

PARALLEL=false
if [[ "$1" == "--parallel" ]]; then
    PARALLEL=true
fi

echo "=============================================="
echo "CTM Validation Experiment"
echo "=============================================="
echo "Running 8 seeds for Baseline CTM"
echo "Running 8 seeds for Lagged-CTM (lags=[0,1,2,4])"
echo "=============================================="

cd "$(dirname "$0")/.."

if $PARALLEL; then
    echo "Running baseline and lagged in parallel (requires 16 GPUs)"
    bash validation/run_baseline_validation.sh &
    BASELINE_PID=$!
    
    # Offset GPU assignments for lagged runs
    for i in $(seq 0 7); do
        SEED=$((42 + i))
        GPU=$((i + 8))  # Use GPUs 8-15
        
        CUDA_VISIBLE_DEVICES=$GPU python -m validation.train_validation \
            --model_type lagged \
            --seed $SEED \
            --experiment_name ctm_validation_v1 \
            --wandb_project lagged-ctm-validation \
            --lags 0 1 2 3 4 \
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
        
        sleep 5
    done
    
    wait $BASELINE_PID
    wait
else
    echo "Running baseline first, then lagged (8 GPUs)"
    
    echo ""
    echo "Phase 1/2: Baseline CTM"
    echo "=============================================="
    bash validation/run_baseline_validation.sh
    
    echo ""
    echo "Phase 2/2: Lagged CTM"
    echo "=============================================="
    bash validation/run_lagged_validation.sh
fi

echo ""
echo "=============================================="
echo "Validation experiment complete!"
echo "Run analysis with: python -m validation.analyze_results"
echo "=============================================="