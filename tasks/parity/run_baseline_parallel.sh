#!/bin/bash
# Run 8 baseline seeds in parallel, one per GPU
for i in {0..7}
do
   # Set the GPU index and a unique seed for each run
   CUDA_VISIBLE_DEVICES=$i uv run python -m tasks.parity.train_baseline_wandb \
       --seed $((42 + i)) \
       --wandb_project lagged-ctm-research &
done
wait