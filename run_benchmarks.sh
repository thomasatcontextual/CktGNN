#!/bin/bash

# Array of models to test
MODELS=("CktGNN" "DVAE" "DAGNN" "PACE")

# Run each model
for MODEL in "${MODELS[@]}"; do
    echo "=== Running $MODEL ==="
    docker-compose run --rm cktgnn python main.py \
        --epochs 5 \
        --save-appendix _$(echo "$MODEL" | tr '[:upper:]' '[:lower:]')_test \
        --model $MODEL \
        --hs 301 \
        --num-workers 2 \
        --batch-size 128
    echo ""  # Add spacing between runs
done

# Print summary from results
echo "=== Benchmark Summary ==="
echo "Model      Recon Acc  Valid DAG  Valid Ckt  Novel"
echo "------------------------------------------------"
for MODEL in "${MODELS[@]}"; do
    RESULTS=$(tail -n 1 "results/ckt_bench_101_$(echo "$MODEL" | tr '[:upper:]' '[:lower:]')_test/decode_results.txt")
    printf "%-10s %s\n" "$MODEL" "$RESULTS"
done 