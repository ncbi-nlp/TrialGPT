#!/bin/bash

: '
TrialGPT Pipeline Overview:

This script orchestrates the TrialGPT pipeline, a comprehensive system for matching patients to clinical trials using advanced AI techniques.

Components and Workflow:

1. TrialGPT-Retrieval:
   a) Keyword Generation (trialgpt_retrieval/keyword_generation.py):
      Generates search keywords for each patient description.
   b) Hybrid Fusion Retrieval (trialgpt_retrieval/hybrid_fusion_retrieval.py):
      Combines BM25 and MedCPT-based retrieval to find relevant clinical trials.

2. TrialGPT-Matching (trialgpt_matching/run_matching.py):
   Performs detailed matching between patients and retrieved trials.

3. TrialGPT-Ranking:
   a) Aggregation (trialgpt_ranking/run_aggregation.py):
      Aggregates matching results to produce overall scores.
   b) Final Ranking (trialgpt_ranking/rank_results.py):
      Produces the final ranking of trials for each patient.

Key Features:

1. Modular Design: Each major component is separated, allowing for easier maintenance and future improvements.
2. AI Model Utilization: Leverages advanced language models (e.g., GPT-4) and specialized models like MedCPT.
3. Hybrid Approach: Combines traditional (BM25) and AI-based (MedCPT) methods for improved retrieval.
4. Checkpoint System: Tracks progress and allows for resumption of interrupted runs.
5. Configurability: Various parameters can be adjusted for experimentation and optimization.

Technical Highlights:

- Extensive use of NLP techniques for tokenization, embedding, and semantic matching.
- Integration with OpenAI API for leveraging large language models.
- Sophisticated data processing pipeline handling various formats and transformations.
- Performance optimization through batching in encoding and retrieval.
- Consideration of ethical aspects, including patient consent in descriptions.

This pipeline represents a state-of-the-art approach to clinical trial matching, demonstrating the potential of AI to enhance efficiency and accuracy in this critical healthcare domain.
'
#TODO update above with "It be compatible with both GPT and Llama models"

# Exit immediately if a command exits with a non-zero status
# set -e

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Checkpoint file
CHECKPOINT_FILE="results/trialgpt_checkpoint.txt"

# Function to measure execution time
measure_time() {
    start_time=$(date +%s)
    "$@"
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    echo "Time taken: $duration seconds"
    return $duration
}

# Function to update checkpoint
update_checkpoint() {
    local step=$1
    local duration=$2
    local timestamp=$(date "+%Y-%m-%d %H:%M:%S")
    echo "Step $step completed at $timestamp (Duration: ${duration}s)" >> "$CHECKPOINT_FILE"
}

# Function to get last completed step
get_last_step() {
    if [ -f "$CHECKPOINT_FILE" ]; then
        tail -n 1 "$CHECKPOINT_FILE" | cut -d' ' -f2
    else
        echo "0"
    fi
}

# Set variables
CORPUS="sigir" #"sigir_small" "sigir_mini"
MODEL="gpt-4o" #"gpt-4o-mini" "gpt-4o" "Llama-3.2-3B-Instruct" "Llama-3.3-70B-Instruct"
QUANTIZE=true

K=20
BM25_WEIGHT=1
MEDCPT_WEIGHT=1

#overwrites data for the sage, else resumes in the stage
OVER_WRITE=false #true #'true'

# critical number, determines how many trials from retrieval to run on per patient
TOP_K=10

# batch size for the bert like retrieval model
BATCH_SIZE=512

# these two numbers below don't matter to the final score
#TODO: move them to TrialGPT/trialgpt_ranking/rank_results.py
ELIGIBILITY=0.0563 #0.0804 #0.0872 #0.1216
EXCLUSION=0.0432 #0.05 #0.0530 #0.0931

# Set the number of GPUs based on the model
case $MODEL in
    "gpt-4o-mini"|"gpt-4o")
        NGPUS=0  # OpenAI models don't use local GPUs
        QUANTIZE_ARG=""
        CHECKPOINT_DIR="none"
        ;;
    "Llama-3.2-3B-Instruct")
        NGPUS=1
        CHECKPOINT_DIR="../../models/Llama-3.2-3B-Instruct/"
        QUANTIZE_ARG=$([ "$QUANTIZE" = true ] && echo "-q" || echo "")
        #ls "$CHECKPOINT_DIR"
        #echo "QUANTIZE_ARG $QUANTIZE_ARG"
        ;;
    "Llama-3.3-70B-Instruct")
        NGPUS=4
        CHECKPOINT_DIR="../../models/Llama-3.3-70B-Instruct/"
        QUANTIZE_ARG=$([ "$QUANTIZE" = true ] && echo "-q" || echo "")
        #ls "$CHECKPOINT_DIR"
        #echo "QUANTIZE_ARG $QUANTIZE_ARG"
        ;;
    *)
        echo "Invalid model name. Exiting."
        exit 1
        ;;
esac

echo "Starting TrialGPT pipeline..."

# Get the last completed step
LAST_STEP=$(get_last_step)

# Step 1: TrialGPT-Retrieval - Keyword Generation
if [ "$LAST_STEP" -lt 1 ]; then
    echo "Step 1: Keyword Generation"
    measure_time python trialgpt_retrieval/keyword_generation.py -c $CORPUS -m $MODEL -g $NGPUS -d $CHECKPOINT_DIR $QUANTIZE_ARG
    duration=$?
    update_checkpoint "1" $duration
else
    echo "Skipping Step 1: Keyword Generation Already completed"
fi

# Step 2: TrialGPT-Retrieval - Hybrid Fusion Retrieval and Generate Retrieved Trials
if [ "$LAST_STEP" -lt 2 ]; then
    echo "Step 2: Hybrid Fusion Retrieval and Generate Retrieved Trials"
    measure_time python trialgpt_retrieval/hybrid_fusion_retrieval.py $CORPUS $MODEL $K $BM25_WEIGHT $MEDCPT_WEIGHT $OVER_WRITE $TOP_K $BATCH_SIZE $ELIGIBILITY $EXCLUSION
    duration=$?
    update_checkpoint "2" $duration
else
    echo "Skipping Step 2: Hybrid Fusion Already completed"
fi

# Step 3: TrialGPT-Matching
if [ "$LAST_STEP" -lt 3 ]; then
    echo "Step 3: TrialGPT-Matching"
    measure_time python trialgpt_matching/run_matching.py $CORPUS $MODEL $OVER_WRITE -g $NGPUS -d $CHECKPOINT_DIR $QUANTIZE_ARG
    duration=$?
    update_checkpoint "3" $duration
else
    echo "Skipping Step 3: TrialGPT-Matching Already completed"
fi

# Step 4: TrialGPT-Ranking - Aggregation
if [ "$LAST_STEP" -lt 4 ]; then
    echo "Step 4: TrialGPT-Ranking Aggregation"
    MATCHING_RESULTS="results/matching_results_${CORPUS}_${MODEL}.json"
    measure_time python trialgpt_ranking/run_aggregation.py $CORPUS $MODEL $MATCHING_RESULTS $OVER_WRITE -g $NGPUS -d $CHECKPOINT_DIR $QUANTIZE_ARG
    duration=$?
    update_checkpoint "4" $duration
else
    echo "Skipping Step 4: Aggregation Already completed"
fi

# Step 5: TrialGPT-Ranking - Final Ranking
if [ "$LAST_STEP" -lt 5 ]; then
    echo "Step 5: TrialGPT-Ranking Final Ranking"
    MATCHING_RESULTS="results/matching_results_${CORPUS}_${MODEL}.json"
    AGGREGATION_RESULTS="results/aggregation_results_${CORPUS}_${MODEL}.json"
    measure_time python trialgpt_ranking/rank_results.py $MATCHING_RESULTS $AGGREGATION_RESULTS $OVER_WRITE $CORPUS $MODEL
    duration=$?
    update_checkpoint "5" $duration
else
    echo "Skipping Step 5: Final Ranking Already completed"
fi

echo "TrialGPT pipeline completed successfully!"
echo "Checkpoint file contents:"
cat "$CHECKPOINT_FILE"
