#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# container: docker.io/cphsieh/ruler:0.1.0
# bash run.sh MODEL_NAME BENCHMARK_NAME

set -e

if [ $# -ne 3 ]; then
    echo "Usage: $0 <algo_config_path> $1 <algo_name> $2 <benchmark_name>"
    exit 1
fi


# Root Directories
# export CUDA_VISIBLE_DEVICES=1
ROOT_DIR="${TWILIGHT_RULER_ROOT_DIR:-./dataset}" # generated samples and predictions.

ENGINE_DIR="" # the path that contains individual engine folders from TensorRT-LLM.

ALGO_CONFIG_PATH=${1}
ALGO_NAME=${2}
TASK_BENCHMARK=${3}
DATA_BENCHMARK="${TWILIGHT_RULER_DATA_BENCHMARK:-synthetic}"
RANDOM_SEED="${TWILIGHT_RULER_RANDOM_SEED:-42}"

# Model and Tokenizer
source config_models.sh
if [ -n "${TWILIGHT_SEQ_LENGTHS:-}" ]; then
    read -r -a SEQ_LENGTHS <<< "${TWILIGHT_SEQ_LENGTHS}"
fi
MODEL_DIR="${TWILIGHT_MODEL_DIR:-/data2/lsl_data/models/Meta-Llama-3.1-8B-Instruct}" # the path that contains individual model folders from HuggingFace.
MODEL_NAME="${TWILIGHT_MODEL_NAME:-llama3-8b-chat-128k}"

# MODEL_DIR="/data2/lsl_data/models/Qwen3-8B"
# MODEL_NAME=qwen3-8b

MODEL_CONFIG=$(MODEL_SELECT ${MODEL_NAME} ${MODEL_DIR} ${ENGINE_DIR})
IFS=":" read MODEL_PATH MODEL_TEMPLATE_TYPE MODEL_FRAMEWORK TOKENIZER_PATH TOKENIZER_TYPE OPENAI_API_KEY GEMINI_API_KEY AZURE_ID AZURE_SECRET AZURE_ENDPOINT <<< "$MODEL_CONFIG"
if [ -z "${MODEL_PATH}" ]; then
    echo "Model: ${MODEL_NAME} is not supported"
    exit 1
fi


export OPENAI_API_KEY=${OPENAI_API_KEY}
export GEMINI_API_KEY=${GEMINI_API_KEY}
export AZURE_API_ID=${AZURE_ID}
export AZURE_API_SECRET=${AZURE_SECRET}
export AZURE_API_ENDPOINT=${AZURE_ENDPOINT}


# Benchmark and Tasks
source config_tasks.sh
if [ -n "${TWILIGHT_RULER_TASKS:-}" ]; then
    read -r -a SELECTED_TASKS <<< "${TWILIGHT_RULER_TASKS}"
else
    declare -n SELECTED_TASKS=$TASK_BENCHMARK
fi
if [ "${#SELECTED_TASKS[@]}" -eq 0 ]; then
    echo "Benchmark: ${TASK_BENCHMARK} is not supported"
    exit 1
fi



# Start client (prepare data / call model API / obtain final metrics)
for MAX_SEQ_LENGTH in "${SEQ_LENGTHS[@]}"; do
    
    RESULTS_DIR="${ROOT_DIR}/${MODEL_NAME}/${TASK_BENCHMARK}/${MAX_SEQ_LENGTH}"
    DATA_DIR="${RESULTS_DIR}/data/"
    PRED_DIR="${RESULTS_DIR}/pred/${ALGO_NAME}/"
    mkdir -p ${DATA_DIR}
    mkdir -p ${PRED_DIR}
    
    for TASK in "${SELECTED_TASKS[@]}"; do
        if [ "${TWILIGHT_RULER_PROFILE:-0}" = "1" ]; then
            export TWILIGHT_PROFILE="1"
            export TWILIGHT_PROFILE_MODEL="${MODEL_PATH}"
            export TWILIGHT_PROFILE_CONFIG="${ALGO_CONFIG_PATH}"
            export TWILIGHT_PROFILE_OUTPUT="${PRED_DIR}/${TASK}.profile.jsonl"
            export TWILIGHT_PROFILE_RAW="${TWILIGHT_RULER_PROFILE_RAW:-0}"
        else
            unset TWILIGHT_PROFILE TWILIGHT_PROFILE_MODEL TWILIGHT_PROFILE_CONFIG TWILIGHT_PROFILE_OUTPUT TWILIGHT_PROFILE_RAW
        fi

        python data/prepare.py \
            --save_dir ${DATA_DIR} \
            --benchmark ${DATA_BENCHMARK} \
            --task ${TASK} \
            --tokenizer_path ${TOKENIZER_PATH} \
            --tokenizer_type ${TOKENIZER_TYPE} \
            --max_seq_length ${MAX_SEQ_LENGTH} \
            --model_template_type ${MODEL_TEMPLATE_TYPE} \
            --num_samples ${NUM_SAMPLES} \
            --random_seed ${RANDOM_SEED} \
            ${REMOVE_NEWLINE_TAB}
        
        python pred/call_api.py \
            --data_dir ${DATA_DIR} \
            --save_dir ${PRED_DIR} \
            --benchmark ${DATA_BENCHMARK} \
            --task ${TASK} \
            --server_type ${MODEL_FRAMEWORK} \
            --model_name_or_path ${MODEL_PATH} \
            --temperature ${TEMPERATURE} \
            --top_k ${TOP_K} \
            --top_p ${TOP_P} \
            --algo_config_path ${ALGO_CONFIG_PATH}  \
            ${STOP_WORDS}
    done
    
    python eval/evaluate.py \
        --data_dir ${PRED_DIR} \
        --benchmark ${DATA_BENCHMARK}
done
