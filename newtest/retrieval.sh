#!/bin/bash
set -e
source ./export.sh
export LD_LIBRARY_PATH=/data/songbo/miniforge3/lib:${LD_LIBRARY_PATH:-}

# 日志目录
LOGDIR=./newtest
mkdir -p "$LOGDIR"
mkdir -p output/trajs_agent output/reflections output/memory output/critic_memory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_LOG="$LOGDIR/retrieval_${TIMESTAMP}.log"
exec > >(tee -a "$RUN_LOG") 2>&1

echo "=== Reflection Retrieval Timing Experiment Started at $(date) ==="
echo "Log file: $RUN_LOG"

TASK_END=10
MAX_ITER=20
MAX_TOKENS=8000
BATCH_SIZE=10
CUDA_ID=4
GROUNDING_MODEL_PATH=./model/llama-2-hf-7b
INPUT_MEMORY=steam_test_0_50_deepseek-v4-flash_0.5_retrieval_episode_2026-06-06-09-26-12

for memory_file in \
  "output/reflections/${INPUT_MEMORY}.txt" \
  "output/memory/${INPUT_MEMORY}.json" \
  "output/critic_memory/${INPUT_MEMORY}.json"; do
  if [[ ! -f "$memory_file" ]]; then
    echo "Missing required fixed memory file: $memory_file"
    exit 1
  fi
done

echo "Using fixed initial memory: $INPUT_MEMORY"

COMMON_ARGS=(
  --task steam
  --backend "$BACKEND_MODEL"
  --promptpath cot_movie_upper
  --evaluate
  --random
  --task_split test
  --task_start_index 0
  --task_end_index "$TASK_END"
  --temperature 0.5
  --env steam
  --env_threshold 30
  --env_window_length 4
  --Max_Iteration "$MAX_ITER"
  --agent_name agent_a2c
  --Max_Reflections 2
  --batch_size "$BATCH_SIZE"
  --input_file_name "$INPUT_MEMORY"
  --grounding_model_path "$GROUNDING_MODEL_PATH"
  --max_tokens "$MAX_TOKENS"
  --reflection_memory_policy full
  --reflection_memory_size 0
)

echo "Running Episode Retrieval Baseline..."
CUDA_VISIBLE_DEVICES="$CUDA_ID" python generation_rec_agents.py \
  "${COMMON_ARGS[@]}" \
  --run_name retrieval_episode \
  --reflection_retrieval_mode episode \
  --static_reflection_k 2

echo "Running Dynamic Retrieval Experiment..."
CUDA_VISIBLE_DEVICES="$CUDA_ID" python generation_rec_agents.py \
  "${COMMON_ARGS[@]}" \
  --run_name retrieval_dynamic_k2 \
  --reflection_retrieval_mode dynamic \
  --dynamic_reflection_k 2

echo "Running Hybrid Retrieval Experiment..."
CUDA_VISIBLE_DEVICES="$CUDA_ID" python generation_rec_agents.py \
  "${COMMON_ARGS[@]}" \
  --run_name retrieval_hybrid_s1_d1 \
  --reflection_retrieval_mode hybrid \
  --static_reflection_k 1 \
  --dynamic_reflection_k 1
