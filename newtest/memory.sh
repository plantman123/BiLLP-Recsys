source ./export.sh

# 日志目录
LOGDIR=./newtest
mkdir -p "$LOGDIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_LOG="$LOGDIR/memory_${TIMESTAMP}.log"
exec > >(tee -a "$RUN_LOG") 2>&1

echo "=== Memory Experiment Started at $(date) ==="
echo "Log file: $RUN_LOG"

TASK_END=50
MAX_ITER=30
MAX_TOKENS=6000
BATCH_SIZE=10
GROUNDING_MODEL_PATH=./model/shakechen/Llama-2-7b-hf

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
  --input_file_name steam_train_0_100_gpt-3.5-turbo-16k_0.5_2024-01-04-18-41-25
  --grounding_model_path "$GROUNDING_MODEL_PATH"
  --max_tokens "$MAX_TOKENS"
)

echo "Running Baseline Experiment..."
CUDA_VISIBLE_DEVICES=0 python generation_rec_agents.py \
  "${COMMON_ARGS[@]}" \
  --run_name memory_full \
  --reflection_retrieval_mode episode \
  --static_reflection_k 2 \
  --reflection_memory_policy full \
  --reflection_memory_size 0

echo "Running FIFO Memory Experiment..."
CUDA_VISIBLE_DEVICES=0 python generation_rec_agents.py \
  "${COMMON_ARGS[@]}" \
  --run_name memory_fifo20 \
  --reflection_retrieval_mode episode \
  --static_reflection_k 2 \
  --reflection_memory_policy fifo \
  --reflection_memory_size 8

echo "Running LRU Memory Experiment..."
CUDA_VISIBLE_DEVICES=0 python generation_rec_agents.py \
  "${COMMON_ARGS[@]}" \
  --run_name memory_lru20 \
  --reflection_retrieval_mode episode \
  --static_reflection_k 2 \
  --reflection_memory_policy lru \
  --reflection_memory_size 8
