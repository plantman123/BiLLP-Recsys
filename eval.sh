MAX_ITER=30

# memory 
python3 ./evaluate_agent_runs.py \
  output/memory_end_50_iter_50_size_10/trajs_agent/*memory_full*.json \
  output/memory_end_50_iter_50_size_10/trajs_agent/*memory_fifo20*.json \
  output/memory_end_50_iter_50_size_10/trajs_agent/*memory_lru20*.json \
  --max-iteration "$MAX_ITER"

# topk
# python3 ./evaluate_agent_runs.py \
#   output/trajs_agent/*grounding_original*.json \
#   output/trajs_agent/*grounding_rerank_top5*.json \
#   --max-iteration "${MAX_ITER}"
#   # --max-iteration "$MAX_ITER"