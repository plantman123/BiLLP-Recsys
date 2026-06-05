MAX_ITER=30

python3 ./evaluate_agent_runs.py \
  output/trajs_agent/*memory_full*.json \
  output/trajs_agent/*memory_fifo20*.json \
  output/trajs_agent/*memory_lru20*.json \
  --max-iteration "$MAX_ITER"
  