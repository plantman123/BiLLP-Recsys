# BiLLP-Recsys 三组扩展实验说明

本文档说明当前代码新增的三组实验、具体实现、运行命令与统一评估方式：

1. Planner reflection memory 池大小与淘汰策略
2. Grounding 原版选第一个候选 vs Top-k 候选交给 LLM 二次筛选
3. Planner reflection 在 episode 开头检索 vs 每一步动态检索 vs 混合检索

实验代码直接使用原论文项目原本能够运行的环境，不修改原项目的
`requirements.txt`、`recsys.yaml`、OpenAI 调用方式或 embedding 实现。

未传入任何新增实验参数时：

```bash
source run_steam_test.sh
```

仍然走 `original` 路径，保持原论文运行行为、输出文件名和轨迹 JSON 结构。

## 1. 修改文件与实现位置

相对于原仓库，实验相关文件只有：

| 文件 | 作用 |
| --- | --- |
| `Agents/agent_a2c.py` | 实现 memory 策略、grounding rerank、动态 reflection 检索和实验日志 |
| `generation_rec_agents.py` | 增加实验命令行参数、`run_name` 和实验输出文件名 |
| `evaluate_agent_runs.py` | 汇总 `Len`、`R_each`、`R_traj` 等指标 |
| `EXPERIMENT_README.md` | 实验说明和运行命令 |

`agent_a2c.py` 中的主要实现：

| 功能 | 实现 |
| --- | --- |
| Reflection 检索模式 | `get_reflect_str()`、`_refresh_step_reflections()` |
| 当前状态 query | `_current_reflection_query()` |
| FIFO/LRU memory | `_apply_reflection_memory_policy()` |
| LRU 访问更新 | `_mark_reflections_used()` |
| Grounding LLM 二筛 | `_select_grounded_item()` |
| Grounding 独立日志 | `_record_grounding_rerank()` |
| 实验配置与检索日志 | `_experiment_config()`、`_build_info()` |

Grounding Top-k 候选列表不会写入 Actor scratchpad，只保存在结果 JSON 的
`grounding_reranks` 中，因此不会让 rerank 版本在后续推荐中额外看到候选列表。

## 2. 新增参数

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--run_name` | 空 | 输出文件名中的实验标签 |
| `--max_tokens` | `6000` | LLM 最大输出 token 数 |
| `--reflection_memory_policy` | `full` | `full`、`fifo` 或 `lru` |
| `--reflection_memory_size` | `0` | FIFO/LRU 最大 reflection 数；`0` 表示不裁剪 |
| `--rerank_after_grounding` | 关闭 | 是否启用 Grounding 后 LLM 二次筛选 |
| `--grounding_topk` | `5` | 二次筛选时展示给 LLM 的候选数量 |
| `--reflection_retrieval_mode` | `original` | `original`、`episode`、`dynamic` 或 `hybrid` |
| `--static_reflection_k` | `0` | episode 开头检索数量；`0` 使用 `Max_Reflections` |
| `--dynamic_reflection_k` | `0` | 当前状态动态检索数量；`0` 使用 `Max_Reflections` |
| `--reflection_query_window` | `15` | 动态 query 保留的最近物品数量 |

`original` 表示完全走原代码检索路径。只有显式传入其他实验参数时才启用扩展实验。

## 3. 公共运行参数

进入项目目录，并加载原来已经能运行项目的环境变量：

```bash
cd /home/pingan666/Homework/BiLLP-Recsys
source /path/to/your/export.sh
```

确认环境变量：

```bash
echo "$OPENAI_API_BASE"
echo "$BACKEND_MODEL"
```

定义三组实验共同使用的参数：

```bash
TASK_END=10
MAX_ITER=70
MAX_TOKENS=6000
BATCH_SIZE=10
GROUNDING_MODEL_PATH=./model/llama-2-hf-7b

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
  --env_threshold 50
  --env_window_length 4
  --Max_Iteration "$MAX_ITER"
  --agent_name agent_a2c
  --Max_Reflections 2
  --batch_size "$BATCH_SIZE"
  --input_file_name steam_train_0_100_gpt-3.5-turbo-16k_0.5_2024-01-04-18-41-25
  --grounding_model_path "$GROUNDING_MODEL_PATH"
  --max_tokens "$MAX_TOKENS"
)
```

如果你原来使用的 `grounding_model_path` 不同，继续使用原来的路径。

正式实验前可先做 smoke test：

```bash
TASK_END=2
MAX_ITER=3
MAX_TOKENS=500
BATCH_SIZE=1
```

正式比较时，三组对照必须使用相同的：

- 测试用户范围与顺序
- 初始 reflection、actor、critic memory
- backend model、temperature、token 限制
- `Max_Iteration`、`Max_Reflections`、`batch_size`
- Grounding 模型和环境参数

## 4. 实验一：Reflection Memory 池

### 4.1 实验问题

研究 Planner reflection memory 保留全部内容，或仅保留有限数量内容时，对推荐轨迹的影响。

实现策略：

| 策略 | 行为 |
| --- | --- |
| `full` | 原版行为，保留所有 reflections |
| `fifo` | 仅保留最近加入的 N 条 reflections |
| `lru` | 保留最近被检索使用的 N 条 reflections |

Memory 会在初始化加载后和每批 episode 产生新 reflection 后执行裁剪。

为了让 LRU 能记录检索访问，本实验统一使用：

```text
reflection_retrieval_mode=episode
static_reflection_k=2
```

该模式仍然只在 episode 开头检索一次，与原版检索时机一致。

### 4.2 Full memory baseline

```bash
CUDA_VISIBLE_DEVICES=0 python generation_rec_agents.py \
  "${COMMON_ARGS[@]}" \
  --run_name memory_full \
  --reflection_retrieval_mode episode \
  --static_reflection_k 2 \
  --reflection_memory_policy full \
  --reflection_memory_size 0
```

### 4.3 FIFO-20

```bash
CUDA_VISIBLE_DEVICES=0 python generation_rec_agents.py \
  "${COMMON_ARGS[@]}" \
  --run_name memory_fifo20 \
  --reflection_retrieval_mode episode \
  --static_reflection_k 2 \
  --reflection_memory_policy fifo \
  --reflection_memory_size 20
```

### 4.4 LRU-20

```bash
CUDA_VISIBLE_DEVICES=0 python generation_rec_agents.py \
  "${COMMON_ARGS[@]}" \
  --run_name memory_lru20 \
  --reflection_retrieval_mode episode \
  --static_reflection_k 2 \
  --reflection_memory_policy lru \
  --reflection_memory_size 20
```

### 4.5 可选：不同 memory 大小

```bash
for SIZE in 10 20 50; do
  CUDA_VISIBLE_DEVICES=0 python generation_rec_agents.py \
    "${COMMON_ARGS[@]}" \
    --run_name "memory_fifo${SIZE}" \
    --reflection_retrieval_mode episode \
    --static_reflection_k 2 \
    --reflection_memory_policy fifo \
    --reflection_memory_size "$SIZE"
done
```

LRU 初始 memory 尚未产生真实访问记录，因此第一次裁剪可能与 FIFO 接近；随着 episode
检索发生，LRU 才会逐渐表现为按访问热度保留。

### 4.6 检查 memory 是否生效

```bash
wc -l reflections/*memory_full*.txt
wc -l reflections/*memory_fifo20*.txt
wc -l reflections/*memory_lru20*.txt
```

FIFO-20 和 LRU-20 最终 reflection 文件行数应不超过 20。

### 4.7 评估 memory 实验

```bash
python3 evaluate_agent_runs.py \
  trajs_agent/*memory_full*.json \
  trajs_agent/*memory_fifo20*.json \
  trajs_agent/*memory_lru20*.json \
  --max-iteration "$MAX_ITER"
```

判断方式：

- 有限 memory 的 `Len`、`R_each`、`R_traj` 与 Full 接近或更高：说明可以减少 memory 成本。
- LRU 高于 FIFO：说明按检索访问保留 reflection 比仅保留最新 reflection 更有效。
- 有限 memory 明显下降：说明长期 reflection 对当前任务仍然重要。

## 5. 实验二：Grounding Top-k + LLM 二次筛选

### 5.1 实验问题

原版 Actor 先输出一个推荐动作，Grounding 找到真实目录中的相近物品，然后直接返回第一个候选。

扩展版本把 Grounding 返回的 Top-k 候选交给 LLM，再从中选择最终推荐物品。

实现流程：

```text
Actor 输出 recommend[item]
    -> Grounding 返回 Top-k 真实候选
    -> LLM 根据历史、当前轨迹、reflection、actor memory 二次选择
    -> 校验输出必须来自候选列表
    -> 解析失败时回退到第一个候选
```

候选、最终选择和是否 fallback 会独立保存到：

```text
trajs_agent/*.json -> grounding_reranks
```

注意：这里的 baseline 是原论文 Grounding 行为。原论文仍请求
`Max_Iteration` 个候选，但直接选择第一个，因此本实验比较的是：

```text
原版选择第一个候选 vs LLM 从 Top-k 中二次筛选
```

不是 Grounding 检索计算量上的 `k=1 vs k=5`。

### 5.2 原版 Grounding baseline

不传 `--rerank_after_grounding`：

```bash
CUDA_VISIBLE_DEVICES=0 python generation_rec_agents.py \
  "${COMMON_ARGS[@]}" \
  --run_name grounding_original \
  --reflection_retrieval_mode original \
  --reflection_memory_policy full \
  --reflection_memory_size 0
```

### 5.3 Top-5 + LLM rerank

```bash
CUDA_VISIBLE_DEVICES=0 python generation_rec_agents.py \
  "${COMMON_ARGS[@]}" \
  --run_name grounding_rerank_top5 \
  --reflection_retrieval_mode original \
  --reflection_memory_policy full \
  --reflection_memory_size 0 \
  --rerank_after_grounding \
  --grounding_topk 5
```

### 5.4 可选：Top-10 + LLM rerank

```bash
CUDA_VISIBLE_DEVICES=0 python generation_rec_agents.py \
  "${COMMON_ARGS[@]}" \
  --run_name grounding_rerank_top10 \
  --reflection_retrieval_mode original \
  --reflection_memory_policy full \
  --reflection_memory_size 0 \
  --rerank_after_grounding \
  --grounding_topk 10
```

### 5.5 检查 rerank 是否生效

```bash
python3 - <<'PY'
import glob
import json

path = sorted(glob.glob("trajs_agent/*grounding_rerank_top5*.json"))[-1]
data = json.load(open(path, encoding="utf-8"))
records = [r for episode in data.values() for r in episode.get("grounding_reranks", [])]
print("file:", path)
print("rerank count:", len(records))
print("example:", records[0] if records else None)
PY
```

### 5.6 评估 Grounding 实验

```bash
python3 evaluate_agent_runs.py \
  trajs_agent/*grounding_original*.json \
  trajs_agent/*grounding_rerank_top5*.json \
  trajs_agent/*grounding_rerank_top10*.json \
  --max-iteration "$MAX_ITER"
```

除三个主指标外，重点查看：

| 指标 | 含义 |
| --- | --- |
| `grounding_replace_rate` | Actor 输出被 Grounding 替换的比例 |
| `rerank_per_episode` | 每个 episode 的平均 LLM rerank 次数 |

若 rerank 提高 `R_each` 和 `R_traj`，说明二次筛选提高了最终推荐质量；同时需要记录额外
LLM 调用带来的运行时间和成本。

## 6. 实验三：Planner Reflection 检索时机

### 6.1 实验问题

比较 Planner reflection 仅在 episode 开头检索，还是根据当前推荐状态动态刷新。

| 模式 | 检索行为 |
| --- | --- |
| `episode` | 用初始状态 `s1` 检索一次，后续步骤复用 |
| `dynamic` | 每一步使用当前状态 `sn` 重新检索 |
| `hybrid` | 同时使用 `s1` 的静态 reflection 和 `sn` 的动态 reflection |

状态定义：

```text
s1 = 用户进入 episode 前的历史物品
sn = 初始历史物品 + 当前 episode 已经推荐过的物品
```

`episode` 直接使用原论文的 `self.task[id]` query。`dynamic` 使用相同句式更新当前状态，
并保留最近 `reflection_query_window=15` 个物品。

为控制 Planner 每一步看到的 reflection 总数：

```text
episode: 2
dynamic: 2
hybrid: static 1 + dynamic 1
```

本实验统一使用 Full memory，并关闭 Grounding rerank。

### 6.2 Episode-start baseline

```bash
CUDA_VISIBLE_DEVICES=0 python generation_rec_agents.py \
  "${COMMON_ARGS[@]}" \
  --run_name retrieval_episode \
  --reflection_retrieval_mode episode \
  --static_reflection_k 2 \
  --reflection_query_window 15 \
  --reflection_memory_policy full \
  --reflection_memory_size 0
```

### 6.3 Dynamic retrieval

```bash
CUDA_VISIBLE_DEVICES=0 python generation_rec_agents.py \
  "${COMMON_ARGS[@]}" \
  --run_name retrieval_dynamic \
  --reflection_retrieval_mode dynamic \
  --dynamic_reflection_k 2 \
  --reflection_query_window 15 \
  --reflection_memory_policy full \
  --reflection_memory_size 0
```

### 6.4 Hybrid retrieval

```bash
CUDA_VISIBLE_DEVICES=0 python generation_rec_agents.py \
  "${COMMON_ARGS[@]}" \
  --run_name retrieval_hybrid \
  --reflection_retrieval_mode hybrid \
  --static_reflection_k 1 \
  --dynamic_reflection_k 1 \
  --reflection_query_window 15 \
  --reflection_memory_policy full \
  --reflection_memory_size 0
```

Hybrid 动态检索会排除已被静态检索选中的 reflection，避免把同一内容重复放入 Planner
提示词。

### 6.5 检查检索模式是否生效

每个实验 episode 的 JSON 会保存：

```text
experiment_config
reflection_retrievals
```

检查最近一次运行：

```bash
python3 - <<'PY'
import glob
import json

for name in ["retrieval_episode", "retrieval_dynamic", "retrieval_hybrid"]:
    path = sorted(glob.glob(f"trajs_agent/*{name}*.json"))[-1]
    data = json.load(open(path, encoding="utf-8"))
    first = next(iter(data.values()))
    records = first.get("reflection_retrievals", [])
    print(name, [(r["scope"], r["step"], len(r["selected_reflections"])) for r in records])
PY
```

预期：

- `episode`：每个用户仅有一条 `scope=episode`。
- `dynamic`：每一步有一条 `scope=dynamic`。
- `hybrid`：一条 `scope=episode`，并且每一步有一条 `scope=dynamic`。

### 6.6 评估检索时机实验

```bash
python3 evaluate_agent_runs.py \
  trajs_agent/*retrieval_episode*.json \
  trajs_agent/*retrieval_dynamic*.json \
  trajs_agent/*retrieval_hybrid*.json \
  --max-iteration "$MAX_ITER"
```

判断方式：

- Dynamic 高于 Episode：当前状态动态变化时重新检索 reflection 有效。
- Hybrid 高于 Dynamic：初始长期偏好与当前短期状态同时有价值。
- Episode 更高：动态 query 可能引入噪声，或当前 reflection memory 不适合细粒度检索。

## 7. 统一评估脚本

### 7.1 输出指标

`evaluate_agent_runs.py` 从 `trajs_agent/*.json` 汇总：

| 指标 | 含义 | 趋势 |
| --- | --- | --- |
| `Len` | 每个 episode 的平均有效推荐轮数，不含用户离开时的 `-1000` 轮 | 越高越好 |
| `R_each` | 所有有效推荐轮次的平均 reward | 越高越好 |
| `R_traj` | 每个 episode 的平均累计有效 reward，不含 `-1000` | 越高越好 |
| `reach_rate` | 未提前退出并达到 `Max_Iteration` 的比例 | 越高越好 |
| `stop_rate` | 用户提前退出比例 | 越低越好 |
| `invalid_rate` | 非法 Actor action 的 episode 比例 | 越低越好 |
| `action_len` | 包含最终失败轮的平均 action 数 | 辅助分析 |
| `raw_rtraj` | 包含 `-1000` 的原始累计 reward | 辅助分析 |
| `grounding_replace_rate` | Actor 推荐被 Grounding 替换的比例 | Grounding 分析 |
| `rerank_per_episode` | 每个 episode 的平均 rerank 次数 | Grounding 成本分析 |

主要结论应基于：

```text
Len、R_each、R_traj
```

### 7.2 评估单个文件

```bash
python3 evaluate_agent_runs.py \
  trajs_agent/你的结果文件.json \
  --max-iteration "$MAX_ITER"
```

### 7.3 评估多个文件

```bash
python3 evaluate_agent_runs.py \
  trajs_agent/*memory_*.json \
  trajs_agent/*grounding_*.json \
  trajs_agent/*retrieval_*.json \
  --max-iteration "$MAX_ITER"
```

### 7.4 输出 CSV

```bash
python3 evaluate_agent_runs.py \
  trajs_agent/*memory_*.json \
  trajs_agent/*grounding_*.json \
  trajs_agent/*retrieval_*.json \
  --max-iteration "$MAX_ITER" \
  --csv > experiment_results.csv
```

## 8. 如何公平比较

每次只改变当前实验研究的变量：

| 实验 | 改变 | 必须固定 |
| --- | --- | --- |
| Memory | policy、memory size | episode 检索、无 rerank、其他参数 |
| Grounding | 是否 rerank、Top-k | original 检索、Full memory、其他参数 |
| Retrieval | episode/dynamic/hybrid | Full memory、无 rerank、其他参数 |

建议每个配置至少运行 3 次，报告均值与标准差。所有配置必须使用相同用户范围和相同初始
memory 文件，否则结果不能直接比较。

## 9. 结果记录模板

### Memory

| 配置 | Len | R_each | R_traj | stop_rate | 最终 memory 数量 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full |  |  |  |  |  |
| FIFO-20 |  |  |  |  |  |
| LRU-20 |  |  |  |  |  |

### Grounding

| 配置 | Len | R_each | R_traj | grounding_replace_rate | rerank_per_episode |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original |  |  |  |  |  |
| Top-5 Rerank |  |  |  |  |  |
| Top-10 Rerank |  |  |  |  |  |

### Retrieval

| 配置 | Len | R_each | R_traj | reach_rate | stop_rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Episode |  |  |  |  |  |
| Dynamic |  |  |  |  |  |
| Hybrid |  |  |  |  |  |
