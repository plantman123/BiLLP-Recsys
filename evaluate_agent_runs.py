#!/usr/bin/env python3
import argparse
import csv
import glob
import json
import re
from pathlib import Path


REWARD_RE = re.compile(r"reward\s*=\s*(-?\d+(?:\.\d+)?)")
ACTION_RE = re.compile(r"^Action\s+\d+:\s*recommend\[", re.IGNORECASE)


def parse_rewards(lines):
    rewards = []
    for line in lines:
        match = REWARD_RE.search(line)
        if match:
            rewards.append(float(match.group(1)))
    return rewards


def summarize_episode(entry, max_iteration):
    lines = entry.get("traj_by_line") or entry.get("traj", "").splitlines()
    rewards = parse_rewards(lines)
    accepted_rewards = []
    for line in lines:
        if "Episode continue" not in line:
            continue
        match = REWARD_RE.search(line)
        if match:
            accepted_rewards.append(float(match.group(1)))

    action_count = sum(1 for line in lines if ACTION_RE.search(line))
    user_stop = any("User Stop" in line for line in lines)
    invalid = any("Invalid Action" in line for line in lines)
    grounding_replacements = sum(
        1
        for line in lines
        if "can not be recommened, instead, recommend" in line
        or "can not be recommended, instead, recommend" in line
    )
    rerank_count = len(entry.get("grounding_reranks", []))
    if not rerank_count:
        rerank_count = sum(1 for line in lines if line.startswith("Grounding Rerank"))

    reached = False
    if max_iteration is not None:
        reached = (not user_stop) and (not invalid) and action_count >= max_iteration

    return {
        "Len": len(accepted_rewards),
        "R_traj": sum(accepted_rewards),
        "action_len": action_count,
        "raw_rtraj": sum(rewards),
        "user_stop": user_stop,
        "invalid": invalid,
        "reached": reached,
        "grounding_replacements": grounding_replacements,
        "rerank_count": rerank_count,
    }


def mean(values):
    return sum(values) / len(values) if values else 0.0


def rate(values):
    return mean([1.0 if value else 0.0 for value in values])


def summarize_file(path, max_iteration):
    with open(path, "r", encoding="utf-8") as fin:
        data = json.load(fin)

    episodes = [summarize_episode(entry, max_iteration) for entry in data.values()]
    n = len(episodes)
    total_actions = sum(item["action_len"] for item in episodes)
    total_accepted_rounds = sum(item["Len"] for item in episodes)
    total_accepted_reward = sum(item["R_traj"] for item in episodes)

    return {
        "run": Path(path).stem,
        "n": n,
        "Len": mean([item["Len"] for item in episodes]),
        "R_each": (
            total_accepted_reward / total_accepted_rounds
            if total_accepted_rounds
            else 0.0
        ),
        "R_traj": mean([item["R_traj"] for item in episodes]),
        "reach_rate": rate([item["reached"] for item in episodes]) if max_iteration is not None else "",
        "stop_rate": rate([item["user_stop"] for item in episodes]),
        "invalid_rate": rate([item["invalid"] for item in episodes]),
        "action_len": mean([item["action_len"] for item in episodes]),
        "raw_rtraj": mean([item["raw_rtraj"] for item in episodes]),
        "grounding_replace_rate": (
            sum(item["grounding_replacements"] for item in episodes) / total_actions
            if total_actions
            else 0.0
        ),
        "rerank_per_episode": mean([item["rerank_count"] for item in episodes]),
    }


def expand_paths(patterns):
    paths = []
    for pattern in patterns:
        matched = glob.glob(pattern)
        paths.extend(matched if matched else [pattern])
    return sorted(set(paths))


def main():
    parser = argparse.ArgumentParser(
        description="Summarize BiLLP-Recsys agent trajectory JSON files."
    )
    parser.add_argument("files", nargs="+", help="Trajectory JSON files or glob patterns.")
    parser.add_argument(
        "--max-iteration",
        type=int,
        default=None,
        help="Set this to compute Reach as episodes that do not stop before max steps.",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Print CSV instead of a compact aligned table.",
    )
    args = parser.parse_args()

    rows = [summarize_file(path, args.max_iteration) for path in expand_paths(args.files)]
    if not rows:
        return

    columns = list(rows[0].keys())
    if args.csv:
        writer = csv.DictWriter(__import__("sys").stdout, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
        return

    widths = {
        column: max(len(column), *(len(f"{row[column]:.4f}") if isinstance(row[column], float) else len(str(row[column])) for row in rows))
        for column in columns
    }
    print("  ".join(column.ljust(widths[column]) for column in columns))
    for row in rows:
        cells = []
        for column in columns:
            value = row[column]
            text = f"{value:.4f}" if isinstance(value, float) else str(value)
            cells.append(text.ljust(widths[column]))
        print("  ".join(cells))


if __name__ == "__main__":
    main()
