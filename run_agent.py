#!/usr/bin/env python3
"""Full integration test — run HermesAgent on a single game."""

import logging
import os
import sys

import arc_agi
from arcengine import GameAction

from executor import AdaptiveExecutor, RunStats
from llm_reasoner import LLMReasoner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("hermes_agent")


def run_game(arc, game_id: str, title: str, tags: list[str],
             max_actions: int = 30, exploration_budget: int = 10) -> RunStats:
    """Run the full agent on a single game."""
    print(f"\n{'#'*60}")
    print(f"#  HermesAgent: {title} ({game_id})")
    print(f"#  Max actions: {max_actions}, Exploration: {exploration_budget}")
    print(f"{'#'*60}\n")

    env = arc.make(game_id)

    # Create reasoner
    nim_key = os.environ.get("NVIDIA_NIM_API_KEY", "")
    reasoner = LLMReasoner(
        api_key=nim_key,
        model="z-ai/glm-5.1",
        timeout=90,  # shorter for testing from Termux
    )

    # Create and run executor
    executor = AdaptiveExecutor(
        env=env,
        game_id=game_id,
        tags=tags,
        reasoner=reasoner,
        max_actions=max_actions,
        exploration_budget=exploration_budget,
    )

    stats = executor.run()
    return stats


def main():
    API_KEY = os.environ.get("ARC_API_KEY", "")
    arc = arc_agi.Arcade(arc_api_key=API_KEY)
    envs = arc.get_environments()

    # Quick test: run on a few games with limited budget
    test_games = [
        # (game_id, title, tags, max_actions, exploration_budget)
        ("ar25-0c556536", "AR25", ["keyboard_click"], 25, 10),
    ]

    env_map = {e.game_id: e for e in envs}

    all_stats = []
    for gid, title, tags, max_a, exp_b in test_games:
        if gid not in env_map:
            continue
        # Get actual tags from env_map
        actual_tags = list(env_map[gid].tags) if env_map[gid].tags else tags
        stats = run_game(arc, gid, title, actual_tags, max_actions=max_a,
                         exploration_budget=exp_b)
        all_stats.append(stats)

    # Summary
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    for s in all_stats:
        print(f"  {s.summary()}")


if __name__ == "__main__":
    main()
