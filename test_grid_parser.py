#!/usr/bin/env python3
"""Test GridParser & StateTracker on all 25 games."""

import json
import os
import sys

import arc_agi
from arcengine import FrameData, GameAction, GameState
from grid_parser import GridParser, StateTracker

API_KEY = os.environ.get("ARC_API_KEY", "")


def test_game(arc, gid, title):
    """Test parser on a game — reset + 3 actions."""
    print(f"\n{'─'*50}")
    print(f"  {title} ({gid})")
    print(f"{'─'*50}")

    env = arc.make(gid)
    actions = getattr(env, 'action_space', [])
    tracker = StateTracker()

    # Reset
    obs = env.step(GameAction.RESET)
    snap = tracker.update(obs)
    if snap is None:
        print("  ❌ Empty frame after reset")
        return

    print(f"  Grid: {snap.grid_size[1]}x{snap.grid_size[0]} | "
          f"Layers: {snap.num_layers} | "
          f"Objects: {len(snap.objects)} | "
          f"Colors: {sorted(c for c in snap.color_histogram if c != 0)}")

    # Take 3 steps
    for i, action in enumerate(actions[:3]):
        try:
            obs = env.step(action)
            if obs is None:
                print(f"  Step {i+1} {action}: NULL response")
                continue
            snap = tracker.update(obs)
            diff = tracker.last_diff
            print(f"  Step {i+1} {action}: state={obs.state.name} | "
                  f"objs={len(snap.objects)} | {diff.summary()}")
        except Exception as e:
            print(f"  Step {i+1} {action}: ERROR {str(e)[:60]}")

    # Test LLM context output
    ctx = tracker.build_llm_context()
    print(f"  LLM context ({len(ctx)} chars):")
    for line in ctx.split("\n")[:8]:
        print(f"    {line}")

    # Test ASCII render
    ascii_grid = tracker.frame_to_ascii(obs, max_w=32)
    lines = ascii_grid.split("\n")
    print(f"  ASCII preview ({len(lines)} rows):")
    for line in lines[:4]:
        print(f"    {line}")
    print(f"    ...")
    for line in lines[-2:]:
        print(f"    {line}")

    return True


def main():
    arc = arc_agi.Arcade(arc_api_key=API_KEY)
    envs = arc.get_environments()

    ok = 0
    for i, env_info in enumerate(envs):
        try:
            if test_game(arc, env_info.game_id, env_info.title):
                ok += 1
        except Exception as e:
            print(f"\n  ❌ FATAL: {env_info.title} -> {str(e)[:100]}")

    print(f"\n{'='*50}")
    print(f"  Results: {ok}/25 games parsed successfully")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
