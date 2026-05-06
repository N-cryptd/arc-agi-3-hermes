#!/usr/bin/env python3
"""Explore all 25 public ARC-AGI-3 games — probe actions, capture state, classify."""

import json
import os
import sys

import arc_agi
from arcengine import FrameData, GameAction, GameState

API_KEY = os.environ.get("ARC_API_KEY", "")


def explore_game(arc, env_info) -> dict:
    """Explore a single game."""
    gid = env_info.game_id
    result = {
        "game_id": gid,
        "title": env_info.title,
        "tags": list(env_info.tags) if env_info.tags else [],
        "baseline_actions": env_info.baseline_actions if hasattr(env_info, 'baseline_actions') else [],
        "actions": [],
        "action_count": 0,
        "grid_size": [64, 64],
        "objects_on_start": [],
        "probed_effects": {},
        "error": None,
    }

    try:
        env = arc.make(gid)
        if env is None:
            result["error"] = "env is None"
            return result

        result["actions"] = getattr(env, 'action_space', [])
        result["action_count"] = len(result["actions"])

        # Reset
        obs = env.step(GameAction.RESET)

        # Grid size
        if hasattr(obs, 'width') and hasattr(obs, 'height'):
            result["grid_size"] = [obs.width, obs.height]

        # Frame data extraction
        if hasattr(obs, 'frame'):
            frame = obs.frame
            result["grid_size"] = [
                getattr(frame, 'width', result["grid_size"][0]),
                getattr(frame, 'height', result["grid_size"][1]),
            ]
            # Try to get objects
            if hasattr(frame, 'objects'):
                for obj in frame.objects[:15]:
                    obj_d = {}
                    for attr in ['position', 'object_type', 'id', 'color', 'shape', 'name']:
                        if hasattr(obj, attr):
                            val = getattr(obj, attr)
                            obj_d[attr] = str(val) if val else None
                    result["objects_on_start"].append(obj_d)

        # Probe actions (up to 6)
        for action_name in result["actions"][:6]:
            try:
                env.step(GameAction.RESET)
                r = env.step(action_name)
                result["probed_effects"][action_name] = {
                    "state": str(getattr(r, 'state', '?')),
                    "score": getattr(r, 'score', None),
                }
            except Exception as e:
                result["probed_effects"][action_name] = {"error": str(e)[:80]}

    except Exception as e:
        result["error"] = str(e)[:200]

    return result


def main():
    arc = arc_agi.Arcade(arc_api_key=API_KEY)
    envs = arc.get_environments()
    print(f"Exploring {len(envs)} games...\n")

    results = []
    for i, env_info in enumerate(envs):
        gid = env_info.game_id
        print(f"[{i+1:2d}/25] {env_info.title} ({gid})", end=" ... ", flush=True)
        info = explore_game(arc, env_info)
        results.append(info)

        if info["error"]:
            print(f"❌ {info['error'][:60]}")
        else:
            print(f"✅ {info['action_count']} actions | grid={info['grid_size']} | objects={len(info['objects_on_start'])}")

    # Save
    out = os.path.join(os.path.dirname(__file__), "game_exploration.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {out}")

    # Summary stats
    ok = [r for r in results if not r["error"]]
    print(f"\n{'='*60}")
    print(f"Successful: {len(ok)}/25")
    tag_counts = {}
    for r in ok:
        for t in r["tags"]:
            tag_counts[t] = tag_counts.get(t, 0) + 1
    print(f"Tag distribution: {tag_counts}")
    action_counts = [r["action_count"] for r in ok]
    if action_counts:
        print(f"Actions per game: min={min(action_counts)}, max={max(action_counts)}, avg={sum(action_counts)/len(action_counts):.1f}")
    grid_sizes = set(tuple(r["grid_size"]) for r in ok)
    print(f"Grid sizes: {grid_sizes}")


if __name__ == "__main__":
    main()
