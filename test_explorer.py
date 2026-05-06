#!/usr/bin/env python3
"""Test Explorer on representative games."""

import os
import arc_agi
from arcengine import GameAction
from explorer import Explorer

API_KEY = os.environ.get("ARC_API_KEY", "")

def test_explorer(arc, gid, title, tags, budget=20):
    print(f"\n{'='*55}")
    print(f"  Exploring {title} ({gid}) — tags={tags}")
    print(f"{'='*55}")

    env = arc.make(gid)
    exp = Explorer(env, gid, tags, budget=budget)
    profile = exp.explore()

    print(profile.summary())
    print()
    print("Action descriptions:")
    print(exp.get_action_description())
    return profile


def main():
    arc = arc_agi.Arcade(arc_api_key=API_KEY)
    envs = arc.get_environments()
    env_map = {e.game_id: e for e in envs}

    # One from each category
    tests = [
        ("g50t-5849a774", "keyboard"),      # Movement
        ("ar25-0c556536", "keyboard_click"), # Movement + click
        ("tn36-ef4dde99", "click"),           # Click only
        ("lf52-271a04aa", "click"),           # Click with coordinates
        ("sp80-589a99af", "keyboard_click"),  # Movement with few objects
    ]

    for gid, tag in tests:
        if gid not in env_map:
            continue
        info = env_map[gid]
        tags = list(info.tags) if info.tags else []
        test_explorer(arc, gid, info.title, tags, budget=15)


if __name__ == "__main__":
    main()
