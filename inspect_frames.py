#!/usr/bin/env python3
"""Deep inspection of FrameData from different game types."""

import json
import os
import sys

import arc_agi
from arcengine import FrameData, GameAction, GameState

API_KEY = os.environ.get("ARC_API_KEY", "")

def inspect_game(arc, gid, steps=3):
    """Reset a game, capture frame, take a few steps, show diffs."""
    print(f"\n{'='*60}")
    print(f"Inspecting {gid}")
    print(f"{'='*60}")
    
    env = arc.make(gid)
    actions = getattr(env, 'action_space', [])
    print(f"Actions: {[str(a) for a in actions]}")
    
    obs = env.step(GameAction.RESET)
    frame = obs.frame
    
    # Frame structure
    if frame:
        print(f"Frame shape: [{len(frame)}][{len(frame[0])}][{len(frame[0][0])}]")
        print(f"Frame[0][0]: {frame[0][0]}")
        print(f"Frame[0][32]: {frame[0][32]}")
    else:
        print("Frame is EMPTY after reset!")
        # Try stepping an action
        if actions:
            obs2 = env.step(actions[0])
            if obs2.frame:
                print(f"Frame after {actions[0]}: [{len(obs2.frame)}][{len(obs2.frame[0])}][{len(obs2.frame[0][0])}]")
                print(f"Frame[0][0]: {obs2.frame[0][0]}")
                print(f"Frame[0][32]: {obs2.frame[0][32]}")
    
    print(f"State: {obs.state}")
    print(f"Levels completed: {obs.levels_completed}")
    print(f"Win levels: {obs.win_levels}")
    print(f"Available actions: {obs.available_actions}")
    print(f"Action input: id={obs.action_input.id}, data={obs.action_input.data}")
    print(f"GUID: {obs.guid}")
    
    # Take a few steps and show state changes
    env.step(GameAction.RESET)
    prev_state = None
    for i, action in enumerate(actions[:steps]):
        obs = env.step(action)
        has_frame = bool(obs.frame)
        frame_dims = f"[{len(obs.frame)}][{len(obs.frame[0])}][{len(obs.frame[0][0])}]" if obs.frame else "None"
        state_change = "NEW!" if obs.state != prev_state else ""
        print(f"  Step {i+1}: {action} -> state={obs.state} {state_change} | frame={frame_dims} | avail={obs.available_actions}")
        prev_state = obs.state
        
        if obs.state == GameState.WIN:
            print(f"  🎉 WON after {i+1} steps!")
            break
        if obs.state == GameState.GAME_OVER:
            print(f"  💀 GAME OVER after {i+1} steps!")
            break


def main():
    arc = arc_agi.Arcade(arc_api_key=API_KEY)
    envs = arc.get_environments()
    
    # Pick one from each category: keyboard, keyboard_click, click
    pick_ids = {
        "keyboard": "G50T",
        "keyboard_click": "AR25",
        "click": "LF52",
        "1-action": "TN36",
    }
    
    env_map = {e.title: e.game_id for e in envs}
    
    for cat, title in pick_ids.items():
        if title in env_map:
            inspect_game(arc, env_map[title], steps=5)
        else:
            print(f"\nGame {title} not found!")


if __name__ == "__main__":
    main()
