#!/usr/bin/env python3
"""Test LLM Reasoner end-to-end."""

import os
import arc_agi
from arcengine import GameAction, GameState
from grid_parser import StateTracker
from explorer import Explorer
from llm_reasoner import LLMReasoner

API_KEY = os.environ.get("ARC_API_KEY", "")
NIM_KEY = os.environ.get("NVIDIA_NIM_API_KEY", "")


def test_reasoner():
    arc = arc_agi.Arcade(arc_api_key=API_KEY)
    envs = arc.get_environments()
    env_map = {e.game_id: e for e in envs}

    # Use AR25 (keyboard_click, has movement + click)
    gid = "ar25-0c556536"
    info = env_map[gid]
    print(f"Testing LLM Reasoner on {info.title} ({gid})")

    env = arc.make(gid)

    # Phase 1: Explore
    print("\n--- Phase 1: Exploration ---")
    tags = list(info.tags) if info.tags else []
    exp = Explorer(env, gid, tags, budget=15)
    profile = exp.explore()
    print(profile.summary())

    # Phase 2: Initial reasoning
    print("\n--- Phase 2: Initial LLM Reasoning ---")
    tracker = StateTracker()
    obs = env.step(GameAction.RESET)
    tracker.update(obs)

    reasoner = LLMReasoner(api_key=NIM_KEY, model="z-ai/glm-5.1", timeout=120)
    plan = reasoner.reason_initial(exp, tracker)

    print(f"\nRaw response ({len(plan.raw_response)} chars):")
    print(plan.raw_response[:1000])
    print(f"\nParsed:")
    print(f"  Understanding: {plan.understanding[:200] if plan.understanding else 'N/A'}")
    print(f"  Next action: {plan.next_action}")
    print(f"  Confidence: {plan.confidence}")
    print(f"  Plan steps: {len(plan.plan_steps)}")

    # Phase 3: Execute the suggested action and reason again
    if plan.next_action:
        parsed = reasoner.parse_next_action(plan, obs.available_actions)
        if parsed:
            action, data = parsed
            print(f"\n--- Phase 3: Executing {action} with data={data} ---")
            obs2 = env.step(action, data=data)
            if obs2:
                tracker.update(obs2)
                diff = tracker.last_diff
                print(f"  Result: state={obs2.state.name}, diff={diff.summary()}")

                # Reason about the next step
                plan2 = reasoner.reason_step(tracker, str(action), diff.summary())
                print(f"\n  Next reasoning:")
                print(f"  Next action: {plan2.next_action}")
                print(f"  Confidence: {plan2.confidence}")
                print(f"  Tokens used: {reasoner.total_tokens_used}")
            else:
                print("  Action returned None!")
        else:
            print(f"  Could not parse action: {plan.next_action}")
    else:
        print("  No next action suggested!")


if __name__ == "__main__":
    test_reasoner()
