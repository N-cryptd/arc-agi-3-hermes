#!/usr/bin/env python3
"""Minimal ARC-AGI-3 agent runner — no langgraph dependency."""

import argparse
import json
import logging
import os
import sys
import time
from typing import Any, Optional

import arc_agi
from arc_agi import OperationMode
from arcengine import FrameData, GameAction, GameState

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger()


class HermesAgent:
    """ARC-AGI-3 agent with explore-understand-execute loop."""

    MAX_ACTIONS = 80

    def __init__(self, game_id: str, api_key: str = "", competition: bool = False):
        self.game_id = game_id
        self.api_key = api_key
        self.competition = competition
        self.action_count = 0
        self.levels_completed = 0
        self.frames: list[FrameData] = []

        # Exploration tracking
        self.action_effects: dict[str, list[dict]] = {}  # action_name -> list of observed effects
        self.exploration_done = False
        self.explored_actions = set()

        # Game understanding
        self.game_model: dict[str, Any] = {
            "player_pos": None,
            "movable_objects": [],
            "goal_type": None,  # "reach_target", "collect_all", "solve_puzzle", etc.
            "action_mappings": {},  # ACTION1 -> "up", ACTION2 -> "down", etc.
            "grid_size": (64, 64),
        }

        # Strategy
        self.plan: list[GameAction] = []
        self.phase = "EXPLORE"  # EXPLORE -> UNDERSTAND -> EXECUTE

    def run(self) -> dict:
        """Run the agent on the game."""
        mode = OperationMode.COMPETITION if self.competition else None
        kwargs = {}
        if mode:
            kwargs["operation_mode"] = mode

        arc = arc_agi.Arcade(arc_api_key=self.api_key, **kwargs)
        env = arc.make(self.game_id)

        if env is None:
            logger.error(f"Failed to create environment for {self.game_id}")
            return {"error": "Failed to create environment"}

        logger.info(f"Starting {self.game_id} | Actions available: {env.action_space}")

        # Initial reset
        obs = env.step(GameAction.RESET)
        self.frames.append(obs)
        self.action_count += 1

        while self.action_count < self.MAX_ACTIONS:
            frame = self.frames[-1]

            if frame.state == GameState.WIN:
                logger.info(f"🎉 Level {self.levels_completed + 1} WON at action {self.action_count}!")
                self.levels_completed = getattr(frame, "levels_completed", self.levels_completed + 1)
                # Continue to next level
                obs = env.step(GameAction.RESET)
                if obs:
                    self.frames.append(obs)
                    self.action_count += 1
                    self.phase = "EXPLORE"
                    self.exploration_done = False
                    self.explored_actions = set()
                    continue
                break

            if frame.state == GameState.GAME_OVER:
                logger.info(f"Game over at action {self.action_count}, resetting...")
                obs = env.step(GameAction.RESET)
                if obs:
                    self.frames.append(obs)
                    self.action_count += 1
                    continue

            action = self.choose_action(self.frames, frame)
            if action is None:
                break

            obs = env.step(action)
            if obs:
                self.frames.append(obs)
                self.action_count += 1

                # Update exploration knowledge
                self._record_action_effect(action, frame, obs)

                # Log progress
                available = getattr(obs, "available_actions", [])
                state_str = obs.state.name if hasattr(obs.state, "name") else str(obs.state)
                logger.info(
                    f"Action {self.action_count}/{self.MAX_ACTIONS}: {action.name} | "
                    f"State: {state_str} | Levels: {getattr(obs, 'levels_completed', '?')}"
                )

        # Get final scorecard
        scorecard = arc.get_scorecard()
        result = {
            "game_id": self.game_id,
            "actions_taken": self.action_count,
            "levels_completed": self.levels_completed,
            "scorecard": scorecard.model_dump() if scorecard else None,
        }
        logger.info(f"Final result: {json.dumps(result, indent=2, default=str)}")
        return result

    def _record_action_effect(self, action: GameAction, before: FrameData, after: FrameData):
        """Record what changed when an action was taken."""
        effect = {
            "action": action.name,
            "before_levels": getattr(before, "levels_completed", 0),
            "after_levels": getattr(after, "levels_completed", 0),
            "state_change": f"{before.state} -> {after.state}",
        }

        # Detect grid changes
        if hasattr(before, "frame") and hasattr(after, "frame") and before.frame and after.frame:
            changed_cells = self._diff_grids(before.frame, after.frame)
            effect["cells_changed"] = len(changed_cells)
            effect["changed_positions"] = changed_cells[:10]  # Store first 10

        if action.name not in self.action_effects:
            self.action_effects[action.name] = []
        self.action_effects[action.name].append(effect)
        self.explored_actions.add(action.name)

    def _diff_grids(self, before, after) -> list[tuple]:
        """Find cells that changed between two grid states."""
        import numpy as np
        changed = []
        try:
            before_arr = np.array(before)
            after_arr = np.array(after)
            if before_arr.shape != after_arr.shape:
                return changed
            diff = before_arr != after_arr
            ys, xs = np.where(diff)
            for y, x in zip(ys, xs):
                changed.append((int(x), int(y), int(before_arr[y, x]), int(after_arr[y, x])))
        except Exception:
            pass
        return changed

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> Optional[GameAction]:
        """Main decision-making logic."""
        # Phase 1: EXPLORE — systematically probe available actions
        if self.phase == "EXPLORE":
            return self._explore(latest_frame)

        # Phase 2: UNDERSTAND — analyze what we learned (placeholder for LLM)
        if self.phase == "UNDERSTAND":
            self._understand()
            self.phase = "EXECUTE"
            return self._execute(latest_frame)

        # Phase 3: EXECUTE — follow discovered strategy
        if self.phase == "EXECUTE":
            return self._execute(latest_frame)

        return GameAction.RESET

    def _explore(self, frame: FrameData) -> GameAction:
        """Systematically explore available actions."""
        # Get available actions from the frame
        available = []
        if hasattr(frame, "available_actions") and frame.available_actions:
            available = frame.available_actions

        # Try each simple action once
        unexplored_simple = [
            a for a in [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3,
                        GameAction.ACTION4, GameAction.ACTION5]
            if a.name not in self.explored_actions
        ]

        if unexplored_simple:
            action = unexplored_simple[0]
            action.reasoning = f"Exploring untried action {action.name}"
            return action

        # Try ACTION6 with a few strategic positions (center, corners)
        if GameAction.ACTION6.name not in self.explored_actions:
            positions = [(32, 32), (0, 0), (63, 63), (0, 63), (63, 0)]
            for x, y in positions:
                pos_key = f"ACTION6@{x},{y}"
                if pos_key not in self.explored_actions:
                    self.explored_actions.add(pos_key)
                    action = GameAction.ACTION6
                    action.set_data({"x": x, "y": y})
                    action.reasoning = f"Exploring ACTION6 at ({x},{y})"
                    return action

        # Try undo
        if GameAction.ACTION7.name not in self.explored_actions:
            action = GameAction.ACTION7
            action.reasoning = "Exploring undo action"
            return action

        # Done exploring
        self.exploration_done = True
        self.phase = "UNDERSTAND"
        logger.info("Exploration complete. Analyzing game rules...")

        # Fallback: random action while waiting
        import random
        valid = [a for a in [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3,
                             GameAction.ACTION4, GameAction.ACTION5] if a.name in self.explored_actions]
        if valid:
            action = random.choice(valid)
            action.reasoning = "Post-exploration random action"
            return action
        return GameAction.RESET

    def _understand(self):
        """Analyze exploration results to build game model."""
        logger.info("=== EXPLORATION RESULTS ===")
        for action_name, effects in self.action_effects.items():
            logger.info(f"  {action_name}: {len(effects)} observations")
            if effects:
                avg_changes = sum(e.get("cells_changed", 0) for e in effects) / len(effects)
                logger.info(f"    Avg cells changed: {avg_changes:.1f}")

        # Detect action mappings based on which actions changed cells
        for action_name, effects in self.action_effects.items():
            if effects:
                avg_changes = sum(e.get("cells_changed", 0) for e in effects) / len(effects)
                if avg_changes > 0:
                    self.game_model["action_mappings"][action_name] = "active"
                else:
                    self.game_model["action_mappings"][action_name] = "no_visible_effect"

    def _execute(self, frame: FrameData) -> GameAction:
        """Execute strategy based on understanding."""
        import random

        # Simple heuristic: if we found actions that change cells, repeat them
        active_actions = [name for name, kind in self.game_model["action_mappings"].items()
                          if kind == "active" and name != "RESET"]

        if active_actions:
            # Try repeating the most effective action
            action_name = active_actions[0]
            for ga in [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3,
                       GameAction.ACTION4, GameAction.ACTION5]:
                if ga.name == action_name:
                    ga.reasoning = f"Executing active action {action_name}"
                    return ga

        # Fallback: random exploration
        valid = [a for a in [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3,
                             GameAction.ACTION4, GameAction.ACTION5]]
        action = random.choice(valid)
        action.reasoning = "Random fallback in execute phase"
        return action


def main():
    parser = argparse.ArgumentParser(description="Hermes ARC-AGI-3 Agent")
    parser.add_argument("-g", "--game", default="ls20", help="Game ID to play")
    parser.add_argument("--api-key", default=os.getenv("ARC_API_KEY", ""), help="ARC API Key")
    parser.add_argument("--competition", action="store_true", help="Competition mode")
    parser.add_argument("--max-actions", type=int, default=80, help="Max actions per level")
    args = parser.parse_args()

    agent = HermesAgent(game_id=args.game, api_key=args.api_key, competition=args.competition)
    agent.MAX_ACTIONS = args.max_actions
    result = agent.run()
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
