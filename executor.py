#!/usr/bin/env python3
"""
Adaptive Executor for ARC-AGI-3.

Orchestrates the full agent loop:
  EXPLORE → UNDERSTAND → PLAN → EXECUTE → ADAPT → REPEAT

Handles:
- Multi-level play (win detection, auto-advance)
- Action budget management (don't waste actions on dead ends)
- Fallback strategies when LLM fails
- Efficiency optimization (reuse successful patterns)
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from arcengine import FrameData, GameAction, GameState

from explorer import Explorer, GameProfile
from grid_parser import StateTracker
from llm_reasoner import LLMReasoner, ReasoningPlan

logger = logging.getLogger(__name__)


class AgentPhase(Enum):
    EXPLORE = "EXPLORE"
    UNDERSTAND = "UNDERSTAND"
    PLAN = "PLAN"
    EXECUTE = "EXECUTE"
    ADAPT = "ADAPT"
    DONE = "DONE"


@dataclass
class RunStats:
    """Statistics from an agent run."""
    game_id: str = ""
    levels_completed: int = 0
    total_actions: int = 0
    total_llm_calls: int = 0
    total_llm_tokens: int = 0
    exploration_actions: int = 0
    plan_actions: int = 0
    failed_actions: int = 0
    game_overs: int = 0
    start_time: float = 0.0
    end_time: float = 0.0

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time if self.end_time else 0.0

    def summary(self) -> str:
        return (
            f"Game: {self.game_id} | "
            f"Levels: {self.levels_completed} | "
            f"Actions: {self.total_actions} | "
            f"LLM calls: {self.total_llm_calls} ({self.total_llm_tokens} tokens) | "
            f"Time: {self.duration:.1f}s"
        )


class AdaptiveExecutor:
    """Orchestrates the explore-understand-plan-execute loop."""

    def __init__(
        self,
        env,
        game_id: str,
        tags: list[str],
        reasoner: LLMReasoner,
        max_actions: int = 80,
        exploration_budget: int = 15,
    ):
        self.env = env
        self.game_id = game_id
        self.tags = tags
        self.reasoner = reasoner
        self.max_actions = max_actions

        self.tracker = StateTracker()
        self.explorer = Explorer(env, game_id, tags, budget=exploration_budget)

        self.phase = AgentPhase.EXPLORE
        self.stats = RunStats(game_id=game_id, start_time=time.time())

        # Strategy state
        self.consecutive_failures = 0
        self.last_successful_actions: list[tuple] = []  # for pattern reuse
        self.level_solutions: list[list[tuple]] = []  # store winning sequences
        self.current_plan: list[tuple] = []  # pending actions to execute

    def run(self) -> RunStats:
        """Main agent loop."""
        logger.info(f"Starting agent for {self.game_id} (max={self.max_actions} actions)")

        # Phase 1: Explore
        self.phase = AgentPhase.EXPLORE
        profile = self.explorer.explore()
        self.stats.exploration_actions = profile.total_probes

        # Reset after exploration
        obs = self._reset()
        if obs is None:
            logger.error("Failed to reset after exploration")
            return self._finish()

        # Phase 2: Initial understanding (LLM reasoning)
        self.phase = AgentPhase.UNDERSTAND
        plan = self.reasoner.reason_initial(self.explorer, self.tracker)
        self.stats.total_llm_calls += 1
        self.stats.total_llm_tokens += self.reasoner.total_tokens_used

        if plan.next_action:
            parsed = self.reasoner.parse_next_action(plan, obs.available_actions)
            if parsed:
                self.current_plan = [parsed]

        # Phase 3: Execute loop
        self.phase = AgentPhase.EXECUTE
        while self.stats.total_actions < self.max_actions:
            snap = self.tracker.last_snapshot
            if snap is None:
                break

            # Check win/lose states
            if snap.state == GameState.WIN:
                self._handle_win()
                continue

            if snap.state == GameState.GAME_OVER:
                self._handle_game_over()
                continue

            # Get next action
            action_tuple = self._get_next_action()
            if action_tuple is None:
                logger.info("No action to execute, stopping")
                break

            # Execute
            success = self._execute_action(*action_tuple)
            if not success:
                self.consecutive_failures += 1
                if self.consecutive_failures >= 5:
                    logger.warning("Too many consecutive failures, trying random reset")
                    self._reset()
                    self.consecutive_failures = 0
            else:
                self.consecutive_failures = 0

            # Phase 4: Adapt (request new plan from LLM periodically)
            if (self.stats.total_actions % 5 == 0
                    and self.stats.total_actions > 0):
                self.phase = AgentPhase.ADAPT
                self._request_adaptation()

        self.phase = AgentPhase.DONE
        return self._finish()

    def _reset(self) -> Optional[FrameData]:
        """Reset the game and update tracker."""
        obs = self.env.step(GameAction.RESET)
        if obs is None:
            return None
        self.tracker.update(obs)
        self.stats.total_actions += 1
        return obs

    def _get_next_action(self) -> Optional[tuple]:
        """Get the next action to execute from plan or request new one."""
        # Try current plan first
        while self.current_plan:
            action_tuple = self.current_plan.pop(0)
            return action_tuple

        # Try reusing last successful level solution
        if self.level_solutions and self.consecutive_failures < 3:
            for sol in reversed(self.level_solutions):
                if sol:
                    self.current_plan = list(sol)
                    return self.current_plan.pop(0)

        # Request new plan from LLM
        self._request_adaptation()
        if self.current_plan:
            return self.current_plan.pop(0)

        # Fallback: try available actions in order
        snap = self.tracker.last_snapshot
        if snap:
            for action_num in snap.available_actions:
                # Match action by value (available_actions returns int values)
                for ga in GameAction:
                    if ga.value == action_num and ga != GameAction.RESET:
                        return (ga, {})

        return None

    def _execute_action(self, action: GameAction, data: dict) -> bool:
        """Execute a single action and update state."""
        try:
            obs = self.env.step(action, data=data)
            if obs is None:
                self.stats.failed_actions += 1
                logger.warning(f"Action {action} returned None")
                return False

            self.tracker.update(obs)
            self.stats.total_actions += 1
            self.stats.plan_actions += 1

            diff = self.tracker.last_diff
            logger.info(
                f"  [{self.stats.total_actions}/{self.max_actions}] "
                f"{action} data={data} -> {obs.state.name} | {diff.summary()}"
            )

            # Track potentially successful actions
            if not diff.is_trivial() or obs.state == GameState.WIN:
                self.last_successful_actions.append((action, data))
                if len(self.last_successful_actions) > 20:
                    self.last_successful_actions.pop(0)

            return True

        except Exception as e:
            self.stats.failed_actions += 1
            logger.error(f"Action {action} failed: {e}")
            return False

    def _handle_win(self):
        """Handle a WIN state."""
        snap = self.tracker.last_snapshot
        logger.info(f"🎉 Level WON! (levels completed: {snap.levels_completed})")
        self.stats.levels_completed = snap.levels_completed
        self.consecutive_failures = 0

        # Store winning solution
        if self.last_successful_actions:
            self.level_solutions.append(list(self.last_successful_actions))
            self.last_successful_actions = []
            self.current_plan = []

        # Reset for next level
        self._reset()

    def _handle_game_over(self):
        """Handle GAME_OVER state."""
        snap = self.tracker.last_snapshot
        logger.info(f"💀 Game Over (levels: {snap.levels_completed})")
        self.stats.game_overs += 1
        self.consecutive_failures += 1

        # Reset and try again
        self._reset()

    def _request_adaptation(self):
        """Request a new plan from the LLM based on current state."""
        snap = self.tracker.last_snapshot
        if snap is None:
            return

        last_action = "none"
        diff_summary = "none"
        if self.stats.total_actions > 1:
            diff = self.tracker.last_diff
            if diff:
                diff_summary = diff.summary()

        plan = self.reasoner.reason_step(
            self.tracker, last_action, diff_summary
        )
        self.stats.total_llm_calls += 1
        self.stats.total_llm_tokens += self.reasoner.total_tokens_used

        if plan.next_action:
            parsed = self.reasoner.parse_next_action(plan, snap.available_actions)
            if parsed:
                # Add to current plan (LLM might suggest multiple in reasoning)
                self.current_plan = [parsed]

    def _finish(self) -> RunStats:
        """Finish the run and return stats."""
        self.stats.end_time = time.time()
        logger.info(self.stats.summary())
        return self.stats
