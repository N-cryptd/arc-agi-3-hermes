#!/usr/bin/env python3
"""
Benchmark — run HermesAgent on all 25 public ARC-AGI-3 games.

Two modes:
  1. LLM mode: uses glm-5.1 for reasoning (slow from Termux, fast on Kaggle)
  2. Baseline mode: pure exploration + heuristic (no LLM, fast everywhere)

Reports per-game stats and overall leaderboard.
"""

import json
import logging
import os
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

import arc_agi
from arcengine import FrameData, GameAction, GameState

from grid_parser import StateTracker
from explorer import Explorer

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# Quieten other loggers
logging.getLogger("arc_agi").setLevel(logging.WARNING)


# ─── Benchmark Stats ────────────────────────────────────────────────────

@dataclass
class BenchResult:
    game_id: str
    title: str
    tags: str
    levels_completed: int = 0
    win_levels: int = 0
    total_actions: int = 0
    game_overs: int = 0
    llm_calls: int = 0
    llm_timeouts: int = 0
    exploration_actions: int = 0
    mode: str = "baseline"
    error: str = ""
    duration: float = 0.0

    @property
    def completion_pct(self) -> float:
        if self.win_levels == 0:
            return 0.0
        return (self.levels_completed / self.win_levels) * 100

    def to_dict(self):
        return {
            "game": self.title,
            "id": self.game_id,
            "tags": self.tags,
            "levels": f"{self.levels_completed}/{self.win_levels}",
            "pct": f"{self.completion_pct:.0f}%",
            "actions": self.total_actions,
            "game_overs": self.game_overs,
            "llm_calls": self.llm_calls,
            "llm_timeouts": self.llm_timeouts,
            "mode": self.mode,
            "duration": f"{self.duration:.1f}s",
        }


# ─── Baseline Agent (no LLM) ───────────────────────────────────────────

class BaselineAgent:
    """Agent that uses exploration + heuristics — no LLM needed."""

    def __init__(self, env, game_id: str, tags: list[str],
                 max_actions: int = 100, exploration_budget: int = 12):
        self.env = env
        self.game_id = game_id
        self.tags = tags
        self.max_actions = max_actions
        self.tracker = StateTracker()
        self.explorer = Explorer(env, game_id, tags, budget=exploration_budget)

        # Learned from exploration
        self.move_actions: list[GameAction] = []
        self.click_action: Optional[GameAction] = None
        self.other_actions: list[GameAction] = []

        # Level tracking
        self.levels_completed = 0
        self.win_levels = 0
        self.action_count = 0
        self.game_overs = 0
        self.consecutive_no_change = 0
        self.last_diff_summary = "none"

    def run(self) -> BenchResult:
        start = time.time()

        try:
            # Phase 1: Explore
            profile = self.explorer.explore()
            exploration_used = profile.total_probes

            # Classify actions
            for name, ap in profile.action_profiles.items():
                try:
                    ga = GameAction[name]
                except KeyError:
                    continue
                if ap.inferred_label and "move" in ap.inferred_label:
                    self.move_actions.append(ga)
                elif ap.needs_coordinates:
                    self.click_action = ga
                else:
                    self.other_actions.append(ga)

            # Reset after exploration
            obs = self._reset()
            if obs is None:
                return self._result(start, exploration_used)

            # Phase 2: Execute with heuristic strategy
            while self.action_count < self.max_actions:
                snap = self.tracker.last_snapshot
                if snap is None:
                    break

                if snap.state == GameState.WIN:
                    self.levels_completed = snap.levels_completed
                    self.win_levels = snap.win_levels
                    self.consecutive_no_change = 0
                    obs = self._reset()
                    if obs is None:
                        break
                    continue

                if snap.state == GameState.GAME_OVER:
                    self.game_overs += 1
                    self.consecutive_no_change = 0
                    obs = self._reset()
                    if obs is None:
                        break
                    continue

                action = self._choose_action(snap)
                if action is None:
                    break

                obs = self._execute(action)
                if obs is None:
                    self.game_overs += 1
                    obs = self._reset()
                    if obs is None:
                        break

        except Exception as e:
            return self._result(start, 0, error=str(e)[:200])

        return self._result(start, exploration_used)

    def _reset(self) -> Optional[FrameData]:
        obs = self.env.step(GameAction.RESET)
        if obs is None:
            return None
        self.tracker.update(obs)
        self.action_count += 1
        return obs

    def _execute(self, action_tuple) -> Optional[FrameData]:
        action, data = action_tuple
        try:
            obs = self.env.step(action, data=data)
            if obs is None:
                return None
            self.tracker.update(obs)
            self.action_count += 1
            diff = self.tracker.last_diff
            self.last_diff_summary = diff.summary() if diff else "none"

            # Track no-change streak
            if diff and diff.is_trivial():
                self.consecutive_no_change += 1
            else:
                self.consecutive_no_change = 0

            return obs
        except Exception:
            return None

    def _choose_action(self, snap) -> Optional[tuple]:
        """Heuristic action selection."""
        available = [GameAction[f"ACTION{n}"] for n in snap.available_actions
                     if n != GameAction.RESET.value]

        if not available:
            return None

        # Strategy 1: If stuck (no change 5+ times), reset and try different action
        if self.consecutive_no_change >= 5:
            self._reset()
            self.consecutive_no_change = 0
            # Try a random action
            return (random.choice(available), {})

        # Strategy 2: Click-based game — probe different locations
        if self.click_action and self.click_action in available:
            # Scan the grid systematically for clickable objects
            if snap.objects:
                # Find the most interesting object to click
                target = self._find_click_target(snap)
                if target:
                    return (self.click_action, {'x': target[0], 'y': target[1]})
            # Random click
            x = random.randint(2, 61)
            y = random.randint(2, 61)
            return (self.click_action, {'x': x, 'y': y})

        # Strategy 3: Movement game — try movement actions
        if self.move_actions:
            active_moves = [a for a in self.move_actions if a in available]
            if active_moves:
                # Pick movement based on game state heuristics
                return (self._smart_move(snap, active_moves), {})

        # Strategy 4: Try other actions
        if self.other_actions:
            active_other = [a for a in self.other_actions if a in available]
            if active_other:
                return (active_other[0], {})

        # Strategy 5: Random from available
        return (random.choice(available), {})

    def _smart_move(self, snap, moves: list) -> GameAction:
        """Choose movement based on grid analysis."""
        if not snap.objects:
            return random.choice(moves)

        # Find player position (assume smallest non-background object)
        player = min(snap.objects, key=lambda o: o.pixel_count)

        # Check if any object is near a different color — move toward it
        for obj in snap.objects:
            if obj.color != player.color and obj.pixel_count > 5:
                dy = obj.bbox.center[0] - player.bbox.center[0]
                dx = obj.bbox.center[1] - player.bbox.center[1]
                if abs(dy) > abs(dx):
                    return GameAction.ACTION2 if dy > 0 else GameAction.ACTION1  # down/up
                else:
                    return GameAction.ACTION3 if dx > 0 else GameAction.ACTION4  # right/left

        # Default: cycle through movements
        idx = self.action_count % len(moves)
        return moves[idx]

    def _find_click_target(self, snap):
        """Find the most likely target to click."""
        # Prefer small, unique-colored objects
        non_bg = [o for o in snap.objects if o.color != 0 and o.color != 5]
        if non_bg:
            # Click the smallest non-background object
            target = min(non_bg, key=lambda o: o.pixel_count)
            return (target.bbox.center[1], target.bbox.center[0])
        return None

    def _result(self, start: float, exploration_used: int, error: str = "") -> BenchResult:
        return BenchResult(
            game_id=self.game_id,
            title=self.game_id.split("-")[0].upper(),
            tags=",".join(self.tags),
            levels_completed=self.levels_completed,
            win_levels=self.win_levels,
            total_actions=self.action_count,
            game_overs=self.game_overs,
            exploration_actions=exploration_used,
            mode="baseline",
            error=error,
            duration=time.time() - start,
        )


# ─── LLM Agent (with graceful fallback) ────────────────────────────────

class LLMAgent:
    """Agent that uses LLM reasoning with fallback to baseline."""

    def __init__(self, env, game_id: str, tags: list[str],
                 max_actions: int = 100, exploration_budget: int = 10,
                 llm_timeout: int = 45):
        self.env = env
        self.game_id = game_id
        self.tags = tags
        self.max_actions = max_actions
        self.tracker = StateTracker()
        self.explorer = Explorer(env, game_id, tags, budget=exploration_budget)
        self.llm_timeout = llm_timeout

        # Lazy-import LLM
        self.reasoner = None
        try:
            from llm_reasoner import LLMReasoner
            nim_key = os.environ.get("NVIDIA_NIM_API_KEY", "")
            if nim_key:
                self.reasoner = LLMReasoner(
                    api_key=nim_key,
                    model="z-ai/glm-5.1",
                    timeout=llm_timeout,
                )
        except Exception as e:
            logging.warning(f"LLM not available: {e}")

        # Tracking
        self.levels_completed = 0
        self.win_levels = 0
        self.action_count = 0
        self.game_overs = 0
        self.llm_calls = 0
        self.llm_timeouts = 0
        self.consecutive_no_change = 0

        # Baseline fallback
        self.baseline = BaselineAgent(env, game_id, tags,
                                       max_actions=max_actions,
                                       exploration_budget=0)

    def run(self) -> BenchResult:
        start = time.time()
        try:
            # Phase 1: Explore
            profile = self.explorer.explore()
            exploration_used = profile.total_probes

            obs = self._reset()
            if obs is None:
                return self._result(start, exploration_used, error="reset failed")

            # Phase 2: Try LLM reasoning
            if self.reasoner:
                plan = self.reasoner.reason_initial(self.explorer, self.tracker)
                self.llm_calls += 1
                if "TIMEOUT" in plan.raw_response or "ERROR" in plan.raw_response:
                    self.llm_timeouts += 1
                    logging.warning(f"LLM initial call failed for {self.game_id}")
                elif plan.next_action:
                    snap = self.tracker.last_snapshot
                    if snap:
                        parsed = self.reasoner.parse_next_action(plan, snap.available_actions)
                        if parsed:
                            obs = self._execute(*parsed)
                            if obs and obs.state == GameState.NOT_FINISHED:
                                # Ask LLM for a few more actions
                                for _ in range(3):
                                    if self.action_count >= self.max_actions:
                                        break
                                    snap = self.tracker.last_snapshot
                                    if snap is None or snap.state != GameState.NOT_FINISHED:
                                        break
                                    diff = self.tracker.last_diff
                                    plan2 = self.reasoner.reason_step(
                                        self.tracker, "LLM guided",
                                        diff.summary() if diff else "none"
                                    )
                                    self.llm_calls += 1
                                    if "TIMEOUT" in plan2.raw_response:
                                        self.llm_timeouts += 1
                                        break
                                    if plan2.next_action:
                                        p2 = self.reasoner.parse_next_action(plan2, snap.available_actions)
                                        if p2:
                                            obs = self._execute(*p2)
                                        else:
                                            break
                                    else:
                                        break

            # Phase 3: Fall back to baseline for remaining budget
            remaining = self.max_actions - self.action_count
            if remaining > 10:
                self.baseline.max_actions = remaining
                self.baseline.action_count = self.action_count
                # Share the already-explored state
                self.baseline.move_actions = self.explorer.profile.action_profiles
                self.baseline.explorer.explored = True

                while self.action_count < self.max_actions:
                    snap = self.tracker.last_snapshot
                    if snap is None:
                        break
                    if snap.state == GameState.WIN:
                        self.levels_completed = snap.levels_completed
                        self.win_levels = snap.win_levels
                        obs = self._reset()
                        if obs is None:
                            break
                        continue
                    if snap.state == GameState.GAME_OVER:
                        self.game_overs += 1
                        obs = self._reset()
                        if obs is None:
                            break
                        continue

                    action = self._baseline_choose(snap)
                    if action is None:
                        break
                    obs = self._execute(*action)
                    if obs is None:
                        self.game_overs += 1
                        obs = self._reset()
                        if obs is None:
                            break

        except Exception as e:
            return self._result(start, 0, error=str(e)[:200])

        return self._result(start, exploration_used or 0)

    def _reset(self) -> Optional[FrameData]:
        obs = self.env.step(GameAction.RESET)
        if obs is None:
            return None
        self.tracker.update(obs)
        self.action_count += 1
        return obs

    def _execute(self, action, data) -> Optional[FrameData]:
        try:
            obs = self.env.step(action, data=data)
            if obs is None:
                return None
            self.tracker.update(obs)
            self.action_count += 1
            diff = self.tracker.last_diff
            if diff and diff.is_trivial():
                self.consecutive_no_change += 1
            else:
                self.consecutive_no_change = 0
            return obs
        except Exception:
            return None

    def _baseline_choose(self, snap):
        """Simplified baseline fallback."""
        available = [GameAction[f"ACTION{n}"] for n in snap.available_actions
                     if n != GameAction.RESET.value]
        if not available:
            return None

        if self.consecutive_no_change >= 5:
            self._reset()
            self.consecutive_no_change = 0
            return (random.choice(available), {})

        # Check for click actions that need coordinates
        for ga in available:
            name = ga.name
            ap = self.explorer.profile.action_profiles.get(name)
            if ap and ap.needs_coordinates:
                if snap.objects:
                    target = min([o for o in snap.objects if o.color not in (0, 5)],
                                key=lambda o: o.pixel_count, default=None)
                    if target:
                        return (ga, {'x': target.bbox.center[1], 'y': target.bbox.center[0]})
                return (ga, {'x': random.randint(2, 61), 'y': random.randint(2, 61)})

        return (random.choice(available), {})

    def _result(self, start: float, exploration_used: int, error: str = "") -> BenchResult:
        return BenchResult(
            game_id=self.game_id,
            title=self.game_id.split("-")[0].upper(),
            tags=",".join(self.tags),
            levels_completed=self.levels_completed,
            win_levels=self.win_levels,
            total_actions=self.action_count,
            game_overs=self.game_overs,
            llm_calls=self.llm_calls,
            llm_timeouts=self.llm_timeouts,
            exploration_actions=exploration_used,
            mode="llm+baseline" if self.reasoner else "baseline",
            error=error,
            duration=time.time() - start,
        )


# ─── Main Benchmark Runner ─────────────────────────────────────────────

def run_benchmark(mode: str = "baseline", max_actions: int = 80,
                  exploration_budget: int = 10):
    API_KEY = os.environ.get("ARC_API_KEY", "")
    arc = arc_agi.Arcade(arc_api_key=API_KEY)
    envs = arc.get_environments()

    print(f"{'='*70}")
    print(f"  ARC-AGI-3 BENCHMARK — {mode.upper()} MODE")
    print(f"  Games: {len(envs)} | Max actions: {max_actions} | "
          f"Exploration budget: {exploration_budget}")
    print(f"{'='*70}\n")

    results: list[BenchResult] = []

    for i, env_info in enumerate(envs):
        gid = env_info.game_id
        title = env_info.title
        tags = list(env_info.tags) if env_info.tags else []

        print(f"[{i+1:2d}/25] {title:6s} ({gid}) ... ", end="", flush=True)
        sys.stdout.flush()

        env = arc.make(gid)

        if mode == "llm":
            agent = LLMAgent(env, gid, tags, max_actions=max_actions,
                             exploration_budget=exploration_budget,
                             llm_timeout=45)
        else:
            agent = BaselineAgent(env, gid, tags, max_actions=max_actions,
                                   exploration_budget=exploration_budget)

        result = agent.run()
        results.append(result)

        # Print result line
        status = f"✅ {result.levels_completed}/{result.win_levels} levels"
        if result.error:
            status = f"❌ {result.error[:40]}"
        if result.completion_pct == 100:
            status = f"🏆 {result.levels_completed}/{result.win_levels} PERFECT!"
        elif result.completion_pct > 0:
            status = f"🎯 {result.levels_completed}/{result.win_levels} ({result.completion_pct:.0f}%)"

        print(f"{status} | {result.total_actions} actions | "
              f"{result.game_overs} GOs | {result.duration:.1f}s"
              + (f" | LLM: {result.llm_calls}c/{result.llm_timeouts}t" if mode == "llm" else ""))

    # Save results
    out_path = os.path.join(os.path.dirname(__file__), "benchmark_results.json")
    with open(out_path, "w") as f:
        json.dump([r.to_dict() for r in results], f, indent=2)

    # Print summary table
    print(f"\n{'='*70}")
    print(f"  LEADERBOARD")
    print(f"{'='*70}")
    print(f"  {'Game':<8s} {'Tags':<18s} {'Levels':<12s} {'Pct':<6s} "
          f"{'Actions':<9s} {'GOs':<5s} {'Time':<7s} {'Mode':<14s}")
    print(f"  {'─'*75}")

    total_levels = 0
    total_possible = 0
    for r in sorted(results, key=lambda x: -x.completion_pct):
        total_levels += r.levels_completed
        total_possible += r.win_levels
        print(f"  {r.title:<8s} {r.tags:<18s} {r.levels_completed}/{r.win_levels:<10d} "
              f"{r.completion_pct:>4.0f}%  {r.total_actions:<9d} {r.game_overs:<5d} "
              f"{r.duration:<7.1f} {r.mode:<14s}")

    print(f"  {'─'*75}")
    print(f"  {'TOTAL':<8s} {'':18s} {total_levels}/{total_possible:<10d} "
          f"{(total_levels/total_possible*100) if total_possible else 0:>4.0f}%")
    print(f"\n  Results saved to {out_path}")

    return results


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    max_actions = int(sys.argv[2]) if len(sys.argv) > 2 else 80
    run_benchmark(mode=mode, max_actions=max_actions)
