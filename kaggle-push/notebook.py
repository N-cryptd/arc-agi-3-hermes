"""
HermesAgent — ARC-AGI-3 Competition Entry (v2 — LLM-First)
LLM-powered grid puzzle solver using NVIDIA NIM (glm-4.7)

v2 Architecture: LLM-First with Smart Exploration
  1. EXPLORE (12 actions): probe each action, record diffs
  2. UNDERSTAND (1 LLM call): analyze game + plan
  3. EXECUTE: follow plan, re-evaluate every 8 steps
  4. No early bailout — full budget used
  5. MAX_ACTIONS = 120
  6. LLM fires proactively, not just as rescue
"""

import os
import sys
import json
import time
import random
import logging
import re
import threading
from typing import Optional
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("hermes")

# ─── Install Dependencies ──────────────────────────────────────────────
print("Installing dependencies...")
os.system("pip install arc-agi requests numpy -q 2>/dev/null")

# ─── Setup Paths ───────────────────────────────────────────────────────
COMPETITION_DATA = "/kaggle/input/competitions/arc-prize-2026-arc-agi-3"
AGENTS_DIR = os.path.join(COMPETITION_DATA, "ARC-AGI-3-Agents")

if os.path.exists(AGENTS_DIR):
    sys.path.insert(0, AGENTS_DIR)
    sys.path.insert(0, os.path.join(AGENTS_DIR, "agents"))
    print(f"✅ Found competition data at {AGENTS_DIR}")
else:
    print(f"⚠️ Competition data not found at {COMPETITION_DATA}")
    print("Available:", os.listdir("/kaggle/input/") if os.path.exists("/kaggle/input") else "none")

# ─── Load API Keys ─────────────────────────────────────────────────────
from kaggle_secrets import UserSecretsClient
secrets = UserSecretsClient()

ARC_API_KEY = secrets.get_secret("ARC_API_KEY")
NIM_API_KEY = secrets.get_secret("NVIDIA_NIM_API_KEY")

os.environ["ARC_API_KEY"] = ARC_API_KEY
os.environ["NVIDIA_NIM_API_KEY"] = NIM_API_KEY

print(f"✅ API keys loaded (ARC: {ARC_API_KEY[:6]}..., NIM: {NIM_API_KEY[:8]}...)")

# ─── Grid Parser & State Tracker ──────────────────────────────────────
import numpy as np

try:
    from arcengine import FrameData, GameAction, GameState
except ImportError:
    print("ERROR: arcengine not available")
    sys.exit(1)


@dataclass
class BoundingBox:
    y0: int; x0: int; y1: int; x1: int
    @property
    def center(self): return ((self.y0 + self.y1) // 2, (self.x0 + self.x1) // 2)
    @property
    def area(self): return (self.y1 - self.y0 + 1) * (self.x1 - self.x0 + 1)


@dataclass
class GridObject:
    color: int; pixel_count: int; bbox: BoundingBox


class StateTracker:
    """State tracker with grid analysis and object detection."""

    def __init__(self):
        self.history: list[dict] = []
        self._prev_px = 0

    def update(self, obs: FrameData) -> dict:
        frame = obs.frame
        layers = len(frame)
        h, w = len(frame[0]), len(frame[0][0])
        grid = np.zeros((h, w), dtype=np.int8)
        for layer in range(layers):
            arr = np.array(frame[layer], dtype=np.int8)
            mask = arr != 0
            grid[mask] = arr[mask]
        objects = self._find_objects(grid)
        total_pixels = int(np.count_nonzero(grid))
        snap = {
            "objects": len(objects),
            "object_list": objects,
            "total_pixels": total_pixels,
            "state": obs.state,
            "levels_completed": obs.levels_completed,
            "win_levels": obs.win_levels,
            "available_actions": obs.available_actions,
            "layers": layers,
            "grid_size": (h, w),
            "grid": grid,
        }
        self.history.append(snap)
        return snap

    def _find_objects(self, grid: np.ndarray) -> list:
        visited = np.zeros_like(grid, dtype=bool)
        objects = []
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                if grid[r][c] != 0 and not visited[r][c]:
                    color = grid[r][c]
                    queue = deque([(r, c)])
                    visited[r][c] = True
                    pixels = 0
                    min_r, min_c, max_r, max_c = r, c, r, c
                    while queue:
                        cr, cc = queue.popleft()
                        pixels += 1
                        min_r = min(min_r, cr); max_r = max(max_r, cr)
                        min_c = min(min_c, cc); max_c = max(max_c, cc)
                        for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
                            nr, nc = cr+dr, cc+dc
                            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                                if not visited[nr][nc] and grid[nr][nc] == color:
                                    visited[nr][nc] = True
                                    queue.append((nr, nc))
                    objects.append(GridObject(
                        color=int(color), pixel_count=pixels,
                        bbox=BoundingBox(min_r, min_c, max_r, max_c),
                    ))
        return sorted(objects, key=lambda o: o.pixel_count)

    def frame_to_ascii(self, obs: FrameData, max_w: int = 48) -> str:
        """Render grid as ASCII art."""
        frame = obs.frame
        h, w = len(frame[0]), len(frame[0][0])
        grid = np.zeros((h, w), dtype=np.int8)
        for layer in frame:
            arr = np.array(layer, dtype=np.int8)
            mask = arr != 0
            grid[mask] = arr[mask]
        chars = " .:=+*#%@&"
        scale = max(1, w // max_w)
        lines = []
        for r in range(0, h, scale):
            row = ""
            for c in range(0, w, scale):
                v = grid[r][c]
                row += chars[min(v, len(chars)-1) % len(chars)]
            lines.append(row)
        return "\n".join(lines)

    def objects_summary(self, snap: dict) -> str:
        """Compact object summary for LLM prompt."""
        objs = snap.get("object_list", [])
        lines = []
        for i, o in enumerate(objs):
            if o.color == 0:
                continue
            cy, cx = o.bbox.center
            lines.append(f"  Obj{i}: color={o.color}, size={o.pixel_count}px, "
                        f"bbox=[{o.bbox.y0},{o.bbox.x0}]-[{o.bbox.y1},{o.bbox.x1}], "
                        f"center=({cx},{cy})")
        return "\n".join(lines) if lines else "  (no non-bg objects)"


# ─── LLM Reasoner ──────────────────────────────────────────────────────
import requests

_http_session = requests.Session()
_http_session.headers.update({"Content-Type": "application/json"})
_session_lock = threading.Lock()


class LLMReasoner:
    def __init__(self, api_key: str, model: str = "z-ai/glm-4.7",
                 base_url: str = "https://integrate.api.nvidia.com/v1", timeout: int = 30):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self.total_tokens = 0
        self.calls = 0

    def reason_initial(self, obs: FrameData, snap: dict, tracker: StateTracker,
                       exploration_log: list[dict]) -> Optional[dict]:
        """Initial reasoning after exploration phase — understand the game."""
        ascii_grid = tracker.frame_to_ascii(obs, max_w=48)
        objects = tracker.objects_summary(snap)

        available = snap.get("available_actions", [])
        actions_str = ", ".join([f"ACTION{a}" for a in available if a != 0])

        # Build exploration summary
        expl_lines = []
        for entry in exploration_log:
            expl_lines.append(
                f"  {entry['action']}: Δpixels={entry.get('delta_pixels', '?')}, "
                f"changed_cells={entry.get('changed_cells', '?')}, "
                f"state={entry.get('state_after', '?')}"
            )
        exploration_str = "\n".join(expl_lines) if expl_lines else "  (none)"

        prompt = (
            f"You are playing an ARC grid puzzle game on a 64x64 grid (colors 0-15).\n"
            f"Goal: figure out the transformation rule and complete all levels.\n\n"
            f"ACTIONS AVAILABLE: {actions_str}\n"
            f"Actions with coordinates need (x,y) to specify WHERE to click/interact.\n\n"
            f"CURRENT GRID STATE:\n{ascii_grid}\n\n"
            f"OBJECTS ON GRID:\n{objects}\n\n"
            f"EXPLORATION RESULTS (what each action did):\n{exploration_str}\n\n"
            f"GAME STATE: {snap['state'].name}, "
            f"Levels: {snap['levels_completed']}/{snap['win_levels']}\n\n"
            f"Analyze the grid and exploration results. What is the game's rule?\n"
            f"What sequence of actions would solve this level?\n"
            f"For click/coordinate actions, specify TARGET x,y.\n\n"
            f"Respond in this EXACT format:\n"
            f"ANALYSIS: <your reasoning about the game rule>\n"
            f"STRATEGY: <your plan for the next few moves>\n"
            f"NEXT_ACTION: ACTION<n>\n"
            f"TARGET: <x,y> or NONE if no coordinates needed\n"
            f"CONFIDENCE: 0.0-1.0"
        )

        return self._call_llm(prompt)

    def reason_step(self, obs: FrameData, snap: dict, tracker: StateTracker,
                    action_history: list[dict], step_num: int) -> Optional[dict]:
        """Re-evaluate strategy mid-execution."""
        ascii_grid = tracker.frame_to_ascii(obs, max_w=48)
        objects = tracker.objects_summary(snap)

        available = snap.get("available_actions", [])
        actions_str = ", ".join([f"ACTION{a}" for a in available if a != 0])

        # Recent action history
        recent = action_history[-6:] if len(action_history) > 6 else action_history
        hist_lines = [f"  {h['action']}: state={h.get('state','?')}" for h in recent]
        history_str = "\n".join(hist_lines)

        prompt = (
            f"ARC grid puzzle — re-evaluation at step {step_num}.\n\n"
            f"CURRENT GRID:\n{ascii_grid}\n\n"
            f"OBJECTS:\n{objects}\n\n"
            f"ACTIONS: {actions_str}\n"
            f"Level: {snap['levels_completed']}/{snap['win_levels']} | "
            f"State: {snap['state'].name}\n\n"
            f"RECENT ACTIONS:\n{history_str}\n\n"
            f"Are we making progress? What should we do next?\n"
            f"If stuck, try a different approach. For coordinate actions, specify TARGET x,y.\n\n"
            f"Respond:\n"
            f"ANALYSIS: <brief assessment>\n"
            f"NEXT_ACTION: ACTION<n>\n"
            f"TARGET: <x,y> or NONE\n"
            f"CONFIDENCE: 0.0-1.0"
        )

        return self._call_llm(prompt)

    def _call_llm(self, prompt: str) -> Optional[dict]:
        messages = [
            {"role": "system", "content": (
                "You are an expert ARC grid puzzle solver. "
                "Analyze grids and figure out transformation rules. "
                "Always respond in the exact format requested. "
                "Be decisive — pick ONE action."
            )},
            {"role": "user", "content": prompt}
        ]
        try:
            with _session_lock:
                resp = _http_session.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={"model": self.model, "messages": messages,
                          "max_tokens": 256, "temperature": 0.3},
                    timeout=self.timeout
                )
            self.calls += 1
            if resp.status_code != 200:
                logger.warning(f"LLM {resp.status_code}: {resp.text[:100]}")
                return None
            data = resp.json()
            self.total_tokens += data.get("usage", {}).get("total_tokens", 0)
            text = data["choices"][0]["message"]["content"]
            return self._parse_response(text)
        except requests.exceptions.Timeout:
            logger.warning("LLM timeout")
            return None
        except Exception as e:
            logger.warning(f"LLM error: {e}")
            return None

    def _parse_response(self, text: str) -> Optional[dict]:
        result = {"action": None, "target": None, "analysis": "", "confidence": 0.5, "raw": text}
        for line in text.strip().split("\n"):
            line = line.strip()
            up = line.upper()
            if up.startswith("ANALYSIS:") or up.startswith("STRATEGY:"):
                result["analysis"] += line.split(":", 1)[1].strip() + " "
            elif up.startswith("NEXT_ACTION:"):
                val = line.split(":", 1)[1].strip().upper()
                for part in val.replace("ACTION", "").replace(" ", ""):
                    if part.isdigit():
                        result["action"] = int(part)
                        break
            elif up.startswith("TARGET:"):
                val = line.split(":", 1)[1].strip().upper()
                if val and val != "NONE":
                    nums = re.findall(r'\d+', val)
                    if len(nums) >= 2:
                        result["target"] = (int(nums[0]), int(nums[1]))
            elif up.startswith("CONFIDENCE:"):
                try:
                    result["confidence"] = float(line.split(":", 1)[1].strip())
                except ValueError:
                    pass
        return result if result["action"] else None


# ─── HermesAgent v2 (LLM-First) ───────────────────────────────────────

class HermesAgent:
    """LLM-First agent: explore → understand → execute → re-evaluate."""

    DEFAULT_MAX_ACTIONS = 120
    EXPLORE_BUDGET = 12         # probe each action at least once
    REEVALUATE_INTERVAL = 8     # LLM re-evaluate every N steps
    STUCK_THRESHOLD = 4          # LLM rescue after stuck steps

    def __init__(self, env, game_id: str, max_actions: int = DEFAULT_MAX_ACTIONS,
                 reasoner: Optional[LLMReasoner] = None):
        self.env = env
        self.game_id = game_id
        self.max_actions = max_actions
        self.tracker = StateTracker()
        self.reasoner = reasoner
        self.action_count = 0
        self.levels_completed = 0
        self.win_levels = 0
        self.llm_calls = 0

        # Exploration
        self.explore_done = False
        self.exploration_log: list[dict] = []
        self.explored_actions: set[str] = set()
        self.last_pixel_count = 0
        self.consecutive_stuck = 0

        # Execution tracking
        self.action_history: list[dict] = []
        self._last_obs: Optional[FrameData] = None
        self._initial_plan = None

    def run(self) -> dict:
        """Main agent loop."""
        title = self.game_id.split("-")[0].upper()
        print(f"\n{'='*50}")
        print(f"  HermesAgent v2: {self.game_id}")
        print(f"{'='*50}")

        obs = self._reset()
        if not obs:
            return {"error": "reset failed", "game_id": self.game_id}

        while self.action_count < self.max_actions:
            snap = self.tracker.history[-1] if self.tracker.history else {}

            # ── Win / Game Over handling ──
            if obs.state == GameState.WIN:
                self.levels_completed = obs.levels_completed
                self.win_levels = obs.win_levels
                self.consecutive_stuck = 0
                print(f"  🎉 Level WON! ({self.levels_completed}/{self.win_levels})")
                obs = self._reset()
                if not obs:
                    break
                # Re-explore briefly after winning
                self.explore_done = False
                self.explored_actions = set()
                continue

            if obs.state == GameState.GAME_OVER:
                self.consecutive_stuck = 0
                print(f"  💀 Game Over ({self.levels_completed}/{self.win_levels})")
                obs = self._reset()
                if not obs:
                    break
                continue

            # ── Stuck detection ──
            total_px = snap.get("total_pixels", 0)
            if total_px == self.last_pixel_count and total_px > 0:
                self.consecutive_stuck += 1
            else:
                self.consecutive_stuck = 0
            self.last_pixel_count = total_px

            # Hard stuck → reset (no early bail, just reset and try again)
            if self.consecutive_stuck >= 10:
                self.consecutive_stuck = 0
                print(f"  🔄 Stuck 10 steps, resetting...")
                obs = self._reset()
                if not obs:
                    break
                self.explore_done = False
                self.explored_actions = set()
                continue

            # ── Choose action ──
            action = self._choose(obs, snap)
            if not action:
                break

            # ── Execute ──
            obs = self._execute(action)
            if not obs:
                print(f"  ⚠️ Action failed, resetting...")
                obs = self._reset()
                if not obs:
                    break

        pct = round(self.levels_completed / self.win_levels * 100, 1) if self.win_levels else 0
        result = {
            "game_id": self.game_id,
            "levels_completed": self.levels_completed,
            "win_levels": self.win_levels,
            "actions": self.action_count,
            "pct": pct,
            "llm_calls": self.llm_calls,
        }
        print(f"  Result: {result['levels_completed']}/{result['win_levels']} "
              f"({result['pct']}%) in {result['actions']} actions, "
              f"{result['llm_calls']} LLM calls")
        return result

    def _reset(self) -> Optional[FrameData]:
        obs = self.env.step(GameAction.RESET)
        if obs:
            self.tracker.update(obs)
            self._last_obs = obs
            self.action_count += 1
        return obs

    def _execute(self, action: GameAction) -> Optional[FrameData]:
        data = {}
        if hasattr(action, 'action_data') and action.action_data:
            data = action.action_data.model_dump()
        try:
            obs = self.env.step(action, data=data)
            if obs:
                # Track exploration effects
                old_px = self.tracker.history[-1]["total_pixels"] if self.tracker.history else 0
                self.tracker.update(obs)
                new_px = self.tracker.history[-1]["total_pixels"] if self.tracker.history else 0

                changed_cells = self._count_changed(obs)
                self.exploration_log.append({
                    "action": action.name,
                    "delta_pixels": new_px - old_px,
                    "changed_cells": changed_cells,
                    "state_after": obs.state.name,
                })
                self.explored_actions.add(action.name)

                self._last_obs = obs
                self.action_count += 1
                self.action_history.append({
                    "step": self.action_count,
                    "action": action.name,
                    "state": obs.state.name,
                })

                if self.action_count % 20 == 0:
                    snap = self.tracker.history[-1]
                    print(f"  [{self.action_count}/{self.max_actions}] "
                          f"{action.name} -> {obs.state.name} | "
                          f"objs={snap['objects']} px={snap['total_pixels']}")
            return obs
        except Exception as e:
            print(f"  ❌ {action.name} error: {e}")
            return None

    def _count_changed(self, obs: FrameData) -> int:
        """Quick count of changed cells between last two frames."""
        if len(self.tracker.history) < 2:
            return 0
        g1 = self.tracker.history[-2].get("grid")
        g2 = self.tracker.history[-1].get("grid")
        if g1 is None or g2 is None:
            return 0
        return int(np.count_nonzero(g1 != g2))

    def _choose(self, obs: FrameData, snap: dict) -> Optional[GameAction]:
        available = obs.available_actions or []
        non_reset = [a for a in available if a != GameAction.RESET.value]
        if not non_reset:
            return GameAction.RESET

        # ── Phase 1: EXPLORE — probe each action once ──
        if not self.explore_done and self.action_count <= self.EXPLORE_BUDGET + 1:
            # Try each action at least once
            for action_num in non_reset:
                action_name = f"ACTION{action_num}"
                if action_name not in self.explored_actions:
                    action = GameAction[action_name]
                    # For complex actions, try center first
                    if action.is_complex():
                        self._set_click_data(action, snap)
                    return action

            # All actions explored
            self.explore_done = True
            print(f"  📋 Exploration done ({len(self.explored_actions)} actions probed)")

        # ── Phase 2: LLM initial understanding (once per game) ──
        if self._initial_plan is None and self.reasoner and self.explore_done:
            print(f"  🧠 LLM: initial analysis...")
            plan = self.reasoner.reason_initial(obs, snap, self.tracker, self.exploration_log)
            self.llm_calls += 1
            if plan:
                self._initial_plan = plan
                print(f"  🧠 Plan: {plan.get('analysis', '')[:80]}...")
                action = self._apply_plan(plan, non_reset, snap)
                if action:
                    return action
            else:
                print(f"  ⚠️ LLM returned no plan, using heuristic")

        # ── Phase 3: Periodic LLM re-evaluation ──
        if (self.reasoner and self.action_count > 0
                and self.action_count % self.REEVALUATE_INTERVAL == 0):
            print(f"  🧠 LLM: re-evaluation at step {self.action_count}...")
            plan = self.reasoner.reason_step(obs, snap, self.tracker,
                                             self.action_history, self.action_count)
            self.llm_calls += 1
            if plan and plan.get("confidence", 0) > 0.2:
                action = self._apply_plan(plan, non_reset, snap)
                if action:
                    self.consecutive_stuck = 0
                    return action

        # ── Phase 4: LLM stuck rescue ──
        if (self.reasoner and self.consecutive_stuck >= self.STUCK_THRESHOLD
                and self.levels_completed < self.win_levels):
            print(f"  🧠 LLM: stuck rescue (consecutive={self.consecutive_stuck})...")
            plan = self.reasoner.reason_step(obs, snap, self.tracker,
                                             self.action_history, self.action_count)
            self.llm_calls += 1
            if plan:
                action = self._apply_plan(plan, non_reset, snap)
                if action:
                    self.consecutive_stuck = 0
                    return action

        # ── Phase 5: Heuristic fallback ──
        return self._heuristic(non_reset, snap)

    def _apply_plan(self, plan: dict, non_reset: list, snap: dict) -> Optional[GameAction]:
        """Convert LLM plan into a GameAction."""
        action_num = plan.get("action")
        if not action_num or action_num not in non_reset:
            return None

        action = GameAction[f"ACTION{action_num}"]

        # Apply target coordinates if specified
        target = plan.get("target")
        if target and action.is_complex():
            action.set_data({"x": int(target[0]), "y": int(target[1])})
        elif action.is_complex():
            self._set_click_data(action, snap)

        return action

    def _set_click_data(self, action: GameAction, snap: dict):
        """Set x,y coordinates for complex (click) actions."""
        objs = snap.get("object_list", [])
        if not objs:
            grid = snap.get("grid")
            if grid is not None:
                objs = self.tracker._find_objects(grid)
                snap["object_list"] = objs
        non_bg = [o for o in objs if o.color != 0]
        if non_bg:
            idx = self.action_count % len(non_bg)
            target = non_bg[idx]
            cy, cx = target.bbox.center
            action.set_data({"x": int(cx), "y": int(cy)})
        else:
            action.set_data({"x": random.randint(5, 58), "y": random.randint(5, 58)})

    def _heuristic(self, non_reset: list, snap: dict) -> GameAction:
        """Smart heuristic: cycle actions with varied click targets."""
        idx = self.action_count % len(non_reset)
        action_num = non_reset[idx]
        action = GameAction[f"ACTION{action_num}"]
        if action.is_complex():
            self._set_click_data(action, snap)
        return action


# ─── Game Runner ──────────────────────────────────────────────────────

def run_single_game(arc, env_info, reasoner: Optional[LLMReasoner],
                    max_actions: int) -> dict:
    """Run agent on a single game. Thread-safe."""
    gid = env_info.game_id
    title = gid.split("-")[0].upper()
    try:
        env = arc.make(gid)
        agent = HermesAgent(env, gid, max_actions=max_actions, reasoner=reasoner)
        result = agent.run()
        return result
    except Exception as e:
        logger.error(f"{title} error: {e}")
        return {"game_id": gid, "error": str(e)[:100], "levels_completed": 0,
                "win_levels": 0, "actions": 0, "pct": 0, "llm_calls": 0}


def main(max_actions: int = 120, parallel: bool = True, workers: int = 4):
    import arc_agi

    print("Initializing ARC-AGI-3...")
    arc = arc_agi.Arcade(arc_api_key=ARC_API_KEY)
    envs = arc.get_environments()
    print(f"Found {len(envs)} games | max_actions={max_actions} | "
          f"parallel={parallel} (workers={workers})")

    # Shared LLM reasoner (thread-safe via session lock)
    reasoner = None
    if NIM_API_KEY:
        reasoner = LLMReasoner(api_key=NIM_API_KEY, model="z-ai/glm-4.7", timeout=30)
        print(f"✅ LLM ready: {reasoner.model}")

    all_results = []
    start_time = time.time()

    if parallel and len(envs) > 1:
        print(f"\n🚀 Running {len(envs)} games in parallel ({workers} workers)...\n")
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(run_single_game, arc, env_info, reasoner, max_actions): env_info
                for env_info in envs
            }
            for future in as_completed(futures):
                env_info = futures[future]
                title = env_info.game_id.split("-")[0].upper()
                try:
                    result = future.result(timeout=600)
                except Exception as e:
                    result = {"game_id": env_info.game_id, "error": str(e)[:100],
                              "levels_completed": 0, "win_levels": 0, "actions": 0,
                              "pct": 0, "llm_calls": 0}
                all_results.append(result)
                r = result
                if r.get("error"):
                    print(f"  {title:<8s} ❌ {r['error'][:40]}")
                else:
                    print(f"  {title:<8s} {r.get('levels_completed',0)}/{r.get('win_levels',0)} "
                          f"({r.get('pct',0):>5.1f}%) | {r.get('actions',0)} actions | "
                          f"LLM: {r.get('llm_calls',0)}")
    else:
        for i, env_info in enumerate(envs):
            gid = env_info.game_id
            title = gid.split("-")[0].upper()
            print(f"\n[{i+1}/{len(envs)}] {title} ({gid})")
            result = run_single_game(arc, env_info, reasoner, max_actions)
            all_results.append(result)
            if result.get("error"):
                print(f"  ❌ {result['error'][:40]}")

    elapsed = time.time() - start_time

    # ── Summary ──
    total_levels = sum(r.get("levels_completed", 0) for r in all_results)
    total_possible = sum(r.get("win_levels", 0) for r in all_results)
    total_pct = (total_levels / total_possible * 100) if total_possible else 0
    total_llm = sum(r.get("llm_calls", 0) for r in all_results)
    total_actions = sum(r.get("actions", 0) for r in all_results)

    print(f"\n{'='*60}")
    print(f"  FINAL RESULTS")
    print(f"{'='*60}")
    for r in sorted(all_results, key=lambda x: -x.get("pct", 0)):
        gid = r.get("game_id", "?")
        title = gid.split("-")[0].upper() if gid else "?"
        err = r.get("error", "")
        if err:
            print(f"  {title:<8s} ❌ {err[:50]}")
        else:
            print(f"  {title:<8s} {r.get('levels_completed',0)}/{r.get('win_levels',0)} "
                  f"({r.get('pct',0):>5.1f}%) | {r.get('actions',0)} actions | "
                  f"LLM: {r.get('llm_calls',0)}")

    print(f"{'─'*60}")
    print(f"  TOTAL: {total_levels}/{total_possible} ({total_pct:.1f}%)")
    print(f"  Score: {total_pct/100:.4f}")
    print(f"  Time: {elapsed:.0f}s ({elapsed/60:.1f}m)")
    print(f"  Actions: {total_actions} | LLM calls: {total_llm} | "
          f"API tokens: {reasoner.total_tokens if reasoner else 0}")

    with open("/kaggle/working/results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to /kaggle/working/results.json")


if __name__ == "__main__":
    _max_actions = int(os.environ.get("MAX_ACTIONS", "120"))
    _parallel = os.environ.get("PARALLEL", "1") == "1"
    _workers = int(os.environ.get("WORKERS", "4"))
    main(max_actions=_max_actions, parallel=_parallel, workers=_workers)
