"""
HermesAgent — ARC-AGI-3 Competition Entry (OPTIMIZED)
LLM-powered grid puzzle solver using NVIDIA NIM (glm-4.7)

Optimizations over v1:
  1. LLM only when stuck (not every Nth step) — ~70% fewer API calls
  2. max_actions 200→80 — 60% fewer steps per game
  3. explore_budget 12→6 — faster ramp-up
  4. Early bailout after 30 actions with 0 progress
  5. Parallel game processing (ThreadPoolExecutor, 4 workers)
  6. Faster model (glm-4.7) + lower timeout (25s)
  7. No raw_obs stored in history (memory savings)
  8. requests.Session for HTTP connection pooling
  9. Skip ASCII rendering when LLM isn't called

Estimated speedup: 5-8× wall-clock time.
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

# ─── Grid Parser & State Tracker (optimized) ──────────────────────────
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
    """Lightweight state tracker — no raw_obs storage."""

    def __init__(self):
        self.history: list[dict] = []
        self._prev_px = 0

    def update(self, obs: FrameData) -> dict:
        frame = obs.frame
        layers = len(frame)
        h, w = len(frame[0]), len(frame[0][0])
        # Vectorized grid merge (much faster than nested loops)
        grid = np.zeros((h, w), dtype=np.int8)
        for layer in range(layers):
            arr = np.array(frame[layer], dtype=np.int8)
            grid[arr != 0] = arr[arr != 0]
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

    def frame_to_ascii(self, obs: FrameData, max_w: int = 32) -> str:
        """Render grid as ASCII — only called when LLM needs it."""
        frame = obs.frame
        h, w = len(frame[0]), len(frame[0][0])
        grid = np.zeros((h, w), dtype=np.int8)
        for layer in frame:
            arr = np.array(layer, dtype=np.int8)
            grid[arr != 0] = arr[arr != 0]
        chars = " .:#@%&+=-*~"
        scale = max(1, w // max_w)
        lines = []
        for r in range(0, h, scale):
            row = ""
            for c in range(0, w, scale):
                v = grid[r][c]
                row += chars[v % len(chars)]
            lines.append(row)
        return "\n".join(lines)


# ─── LLM Reasoner (optimized) ─────────────────────────────────────────
import requests

# Thread-safe session pool
_http_session = requests.Session()
_http_session.headers.update({"Content-Type": "application/json"})

# Thread lock for session access
_session_lock = threading.Lock()


class LLMReasoner:
    def __init__(self, api_key: str, model: str = "z-ai/glm-4.7",
                 base_url: str = "https://integrate.api.nvidia.com/v1", timeout: int = 25):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self.total_tokens = 0
        self.calls = 0

    def reason(self, snap: dict, tracker: StateTracker, step_num: int,
               obs: Optional[FrameData] = None) -> Optional[dict]:
        """Reason about next action. Only called when agent is stuck."""
        # Render ASCII only when needed
        ascii_art = ""
        if obs and obs.frame:
            ascii_art = tracker.frame_to_ascii(obs, max_w=32)

        available = snap.get("available_actions", [])
        actions_str = ", ".join([f"ACTION{a}" for a in available if a != 0])

        # Compact prompt — minimize tokens
        prompt = (
            f"Grid puzzle. 64x64, colors 0-15.\n"
            f"Actions: {actions_str}\n"
            f"Objects: {snap['objects']}, Pixels: {snap['total_pixels']}\n"
            f"State: {snap['state'].name}, Level: {snap['levels_completed']}/{snap['win_levels']}\n"
            f"Step: {step_num}, stuck for several turns.\n\n"
            f"{ascii_art}\n\n"
            f"REASONING:\nNEXT_ACTION: ACTION<n>\nCONFIDENCE: 0.0-1.0"
        )

        messages = [
            {"role": "system", "content": "Expert grid puzzle solver. Concise, decisive. Respond in format: REASONING: ...\nNEXT_ACTION: ACTION<n>\nCONFIDENCE: 0.0-1.0"},
            {"role": "user", "content": prompt}
        ]

        try:
            with _session_lock:
                resp = _http_session.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={"model": self.model, "messages": messages,
                          "max_tokens": 128, "temperature": 0.2},
                    timeout=self.timeout
                )
            self.calls += 1
            if resp.status_code != 200:
                logger.warning(f"LLM {resp.status_code}: {resp.text[:80]}")
                return None
            data = resp.json()
            self.total_tokens += data.get("usage", {}).get("total_tokens", 0)
            text = data["choices"][0]["message"]["content"]
            return self._parse_response(text, available)
        except requests.exceptions.Timeout:
            logger.warning("LLM timeout")
            return None
        except Exception as e:
            logger.warning(f"LLM error: {e}")
            return None

    def _parse_response(self, text: str, available: list) -> Optional[dict]:
        action_num = None
        reasoning = ""
        confidence = 0.5
        for line in text.strip().split("\n"):
            line = line.strip()
            up = line.upper()
            if up.startswith("REASONING:"):
                reasoning = line.split(":", 1)[1].strip()
            elif up.startswith("NEXT_ACTION:"):
                val = line.split(":", 1)[1].strip().upper()
                for part in val.replace("ACTION", "").replace(" ", ""):
                    if part.isdigit():
                        action_num = int(part)
                        break
            elif up.startswith("CONFIDENCE:"):
                try:
                    confidence = float(line.split(":")[1].strip())
                except ValueError:
                    pass
        if action_num and action_num in available:
            return {"action": action_num, "reasoning": reasoning[:100], "confidence": confidence}
        return None


# ─── HermesAgent (optimized) ──────────────────────────────────────────

class HermesAgent:
    """Full agent: explore → heuristic play → LLM rescue when stuck."""

    # ── Tunable defaults (can override per-game) ──
    DEFAULT_MAX_ACTIONS = 80
    EXPLORE_BUDGET = 6
    STUCK_THRESHOLD = 6          # LLM kicks in after this many stuck steps
    EARLY_BAILOUT_ACTIONS = 30   # bail if 0 levels after this many actions
    EARLY_BAILOUT_RESETS = 3     # bail if this many resets with no wins

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

        # Strategy
        self.explore_done = False
        self.consecutive_stuck = 0
        self.last_pixel_count = 0
        self.total_resets = 0
        self._last_obs = None  # only keep latest obs for ASCII rendering

    def run(self) -> dict:
        """Main agent loop with early bailout."""
        print(f"\n{'='*50}")
        print(f"  HermesAgent: {self.game_id}")
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
                self.total_resets = 0
                print(f"  🎉 Level WON! ({self.levels_completed}/{self.win_levels})")
                obs = self._reset()
                if not obs:
                    break
                continue

            if obs.state == GameState.GAME_OVER:
                self.consecutive_stuck = 0
                self.total_resets += 1
                # Early bailout: too many resets without progress
                if (self.total_resets >= self.EARLY_BAILOUT_RESETS
                        and self.levels_completed == 0):
                    print(f"  ⏭️ Early bail: {self.total_resets} resets, 0 levels")
                    break
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

            # Hard stuck → reset
            if self.consecutive_stuck >= 10:
                self.consecutive_stuck = 0
                self.total_resets += 1
                if (self.total_resets >= self.EARLY_BAILOUT_RESETS
                        and self.levels_completed == 0):
                    print(f"  ⏭️ Early bail: stuck {self.total_resets} times")
                    break
                print(f"  🔄 Stuck, resetting...")
                obs = self._reset()
                if not obs:
                    break
                continue

            # ── Early bailout: no progress after N actions ──
            if (self.action_count >= self.EARLY_BAILOUT_ACTIONS
                    and self.levels_completed == 0
                    and self.consecutive_stuck >= 3):
                print(f"  ⏭️ Early bail: {self.action_count} actions, 0 levels, stuck={self.consecutive_stuck}")
                break

            # ── Choose action ──
            action = self._choose(obs, snap)
            if not action:
                break

            # ── Execute ──
            obs = self._execute(action)
            if not obs:
                self.total_resets += 1
                if self.total_resets >= self.EARLY_BAILOUT_RESETS and self.levels_completed == 0:
                    break
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
            self._last_obs = obs  # keep only latest for ASCII
            self.action_count += 1
        return obs

    def _execute(self, action: GameAction) -> Optional[FrameData]:
        data = action.action_data.model_dump() if hasattr(action, 'action_data') and action.action_data else {}
        try:
            obs = self.env.step(action, data=data)
            if obs:
                self.tracker.update(obs)
                self._last_obs = obs
                self.action_count += 1
                if self.action_count % 20 == 0:
                    snap = self.tracker.history[-1]
                    print(f"  [{self.action_count}/{self.max_actions}] "
                          f"{action.name} -> {obs.state.name} | "
                          f"objs={snap['objects']} px={snap['total_pixels']}")
            return obs
        except Exception as e:
            print(f"  ❌ {action.name} error: {e}")
            return None

    def _choose(self, obs: FrameData, snap: dict) -> Optional[GameAction]:
        available = obs.available_actions or []
        non_reset = [a for a in available if a != GameAction.RESET.value]
        if not non_reset:
            return GameAction.RESET

        # ── Phase 1: Quick exploration (6 actions) ──
        if not self.explore_done and self.action_count <= self.EXPLORE_BUDGET:
            self.explore_done = self.action_count >= self.EXPLORE_BUDGET
            idx = self.action_count % len(non_reset)
            action_num = non_reset[idx]
            action = GameAction[f"ACTION{action_num}"]
            if action.is_complex():
                self._set_click_data(action, snap)
            return action

        # ── Phase 2: LLM rescue ONLY when stuck ──
        if (self.reasoner
                and self.consecutive_stuck >= self.STUCK_THRESHOLD
                and self.levels_completed < self.win_levels):
            try:
                plan = self.reasoner.reason(snap, self.tracker, self.action_count, obs)
                self.llm_calls += 1
                if plan and plan.get("confidence", 0) > 0.3:
                    action_num = plan["action"]
                    action = GameAction[f"ACTION{action_num}"]
                    if action.is_complex():
                        self._set_click_data(action, snap)
                    self.consecutive_stuck = 0  # reset stuck counter after LLM call
                    return action
            except Exception as e:
                logger.warning(f"LLM failed: {e}")

        # ── Phase 3: Smart heuristic ──
        return self._heuristic(non_reset, snap)

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
            # Rotate through objects to explore different targets
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


# ─── Game Runner (with optional parallelism) ──────────────────────────

def run_single_game(arc, env_info: dict, reasoner: Optional[LLMReasoner],
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


def main(max_actions: int = 80, parallel: bool = True, workers: int = 4):
    import arc_agi

    print("Initializing ARC-AGI-3...")
    arc = arc_agi.Arcade(arc_api_key=ARC_API_KEY)
    envs = arc.get_environments()
    print(f"Found {len(envs)} games | max_actions={max_actions} | "
          f"parallel={parallel} (workers={workers})")

    # Shared LLM reasoner (thread-safe via session lock)
    reasoner = None
    if NIM_API_KEY:
        reasoner = LLMReasoner(api_key=NIM_API_KEY, model="z-ai/glm-4.7", timeout=25)
        print(f"✅ LLM ready: {reasoner.model}")

    all_results = []
    start_time = time.time()

    if parallel and len(envs) > 1:
        # ── Parallel mode ──
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
                    result = future.result(timeout=600)  # 10 min per game max
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
        # ── Sequential mode (fallback) ──
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

    # Save results
    with open("/kaggle/working/results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to /kaggle/working/results.json")


if __name__ == "__main__":
    # CLI flags for easy tuning
    _max_actions = int(os.environ.get("MAX_ACTIONS", "80"))
    _parallel = os.environ.get("PARALLEL", "1") == "1"
    _workers = int(os.environ.get("WORKERS", "4"))
    main(max_actions=_max_actions, parallel=_parallel, workers=_workers)
