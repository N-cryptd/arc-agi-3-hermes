#!/usr/bin/env python3
"""
Exploration Module for ARC-AGI-3.

Systematically probes each available action to discover:
- What each action does (movement, click, toggle, etc.)
- How actions interact (action ordering effects)
- Game mechanics (gravity, collision, win conditions)
- Action mappings (ACTION1 -> "up", ACTION2 -> "down", etc.)

Uses the GridParser/StateTracker for state comparison.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from arcengine import GameAction, GameState

from grid_parser import FrameDiff, GridParser, StateTracker

logger = logging.getLogger(__name__)


@dataclass
class ActionProfile:
    """Learned profile of a single action."""
    action: str  # "ACTION1", "ACTION2", etc.
    num_tried: int = 0
    avg_pixel_delta: float = 0.0
    avg_new_objects: float = 0.0
    avg_disappeared: float = 0.0
    avg_moved: float = 0.0
    causes_win: bool = False
    causes_game_over: bool = False
    needs_coordinates: bool = False
    coordinate_sensitive: bool = False
    coordinate_effects: dict[str, Any] = field(default_factory=dict)
    inferred_label: str = ""  # "move_up", "click", "toggle", etc.

    def update(self, diff: FrameDiff, state_after: GameState):
        """Update profile with new observation."""
        self.num_tried += 1
        n = self.num_tried
        self.avg_pixel_delta = (self.avg_pixel_delta * (n - 1) + diff.pixel_delta) / n
        self.avg_new_objects = (self.avg_new_objects * (n - 1) + len(diff.new_objects)) / n
        self.avg_disappeared = (self.avg_disappeared * (n - 1) + len(diff.disappeared_objects)) / n
        self.avg_moved = (self.avg_moved * (n - 1) + len(diff.moved_objects)) / n
        if state_after == GameState.WIN:
            self.causes_win = True
        if state_after == GameState.GAME_OVER:
            self.causes_game_over = True


@dataclass
class GameProfile:
    """Learned profile of a game's mechanics."""
    game_id: str
    tags: list[str]
    action_profiles: dict[str, ActionProfile] = field(default_factory=dict)
    has_gravity: bool = False
    has_collision: bool = False
    is_click_based: bool = False
    is_movement_based: bool = False
    num_layers_min: int = 1
    num_layers_max: int = 1
    total_probes: int = 0
    exploration_budget: int = 20  # max actions to spend exploring

    def summary(self) -> str:
        lines = [f"Game {self.game_id} profile:"]
        for name, ap in self.action_profiles.items():
            label = f" ({ap.inferred_label})" if ap.inferred_label else ""
            lines.append(
                f"  {name}{label}: "
                f"tried={ap.num_tried}, "
                f"Δpx={ap.avg_pixel_delta:+.1f}, "
                f"+obj={ap.avg_new_objects:.1f}, "
                f"-obj={ap.avg_disappeared:.1f}, "
                f"~moved={ap.avg_moved:.1f}"
            )
        mechanics = []
        if self.has_gravity:
            mechanics.append("gravity")
        if self.has_collision:
            mechanics.append("collision")
        if self.is_click_based:
            mechanics.append("click")
        if self.is_movement_based:
            mechanics.append("movement")
        if mechanics:
            lines.append(f"  Mechanics: {', '.join(mechanics)}")
        return "\n".join(lines)


class Explorer:
    """Explores game mechanics through systematic action probing."""

    def __init__(self, env, game_id: str, tags: list[str],
                 budget: int = 20):
        self.env = env
        self.game_id = game_id
        self.tags = tags
        self.tracker = StateTracker()
        self.profile = GameProfile(game_id=game_id, tags=tags, exploration_budget=budget)
        self.explored = False

    def explore(self) -> GameProfile:
        """Run full exploration sequence."""
        if self.explored:
            return self.profile

        logger.info(f"Exploring {self.game_id} (budget={self.profile.exploration_budget})")

        # Phase 1: Reset and capture initial state
        obs = self._reset()
        if obs is None:
            logger.error(f"Failed to reset {self.game_id}")
            self.explored = True
            return self.profile

        snap = self.tracker.last_snapshot
        self.profile.num_layers_min = snap.num_layers
        self.profile.num_layers_max = snap.num_layers

        # Detect click-based vs movement-based from tags
        self.profile.is_click_based = 'click' in self.tags
        self.profile.is_movement_based = 'keyboard' in self.tags

        # Phase 2: Probe each action individually
        actions = getattr(self.env, 'action_space', [])
        budget_used = 0

        for action in actions:
            if budget_used >= self.profile.exploration_budget:
                break

            action_name = action.name
            profile = self.profile.action_profiles.get(
                action_name, ActionProfile(action=action_name)
            )

            # Check if action needs coordinates
            obs = self._reset()
            if obs is None:
                continue
            result = self._try_action(action)
            budget_used += 1

            if result is None:
                # Action might need coordinates
                profile.needs_coordinates = True
                logger.info(f"  {action_name} needs coordinates (null response)")

                # Try with center coordinates
                for coords in [(32, 32), (16, 16), (48, 48)]:
                    if budget_used >= self.profile.exploration_budget:
                        break
                    obs = self._reset()
                    result = self._try_action_with_coords(action, *coords)
                    budget_used += 1
                    if result is not None:
                        diff = self.tracker.last_diff
                        profile.coordinate_sensitive = True
                        profile.update(diff, result.state)
                        key = f"({coords[0]},{coords[1]})"
                        profile.coordinate_effects[key] = diff.summary()
                        break
            else:
                diff = self.tracker.last_diff
                profile.update(diff, result.state)

                # Detect movement from direction of moved objects
                if diff.moved_objects and profile.num_tried >= 1:
                    avg_dy = np.mean([m['dy'] for m in diff.moved_objects])
                    avg_dx = np.mean([m['dx'] for m in diff.moved_objects])
                    profile.inferred_label = self._classify_movement(avg_dy, avg_dx)

                # Classify non-movement actions
                if not profile.inferred_label and profile.num_tried >= 2:
                    profile.inferred_label = self._classify_action(profile)

            self.profile.action_profiles[action_name] = profile

            # Track layer count changes
            if result and result.frame:
                self.profile.num_layers_max = max(
                    self.profile.num_layers_max, len(result.frame)
                )

        # Phase 3: Detect game mechanics
        self._detect_mechanics()

        # Phase 4: Probe action sequences for interaction effects
        if budget_used < self.profile.exploration_budget:
            self._probe_sequences(budget_used)

        logger.info(f"Exploration complete: {budget_used} probes used")
        for name, ap in self.profile.action_profiles.items():
            if ap.inferred_label:
                logger.info(f"  {name} -> {ap.inferred_label}")

        self.explored = True
        return self.profile

    def _reset(self):
        """Reset the game and track the frame."""
        obs = self.env.step(GameAction.RESET)
        if obs is None:
            return None
        self.tracker.update(obs)
        return obs

    def _try_action(self, action) -> Any:
        """Try an action without coordinates. Returns obs or None."""
        try:
            obs = self.env.step(action)
            if obs is None:
                return None
            self.tracker.update(obs)
            return obs
        except Exception as e:
            logger.debug(f"Action {action} failed: {e}")
            return None

    def _try_action_with_coords(self, action, x: int, y: int) -> Any:
        """Try an action with coordinates."""
        try:
            obs = self.env.step(action, data={'x': x, 'y': y})
            if obs is None:
                return None
            self.tracker.update(obs)
            return obs
        except Exception as e:
            logger.debug(f"Action {action} at ({x},{y}) failed: {e}")
            return None

    def _classify_movement(self, avg_dy: float, avg_dx: float) -> str:
        """Classify a movement action by average displacement."""
        threshold = 0.5
        if abs(avg_dy) > threshold and abs(avg_dx) < threshold:
            return "move_down" if avg_dy > 0 else "move_up"
        elif abs(avg_dx) > threshold and abs(avg_dy) < threshold:
            return "move_right" if avg_dx > 0 else "move_left"
        elif abs(avg_dy) > threshold and abs(avg_dx) > threshold:
            # Diagonal movement
            parts = []
            parts.append("down" if avg_dy > 0 else "up")
            parts.append("right" if avg_dx > 0 else "left")
            return "move_" + "_".join(parts)
        return ""

    def _classify_action(self, profile: ActionProfile) -> str:
        """Classify a non-movement action by its effects."""
        if profile.causes_win:
            return "win_action"
        if profile.causes_game_over:
            return "danger_action"

        if profile.needs_coordinates:
            return "click"

        # Check for toggle actions (no net pixel change, objects appear/disappear)
        if abs(profile.avg_pixel_delta) < 5 and profile.avg_new_objects > 0.5:
            return "toggle"

        # Check for spawn actions (net pixel increase)
        if profile.avg_pixel_delta > 50:
            return "spawn"

        # Check for clear/remove actions
        if profile.avg_pixel_delta < -50:
            return "clear"

        # Check for push/pull (slight displacement, same object count)
        if (profile.avg_moved > 0.5
                and profile.avg_new_objects < 0.5
                and profile.avg_disappeared < 0.5):
            return "push"

        # Rotation/transform (no pixel count change, objects rearrange)
        if abs(profile.avg_pixel_delta) < 3 and profile.avg_moved > 1:
            return "transform"

        return "unknown"

    def _detect_mechanics(self):
        """Detect game-level mechanics from action profiles."""
        # Check for gravity: objects consistently move down when nothing acts on them
        # We can detect this by checking if objects shift between non-action frames
        # (not directly testable, infer from action patterns)

        # Check for collision: objects don't overlap after movement
        for name, ap in self.profile.action_profiles.items():
            if ap.avg_moved > 0.5:
                # If moved objects but no new/disappeared objects, likely collision exists
                if ap.avg_new_objects < 0.3 and ap.avg_disappeared < 0.3:
                    self.profile.has_collision = True
                break

        # Multi-layer games often have interaction mechanics
        if self.profile.num_layers_max > 1:
            self.profile.has_collision = True

    def _probe_sequences(self, budget_used: int):
        """Probe action sequences to discover interaction effects."""
        actions = list(self.profile.action_profiles.keys())
        if len(actions) < 2:
            return

        # Try a few 2-action sequences
        remaining = self.profile.exploration_budget - budget_used
        seq_count = min(3, remaining)
        if seq_count <= 0:
            return

        import itertools
        pairs = list(itertools.combinations(actions, 2))[:seq_count]

        for a1_name, a2_name in pairs:
            a1 = self._name_to_action(a1_name)
            a2 = self._name_to_action(a2_name)
            if a1 is None or a2 is None:
                continue

            self._reset()
            self._try_action(a1)
            before_seq = self.tracker.last_snapshot
            self._try_action(a2)
            after_seq = self.tracker.last_snapshot

            if before_seq and after_seq:
                # Check if sequential effect differs from individual effects
                logger.debug(
                    f"  Seq {a1_name}+{a2_name}: "
                    f"objs={len(before_seq.objects)}->{len(after_seq.objects)}"
                )

    def _name_to_action(self, name: str):
        """Convert action name string to GameAction enum."""
        try:
            return GameAction[name]
        except KeyError:
            return None

    def get_action_description(self) -> str:
        """Get human-readable action descriptions for LLM prompting."""
        lines = []
        for name, ap in self.profile.action_profiles.items():
            if ap.inferred_label:
                lines.append(f"{name} = {ap.inferred_label}")
            elif ap.needs_coordinates:
                lines.append(f"{name} = click(x, y) — click at grid coordinates")
            else:
                lines.append(f"{name} = unknown (tried {ap.num_tried}x)")
        return "\n".join(lines) if lines else "No actions profiled yet."
