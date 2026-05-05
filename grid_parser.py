#!/usr/bin/env python3
"""
Grid Parser & State Tracker for ARC-AGI-3.

Parses the multi-layer [layers][64][64] pixel grid into structured objects,
tracks state changes between frames, and builds compact representations
for LLM reasoning.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from arcengine import FrameData, GameAction, GameState

logger = logging.getLogger(__name__)


# ─── Data classes ───────────────────────────────────────────────────────

@dataclass
class BoundingBox:
    """Axis-aligned bounding box."""
    y0: int
    x0: int
    y1: int
    x1: int

    @property
    def height(self) -> int:
        return self.y1 - self.y0 + 1

    @property
    def width(self) -> int:
        return self.x1 - self.x0 + 1

    @property
    def area(self) -> int:
        return self.height * self.width

    @property
    def center(self) -> tuple[int, int]:
        return ((self.y0 + self.y1) // 2, (self.x0 + self.x1) // 2)


@dataclass
class GridObject:
    """A connected component (object) on a single grid layer."""
    color: int
    bbox: BoundingBox
    pixel_count: int
    layer: int
    pixels: list[tuple[int, int]] = field(default_factory=list)

    def compact_str(self) -> str:
        return (f"Obj(color={self.color}, layer={self.layer}, "
                f"pos={self.bbox.y0},{self.bbox.x0}, "
                f"size={self.bbox.width}x{self.bbox.height}, "
                f"pixels={self.pixel_count})")


@dataclass
class FrameSnapshot:
    """Parsed snapshot of a single frame."""
    num_layers: int
    grid_size: tuple[int, int]
    state: GameState
    levels_completed: int
    win_levels: int
    available_actions: list[int]
    objects: list[GridObject]
    color_histogram: dict[int, int]  # color -> pixel count
    non_zero_pixels: int
    total_pixels: int
    _raw_obs: Optional[FrameData] = None  # stored separately for ASCII render


@dataclass
class FrameDiff:
    """Difference between two frame snapshots."""
    new_objects: list[GridObject] = field(default_factory=list)
    disappeared_objects: list[GridObject] = field(default_factory=list)
    moved_objects: list[dict] = field(default_factory=list)  # {object, dx, dy}
    color_changes: dict[str, int] = field(default_factory=dict)  # "color:+N" -> count
    pixel_delta: int = 0  # net change in non-zero pixels

    def is_trivial(self) -> bool:
        """No meaningful change."""
        return (not self.new_objects and not self.disappeared_objects
                and not self.moved_objects and self.pixel_delta == 0)

    def summary(self) -> str:
        parts = []
        if self.new_objects:
            parts.append(f"+{len(self.new_objects)} obj")
        if self.disappeared_objects:
            parts.append(f"-{len(self.disappeared_objects)} obj")
        if self.moved_objects:
            parts.append(f"~{len(self.moved_objects)} moved")
        if self.pixel_delta != 0:
            parts.append(f"Δpx={self.pixel_delta:+d}")
        return " | ".join(parts) if parts else "no change"


# ─── Grid Parser ────────────────────────────────────────────────────────

class GridParser:
    """Parses raw frame data into structured representations."""

    # ARC-AGI color palette (index -> name)
    COLOR_NAMES = {
        0: "bg", 1: "blue", 2: "green", 3: "red", 4: "yellow",
        5: "gray", 6: "magenta", 7: "orange", 8: "cyan", 9: "brown",
        10: "light_blue", 11: "black", 12: "white", 13: "pink",
        14: "light_gray", 15: "dark_gray",
    }

    # Standard object colors (non-background)
    OBJECT_COLORS = set(range(1, 16))

    @staticmethod
    def parse_frame(obs: FrameData) -> Optional[FrameSnapshot]:
        """Parse a FrameData into a FrameSnapshot."""
        if obs is None or obs.frame is None or len(obs.frame) == 0:
            return None

        frame = obs.frame
        num_layers = len(frame)
        # All layers should be 64x64
        grid_h = len(frame[0])
        grid_w = len(frame[0][0]) if grid_h > 0 else 0
        total_pixels = num_layers * grid_h * grid_w

        # Extract objects from all layers
        all_objects = []
        color_hist = defaultdict(int)
        non_zero = 0

        for layer_idx, layer in enumerate(frame):
            arr = np.array(layer, dtype=np.int32)
            unique, counts = np.unique(arr, return_counts=True)
            for c, cnt in zip(unique, counts):
                color_hist[int(c)] += int(cnt)
                if c != 0:
                    non_zero += int(cnt)

            # Find connected components per color
            for color in [int(c) for c in unique if c != 0]:
                mask = (arr == color)
                objects = GridParser._find_connected_components(mask)
                for obj_pixels in objects:
                    if not obj_pixels:
                        continue
                    ys = [p[0] for p in obj_pixels]
                    xs = [p[1] for p in obj_pixels]
                    obj = GridObject(
                        color=color,
                        bbox=BoundingBox(min(ys), min(xs), max(ys), max(xs)),
                        pixel_count=len(obj_pixels),
                        layer=layer_idx,
                        pixels=obj_pixels,
                    )
                    all_objects.append(obj)

        return FrameSnapshot(
            num_layers=num_layers,
            grid_size=(grid_h, grid_w),
            state=obs.state,
            levels_completed=obs.levels_completed,
            win_levels=obs.win_levels,
            available_actions=obs.available_actions,
            objects=all_objects,
            color_histogram=dict(color_hist),
            non_zero_pixels=non_zero,
            total_pixels=total_pixels,
            _raw_obs=obs,
        )

    @staticmethod
    def _find_connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
        """Find 4-connected components in a boolean mask."""
        visited = set()
        components = []

        h, w = mask.shape
        for y in range(h):
            for x in range(w):
                if mask[y, x] and (y, x) not in visited:
                    # BFS
                    component = []
                    queue = [(y, x)]
                    visited.add((y, x))
                    while queue:
                        cy, cx = queue.pop(0)
                        component.append((cy, cx))
                        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            ny, nx = cy + dy, cx + dx
                            if (0 <= ny < h and 0 <= nx < w
                                    and mask[ny, nx]
                                    and (ny, nx) not in visited):
                                visited.add((ny, nx))
                                queue.append((ny, nx))
                    components.append(component)

        return components

    @staticmethod
    def color_name(color: int) -> str:
        return GridParser.COLOR_NAMES.get(color, f"c{color}")


# ─── State Tracker ──────────────────────────────────────────────────────

class StateTracker:
    """Tracks frame history and computes diffs."""

    def __init__(self, max_history: int = 50):
        self.snapshots: list[FrameSnapshot] = []
        self.diffs: list[FrameDiff] = []
        self.max_history = max_history

    def update(self, obs: FrameData) -> Optional[FrameSnapshot]:
        """Parse and store a new frame, compute diff from previous."""
        snap = GridParser.parse_frame(obs)
        if snap is None:
            return None

        if self.snapshots:
            prev = self.snapshots[-1]
            diff = StateTracker._compute_diff(prev, snap)
            self.diffs.append(diff)
        else:
            self.diffs.append(FrameDiff())

        self.snapshots.append(snap)

        # Trim history
        if len(self.snapshots) > self.max_history:
            self.snapshots = self.snapshots[-self.max_history:]
            self.diffs = self.diffs[-self.max_history:]

        return snap

    @property
    def last_snapshot(self) -> Optional[FrameSnapshot]:
        return self.snapshots[-1] if self.snapshots else None

    @property
    def last_diff(self) -> Optional[FrameDiff]:
        return self.diffs[-1] if self.diffs else None

    @property
    def action_count(self) -> int:
        return len(self.snapshots) - 1  # -1 because first is RESET

    def get_color_changes_for_action(self, action_idx: int) -> Optional[FrameDiff]:
        """Get the diff caused by action at index."""
        if 0 <= action_idx < len(self.diffs):
            return self.diffs[action_idx]
        return None

    @staticmethod
    def _compute_diff(prev: FrameSnapshot, curr: FrameSnapshot) -> FrameDiff:
        """Compare two snapshots and compute changes."""
        diff = FrameDiff()

        diff.pixel_delta = curr.non_zero_pixels - prev.non_zero_pixels

        # Color histogram changes
        all_colors = set(prev.color_histogram) | set(curr.color_histogram)
        for c in all_colors:
            p = prev.color_histogram.get(c, 0)
            cu = curr.color_histogram.get(c, 0)
            delta = cu - p
            if delta != 0:
                cn = GridParser.color_name(c)
                diff.color_changes[f"{cn}({c})"] = delta

        # Object-level comparison (simple: match by color+layer+pixel_count tolerance)
        prev_objs = {(o.color, o.layer, o.bbox.center): o for o in prev.objects}
        curr_objs = {(o.color, o.layer, o.bbox.center): o for o in curr.objects}

        prev_keys = set(prev_objs.keys())
        curr_keys = set(curr_objs.keys())

        for key in curr_keys - prev_keys:
            diff.new_objects.append(curr_objs[key])

        for key in prev_keys - curr_keys:
            diff.disappeared_objects.append(prev_objs[key])

        # Check for moved objects (same color+layer, different position)
        prev_by_color = defaultdict(list)
        curr_by_color = defaultdict(list)
        for o in prev.objects:
            prev_by_color[(o.color, o.layer)].append(o)
        for o in curr.objects:
            curr_by_color[(o.color, o.layer)].append(o)

        for (color, layer), prev_list in prev_by_color.items():
            curr_list = curr_by_color.get((color, layer), [])
            if len(prev_list) == len(curr_list) and len(prev_list) == 1:
                p, c = prev_list[0], curr_list[0]
                if p.pixel_count == c.pixel_count:
                    dy = c.bbox.center[0] - p.bbox.center[0]
                    dx = c.bbox.center[1] - p.bbox.center[1]
                    if abs(dy) <= 20 and abs(dx) <= 20 and (dy != 0 or dx != 0):
                        diff.moved_objects.append({
                            "color": color,
                            "layer": layer,
                            "dx": dx, "dy": dy,
                            "from": (p.bbox.center),
                            "to": (c.bbox.center),
                        })

        return diff

    def build_llm_context(self, max_tokens: int = 2000) -> str:
        """Build a compact text summary of current state for LLM reasoning."""
        snap = self.last_snapshot
        if snap is None:
            return "No frames observed yet."

        lines = []
        lines.append(f"Grid: {snap.grid_size[1]}x{snap.grid_size[0]}, Layers: {snap.num_layers}")
        lines.append(f"State: {snap.state.name}, Actions taken: {self.action_count}")
        lines.append(f"Levels: {snap.levels_completed}/{snap.win_levels} completed")
        lines.append(f"Available: {snap.available_actions}")

        # Color summary
        colors_sorted = sorted(
            snap.color_histogram.items(), key=lambda x: -x[1]
        )
        color_parts = []
        for c, cnt in colors_sorted:
            if c == 0:
                continue
            cn = GridParser.color_name(c)
            color_parts.append(f"{cn}={cnt}px")
        lines.append(f"Objects: {', '.join(color_parts)}")

        # Object list (top 20 by size)
        top_objs = sorted(snap.objects, key=lambda o: -o.pixel_count)[:20]
        for obj in top_objs:
            lines.append(f"  {obj.compact_str()}")

        # Recent action history (last 5 diffs)
        if len(self.diffs) > 1:
            lines.append("Recent changes:")
            for i, d in enumerate(self.diffs[-6:]):
                action_num = len(self.diffs) - 6 + i
                lines.append(f"  #{action_num}: {d.summary()}")

        text = "\n".join(lines)

        # Truncate if needed
        if len(text) > max_tokens * 4:  # rough char estimate
            lines = lines[:len(lines) // 2]
            lines.append("... (truncated)")
            text = "\n".join(lines)

        return text

    def frame_to_ascii(self, obs: Optional[FrameData] = None, max_w: int = 32) -> str:
        """Render the frame as ASCII art (downscaled) for visual inspection.
        Uses last snapshot if obs is None."""
        if obs is None:
            if self.snapshots:
                obs = self.snapshots[-1]._obs if hasattr(self.snapshots[-1], '_obs') else None
            if obs is None or obs.frame is None:
                return "(no frame)"

        # Use first layer
        layer = obs.frame[0]
        arr = np.array(layer, dtype=np.int32)

        h, w = arr.shape
        scale = max(1, max(h, w) // max_w)

        # Color chars
        char_map = {
            0: ".", 1: "#", 2: "#", 3: "#", 4: "#",
            5: "=", 6: "@", 7: "$", 8: "+", 9: "%",
            10: "~", 11: "*", 12: "O", 13: "&", 14: "-",
            15: ":",
        }

        lines = []
        for y in range(0, h, scale):
            row = ""
            for x in range(0, w, scale):
                c = int(arr[y, x])
                row += char_map.get(c, str(c % 10))
            lines.append(row)

        return "\n".join(lines)
