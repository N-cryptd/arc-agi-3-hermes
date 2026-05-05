#!/usr/bin/env python3
"""
LLM Reasoner for ARC-AGI-3.

Uses a vision-capable LLM (via NVIDIA NIM / OpenRouter) to:
1. Analyze the game grid visually
2. Understand game mechanics from exploration data
3. Plan action sequences to solve levels
4. Adapt strategy based on feedback

Supports both text-only reasoning and vision (image-based) reasoning.
"""

import base64
import io
import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import requests

from grid_parser import GridParser, StateTracker
from explorer import Explorer, GameProfile

logger = logging.getLogger(__name__)


# ─── Color palette for rendering ────────────────────────────────────────

# ARC-AGI color palette (RGB)
ARC_COLORS = {
    0: (0, 0, 0),         # black (background)
    1: (29, 105, 179),    # blue
    2: (49, 140, 231),    # green
    3: (215, 48, 39),     # red
    4: (244, 163, 26),    # yellow
    5: (149, 149, 149),   # gray
    6: (156, 39, 176),    # magenta
    7: (237, 121, 0),     # orange
    8: (0, 177, 159),     # cyan
    9: (108, 70, 46),     # brown
    10: (133, 193, 233),  # light blue
    11: (0, 0, 0),        # black
    12: (255, 255, 255),  # white
    13: (228, 118, 173),  # pink
    14: (199, 199, 199),  # light gray
    15: (100, 100, 100),  # dark gray
}


@dataclass
class ReasoningPlan:
    """Plan output from the LLM."""
    raw_response: str
    understanding: str = ""  # LLM's understanding of the game
    plan_steps: list[str] = field(default_factory=list)  # action sequence
    next_action: Optional[str] = None  # immediate next action to take
    confidence: float = 0.0
    reasoning: str = ""

    def parse(self):
        """Parse structured output from LLM response."""
        text = self.raw_response.strip()

        # Extract understanding
        understand_match = re.search(r'UNDERSTANDING:\s*(.+?)(?=\nPLAN:|\nNEXT:|\nCONFIDENCE:|\Z)', text, re.DOTALL)
        if understand_match:
            self.understanding = understand_match.group(1).strip()

        # Extract plan
        plan_match = re.search(r'PLAN:\s*(.+?)(?=\nNEXT:|\nCONFIDENCE:|\Z)', text, re.DOTALL)
        if plan_match:
            steps = [s.strip() for s in plan_match.group(1).strip().split('\n') if s.strip()]
            self.plan_steps = steps

        # Extract next action
        next_match = re.search(r'NEXT:\s*(.+?)(?=\n|$)', text)
        if next_match:
            self.next_action = next_match.group(1).strip()

        # Extract confidence
        conf_match = re.search(r'CONFIDENCE:\s*([\d.]+)', text)
        if conf_match:
            self.confidence = float(conf_match.group(1))

        # Extract reasoning
        reason_match = re.search(r'REASONING:\s*(.+?)(?=\nUNDERSTANDING:|\nPLAN:|\nNEXT:|\Z)', text, re.DOTALL)
        if reason_match:
            self.reasoning = reason_match.group(1).strip()

        return self


class GridRenderer:
    """Renders grid frames as images for vision-based reasoning."""

    @staticmethod
    def render_to_image(frame_data, scale: int = 4) -> bytes:
        """Render frame to PNG bytes."""
        try:
            from PIL import Image
        except ImportError:
            logger.warning("Pillow not installed, cannot render images")
            return b""

        frame = frame_data.frame if frame_data.frame else None
        if frame is None:
            return b""

        # Render first layer
        layer = frame[0]
        h, w = len(layer), len(layer[0])

        img = Image.new('RGB', (w * scale, h * scale), (0, 0, 0))
        pixels = img.load()

        for y in range(h):
            for x in range(w):
                color_idx = int(layer[y][x])
                rgb = ARC_COLORS.get(color_idx, (128, 0, 128))  # purple for unknown
                # Fill the scaled block
                for sy in range(scale):
                    for sx in range(scale):
                        pixels[x * scale + sx, y * scale + sy] = rgb

        buf = io.BytesIO()
        img.save(buf, format='PNG')
        return buf.getvalue()

    @staticmethod
    def image_to_base64(png_bytes: bytes) -> str:
        return base64.b64encode(png_bytes).decode('utf-8')


class LLMReasoner:
    """LLM-powered game reasoner."""

    def __init__(self, api_key: str, base_url: str = "https://integrate.api.nvidia.com/v1",
                 model: str = "z-ai/glm-5.1", timeout: int = 120):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        self.renderer = GridRenderer()
        self.conversation_history: list[dict] = []
        self.total_tokens_used = 0

        # Prompt templates
        self.system_prompt = """You play ARC-AGI-3 grid puzzles. 64x64 grid, colors 0-15.

Actions: {actions}
{exploration_context}

Respond EXACTLY in this format:
REASONING: <brief logic>
UNDERSTANDING: <game rules>
NEXT: <action e.g. ACTION1 or ACTION6(x=15,y=20)>
CONFIDENCE: <0.0-1.0>"""

        self.vision_prompt = """Look at the game grid image. The colored shapes on the black background are game objects.

Current state: {state}
Actions taken: {action_count}
Levels completed: {levels_completed}/{win_levels}
Available actions: {available_actions}

{action_history}

What is happening in this grid? What should I do next?

Respond:
REASONING: <your analysis>
UNDERSTANDING: <game mechanics>
NEXT: <action to take, e.g. ACTION1 or ACTION6(x=20,y=30)>
CONFIDENCE: <0.0-1.0>"""

        self.adaptive_prompt = """The game state changed after my last action.

PREVIOUS ACTION: {last_action}
CHANGE OBSERVED: {diff_summary}
CURRENT STATE: {state}
ACTIONS TAKEN: {action_count}
LEVELS: {levels_completed}/{win_levels}

{llm_context}

What should I do next?

Respond:
REASONING: <your logic>
NEXT: <action to take>
CONFIDENCE: <0.0-1.0>"""

    def _call_llm(self, messages: list[dict], temperature: float = 0.3) -> str:
        """Call the LLM API."""
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }

        payload = {
            'model': self.model,
            'messages': messages,
            'max_tokens': 512,
            'temperature': temperature,
            'stream': False
        }

        try:
            r = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=self.timeout
            )
            r.raise_for_status()
            data = r.json()
            content = data['choices'][0]['message']['content']
            tokens = data.get('usage', {}).get('total_tokens', 0)
            self.total_tokens_used += tokens
            logger.info(f"LLM response: {len(content)} chars, {tokens} tokens")
            return content
        except requests.exceptions.Timeout:
            logger.error(f"LLM timeout after {self.timeout}s")
            return "TIMEOUT"
        except Exception as e:
            logger.error(f"LLM error: {e}")
            return f"ERROR: {e}"

    def _is_multimodal(self) -> bool:
        """Check if current model supports vision."""
        multimodal_models = {'gemini', 'gpt-4', 'gpt-4o', 'claude', 'llava', 'qwen-vl'}
        return any(m in self.model.lower() for m in multimodal_models)

    def _render_vision_message(self, frame_data) -> Optional[dict]:
        """Create a vision message with the grid image."""
        png_bytes = self.renderer.render_to_image(frame_data, scale=2)
        if not png_bytes:
            return None

        b64 = self.renderer.image_to_base64(png_bytes)
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{b64}",
                "detail": "high"
            }
        }

    def reason_initial(self, explorer: Explorer, tracker: StateTracker) -> ReasoningPlan:
        """Analyze the initial game state and create a first plan."""
        profile = explorer.profile
        snap = tracker.last_snapshot
        if snap is None:
            return ReasoningPlan(raw_response="No frame data")

        action_desc = explorer.get_action_description()

        # Build exploration context
        exploration_context = f"Game exploration results:\n{profile.summary()}"

        # Format system prompt
        system = self.system_prompt.format(
            actions=action_desc,
            exploration_context=exploration_context,
        )

        # Build user message with state
        user_msg = f"Initial game state:\n{tracker.build_llm_context()}\n\nWhat are the game mechanics and what should I do first?"

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ]

        # Try vision only if using a multimodal model override
        vision_msg = self._render_vision_message(snap._raw_obs)
        if vision_msg and self._is_multimodal():
            messages[-1] = {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Analyze this game grid:\n\n{tracker.build_llm_context(max_tokens=500)}"},
                    vision_msg,
                ]
            }
            logger.info("Using vision mode for initial reasoning")
        else:
            # Use ASCII art as fallback
            ascii_art = tracker.frame_to_ascii(snap._raw_obs, max_w=32) if snap._raw_obs else "(no frame)"
            messages[-1] = {
                "role": "user",
                "content": f"Analyze this game grid:\n\nASCII preview:\n```\n{ascii_art}\n```\n\n{tracker.build_llm_context()}\n\nWhat are the game mechanics and what should I do first?"
            }
            logger.info("Using text mode for initial reasoning")

        self.conversation_history = messages.copy()
        response = self._call_llm(messages)
        self.conversation_history.append({"role": "assistant", "content": response})

        plan = ReasoningPlan(raw_response=response)
        plan.parse()
        return plan

    def reason_vision(self, snap, state_str: str, action_count: int,
                      levels_completed: int, win_levels: int,
                      available_actions: list, action_history: str = "",
                      model: str = "") -> ReasoningPlan:
        """Reason using vision (requires multimodal model).
        Falls back to text reasoning if model doesn't support vision."""
        if not model:
            model = self.model

        user_msg = self.vision_prompt.format(
            state=state_str,
            action_count=action_count,
            levels_completed=levels_completed,
            win_levels=win_levels,
            available_actions=available_actions,
            action_history=action_history,
        )

        vision_msg = self._render_vision_message(snap._raw_obs)
        if vision_msg and model != self.model:
            # Only use vision with a multimodal model
            messages = [
                {"role": "user", "content": [
                    {"type": "text", "text": user_msg},
                    vision_msg,
                ]}
            ]
        else:
            # Fallback: use ASCII art as text
            ascii_art = "(no frame)"
            if snap._raw_obs and snap._raw_obs.frame:
                from grid_parser import GridParser
                layer = snap._raw_obs.frame[0]
                arr = __import__('numpy').array(layer)
                char_map = {0: ".", 1: "#", 2: "#", 3: "#", 4: "#", 5: "=", 6: "@", 7: "$", 8: "+", 9: "%", 10: "~", 11: "*", 12: "O", 13: "&", 14: "-", 15: ":"}
                lines = []
                for y in range(0, 64, 2):
                    row = ""
                    for x in range(0, 64, 2):
                        c = int(arr[y, x])
                        row += char_map.get(c, str(c % 10))
                    lines.append(row)
                ascii_art = "\n".join(lines)
            text_msg = f"{user_msg}\n\nGrid preview:\n```\n{ascii_art}\n```"
            messages = [{"role": "user", "content": text_msg}]

        response = self._call_llm(messages, temperature=0.2)
        plan = ReasoningPlan(raw_response=response)
        plan.parse()
        return plan

    def reason_step(self, tracker: StateTracker, last_action: str,
                    last_diff_summary: str) -> ReasoningPlan:
        """Reason about the next step based on observed changes."""
        snap = tracker.last_snapshot
        if snap is None:
            return ReasoningPlan(raw_response="No frame data")

        user_msg = self.adaptive_prompt.format(
            last_action=last_action,
            diff_summary=last_diff_summary,
            state=snap.state.name,
            action_count=tracker.action_count,
            levels_completed=snap.levels_completed,
            win_levels=snap.win_levels,
            llm_context=tracker.build_llm_context(max_tokens=1000),
        )

        # Keep conversation manageable (last 10 turns)
        history = self.conversation_history[-8:]
        history.append({"role": "user", "content": user_msg})

        response = self._call_llm(history)
        self.conversation_history.append({"role": "user", "content": user_msg})
        self.conversation_history.append({"role": "assistant", "content": response})

        plan = ReasoningPlan(raw_response=response)
        plan.parse()
        return plan

    def parse_next_action(self, plan: ReasoningPlan, available_actions: list[int]) -> Optional[tuple]:
        """Parse the LLM's next action into (GameAction, data_dict) or None.

        Returns: (GameAction, {"x": int, "y": int}) or (GameAction, {})
        """
        if plan.next_action is None:
            return None

        next_str = plan.next_action.strip()

        # Try pattern: ACTION6(x=15,y=20)
        match = re.match(r'(ACTION\d+)\s*\(\s*x\s*=\s*(\d+)\s*,\s*y\s*=\s*(\d+)\s*\)', next_str)
        if match:
            action_name = match.group(1)
            x, y = int(match.group(2)), int(match.group(3))
            try:
                from arcengine import GameAction
                return (GameAction[action_name], {'x': x, 'y': y})
            except KeyError:
                return None

        # Try simple action name: ACTION1
        match = re.match(r'(ACTION\d+)', next_str)
        if match:
            action_name = match.group(1)
            try:
                from arcengine import GameAction
                return (GameAction[action_name], {})
            except KeyError:
                return None

        # Try just a number
        match = re.match(r'(\d+)', next_str)
        if match:
            action_num = int(match.group(1))
            if action_num in available_actions:
                try:
                    from arcengine import GameAction
                    return (GameAction[f"ACTION{action_num}"], {})
                except KeyError:
                    pass

        logger.warning(f"Could not parse action: {next_str}")
        return None
