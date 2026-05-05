# HermesAgent — ARC-AGI-3 Competition Entry

LLM-powered grid puzzle solver for the [ARC-AGI-3 Competition](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3) ($850,000 prize pool).

## Architecture

```
EXPLORE → UNDERSTAND → PLAN → EXECUTE → ADAPT → REPEAT
```

| Module | File | Purpose |
|--------|------|---------|
| **Grid Parser** | `grid_parser.py` | Multi-layer `[layers][64][64]` grid → connected components, object tracking, frame diffs, ASCII art |
| **Explorer** | `explorer.py` | Systematic action probing, effect classification (move/click/toggle/spawn) |
| **LLM Reasoner** | `llm_reasoner.py` | Structured reasoning via glm-5.1 (NVIDIA NIM) — parses REASONING/UNDERSTANDING/NEXT/CONFIDENCE |
| **Adaptive Executor** | `executor.py` | Orchestrates the full loop, handles multi-level play, anti-stuck, solution reuse |
| **HermesAgent** | `hermes_agent.py` (in agents/templates/) | Official Agent subclass, integrates all modules |

## Quick Start

```bash
# Clone and setup
git clone https://github.com/N-cryptd/arc-agi-3-hermes.git
cd arc-agi-3-hermes
git submodule update --init  # ARC-AGI-3-Agents framework

# Setup venv
cd ARC-AGI-3-Agents
uv sync

# Set API keys
cp .env.example .env
# Edit .env with your ARC_API_KEY and NVIDIA_NIM_API_KEY

# Run on all games
uv run main.py --agent=hermesagent

# Run on specific game
uv run main.py --agent=hermesagent --game=ar25
```

## Competition Setup

### Kaggle Notebook
1. Create a new Kaggle Notebook with GPU accelerator (T4x2)
2. Add competition data: `arc-prize-2026-arc-agi-3`
3. Upload HermesAgent files as a Kaggle Dataset (or use this repo)
4. Set secrets: `ARC_API_KEY`, `NVIDIA_NIM_API_KEY`
5. Run the notebook

### API Keys Needed
- **ARC API Key**: https://three.arcprize.org/ (for game environment access)
- **NVIDIA NIM API Key**: https://build.nvidia.com/ (for glm-5.1 LLM reasoning)
- **OpenRouter Key** (optional fallback): https://openrouter.ai/

## Game Insights (25 Public Games)

| Type | Count | Actions | Strategy |
|------|-------|---------|----------|
| keyboard_click | 13 | 3-7 | Movement + click with coordinates |
| keyboard | 4 | 4-5 | Movement only |
| click | 8 | 1-2 | Click on grid objects |

All games use 64×64 grids with 1-9 layers. Baseline solutions range from 6-578 steps.

## Leaderboard Target
- Top score: **0.68** (as of May 2026)
- Random baseline: ~0.0
- **Our target: 0.50+** (top 10)

## Milestones
- [x] Milestone 0: Environment setup, API keys, game exploration
- [ ] Milestone 1: First submission scoring >0.30 (June 30)
- [ ] Milestone 2: Score >0.50, cross-level learning (Aug 30)
- [ ] Milestone 3: Score >0.65, multimodal vision (Oct 15)
- [ ] Final: Optimize for competition deadline (Nov 2, 2026)
