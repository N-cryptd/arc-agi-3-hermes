# ARC-AGI-3 Competition Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Build a competitive ARC-AGI-3 agent and submit for Milestone #1 (June 30) and the final competition (Nov 2) to win prize money from the $850K pool.

**Architecture:** Tiered hybrid agent — Vision LLM for observation & reasoning, heuristic policies for execution, cross-level meta-learning for generalization.

**Tech Stack:** Python 3.10+, arc-agi SDK, OpenRouter API (vision LLMs), Kaggle notebooks (GPU), PyTorch (optional DRL components), GitHub (open-source)

---

## Competition Rules Summary

- **Scoring:** RHAE = (human_actions / ai_actions)^2 per level, weighted by level index (1-indexed)
- **Actions:** RESET, ACTION1-5 (simple), ACTION6 (x,y coords 0-63), ACTION7 (undo)
- **Grid:** 64x64, integer values 0-15
- **Competition mode:** API only, single interaction per env, level resets only, all envs scored
- **Environments:** 25 public + 55 semi-private + 55 fully private
- **Rate limit:** 600 RPM (free)
- **Open source required** for prize eligibility

## Key Deadlines

| Date | Milestone |
|------|-----------|
| **June 30, 2026** | Milestone #1 — 1st: $25K, 2nd: $10K, 3rd: $2.5K |
| **Sep 30, 2026** | Milestone #2 — 1st: $25K, 2nd: $10K, 3rd: $2.5K |
| **Oct 26, 2026** | Entry deadline |
| **Nov 2, 2026** | Final submission deadline |
| **Dec 4, 2026** | Winners announcement |

## Current State

- Best AI score: 0.37% (Gemini 3.1 Pro)
- Human score: 100%
- 541 teams competing
- No one has solved even a single private environment meaningfully

---

## Agent Architecture

```
┌─────────────────────────────────────────┐
│            ARC-AGI-3 Agent              │
├─────────────────────────────────────────┤
│  1. Observer                            │
│     - Parse 64x64 grid from FrameData   │
│     - Track state changes between steps │
│     - Extract objects & spatial patterns│
├─────────────────────────────────────────┤
│  2. Explorer                            │
│     - Systematic action probing         │
│     - Map action→effect relationships   │
│     - Detect game-over & win states     │
│     - Build action model per game       │
├─────────────────────────────────────────┤
│  3. Reasoner (Vision LLM)              │
│     - Analyze observed action effects   │
│     - Hypothesize game rules/goals      │
│     - Plan multi-step strategies        │
│     - Update world model                │
├─────────────────────────────────────────┤
│  4. Executor (Adaptive Policy)          │
│     - Convert plans to action sequences │
│     - Monitor execution & adapt         │
│     - Optimize for action efficiency    │
│     - Handle game-over recovery         │
├─────────────────────────────────────────┤
│  5. Meta-Learner (Cross-Level)         │
│     - Transfer knowledge between levels │
│     - Detect game archetype changes     │
│     - Adjust strategy per level         │
│     - Archive successful patterns       │
└─────────────────────────────────────────┘
```

## Strategic Approach: "Explore-Understand-Execute" Loop

### Phase 1: Exploration (first ~10-20% of actions per level)
- Try each available action once to understand effects
- Track: which actions change the grid, how, and where
- Detect: player avatar position, movable objects, goals, boundaries
- Identify: available_actions changes that signal game phases

### Phase 2: Understanding (LLM reasoning, ~10-20% of actions)
- Feed observed patterns to vision LLM with grid states
- Ask: "What are the rules of this game? What's the goal?"
- Generate hypothesis about win conditions
- Plan sequence of actions to achieve goal

### Phase 3: Execution (remaining ~60-80% of actions)
- Execute planned strategy
- Monitor results — if not progressing, re-enter exploration
- Optimize for minimum actions (RHAE rewards efficiency)
- Use undo (ACTION7) to recover from mistakes

---

## Implementation Plan

### Task 1: Project Setup & Registration (Day 1)
- Clone ARC-AGI-3-Agents repo
- Set up virtual environment with uv
- Get ARC_API_KEY (user action needed)
- Get Kaggle API credentials
- Create GitHub repo for open-source agent
- Verify random agent works on ls20, ft09, vc33
- **Files:** `~/arc-agi-3-hermes/`, `.env`, `README.md`

### Task 2: Environment Exploration & Documentation (Days 2-3)
- Play all 25 public environments manually (terminal mode)
- Document each game: actions available, apparent rules, difficulty
- Categorize games by type (puzzle, navigation, logic, orchestration...)
- Identify common patterns across games
- Create game encyclopedia for agent knowledge base
- **Files:** `docs/game_encyclopedia.md`, `docs/patterns.md`

### Task 3: Grid Parser & State Tracker (Days 3-5)
- Build robust FrameData parser
- Implement state diff tracker (what changed between frames)
- Object detection: connected components, color groups
- Player/avatar tracking (which cells changed due to action)
- Game state classifier: NOT_PLAYED, PLAYING, WIN, GAME_OVER
- **Files:** `src/observer/grid_parser.py`, `src/observer/state_tracker.py`, `src/observer/object_detector.py`
- **Tests:** `tests/test_grid_parser.py`, `tests/test_state_tracker.py`

### Task 4: Exploration Module (Days 5-7)
- Systematic action probing strategy
- Action effect cataloger: map ACTION_N → grid delta
- Goal detector: recognize win/loss patterns from frame data
- Exploration budget manager: allocate actions to exploration vs execution
- **Files:** `src/explorer/action_prober.py`, `src/explorer/effect_catalog.py`, `src/explorer/goal_detector.py`

### Task 5: LLM Reasoner (Days 7-12)
- OpenRouter integration (vision models)
- Frame→prompt converter: grid state to visual/text representation
- Multi-turn reasoning with conversation history
- Rule hypothesis generator
- Strategy planner (action sequence generator)
- Fallback: if LLM unavailable, use heuristic rules
- **Files:** `src/reasoner/llm_client.py`, `src/reasoner/prompt_builder.py`, `src/reasoner/rule_hypothesizer.py`, `src/reasoner/strategy_planner.py`

### Task 6: Adaptive Executor (Days 12-15)
- Plan-to-actions converter
- Action efficiency optimizer (minimize wasted actions)
- Progress monitor (are we getting closer to win?)
- Recovery strategies: undo, re-explore, switch approach
- Timeout management (max actions per level)
- **Files:** `src/executor/plan_runner.py`, `src/executor/efficiency_optimizer.py`, `src/executor/recovery.py`

### Task 7: Meta-Learner (Days 15-18)
- Cross-level knowledge transfer
- Game archetype detector (puzzle vs maze vs logic...)
- Level difficulty estimator
- Strategy adaptation per level
- Pattern archive of successful approaches
- **Files:** `src/meta/archetype_detector.py`, `src/meta/knowledge_transfer.py`, `src/meta/strategy_adapter.py`

### Task 8: Main Agent Integration (Days 18-20)
- Wire all modules into single Agent class
- Implement Agent interface (is_done, choose_action)
- State machine: EXPLORE → UNDERSTAND → EXECUTE → ADAPT
- Comprehensive logging for replays
- Error handling and robustness
- **Files:** `agents/hermes_agent.py`

### Task 9: Benchmarking & Optimization (Days 20-25)
- Benchmark on all 25 public environments
- Analyze failure modes per game
- Iterate on exploration strategy
- Tune LLM prompts for better reasoning
- Optimize action efficiency (critical for RHAE)
- Compare against baseline (random, LLM-only, heuristic-only)
- **Files:** `scripts/benchmark.py`, `scripts/analyze_results.py`, `configs/agent_config.yaml`

### Task 10: Milestone #1 Preparation (Days 25-30)
- Open-source repository on GitHub (MIT license)
- Write comprehensive README with approach documentation
- Prepare Kaggle notebook for submission
- Final testing on public environments
- Submit for Milestone #1 (June 30)
- **Files:** `kaggle/submission.ipynb`, `README.md`

### Task 11: Post-Milestone Development (July-Sep)
- Analyze Milestone #1 results and other teams' approaches
- Improve based on leaderboard feedback
- Add more sophisticated strategies
- Prepare for Milestone #2 (Sep 30)
- Continue development for final submission (Nov 2)

---

## Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| LLM API costs exceed budget | Medium | High | Use cheap models (Gemini Flash, GPT-4o-mini) for exploration; save expensive models for reasoning |
| Agent doesn't generalize to private envs | High | High | Focus on general exploration strategy, not game-specific tricks |
| Action efficiency too low | Medium | High | Heavy emphasis on RHAE optimization; early action counting |
| Rate limiting in competition | Low | Medium | Implement backoff; cache exploration results |
| Kaggle notebook memory/GPU limits | Medium | Medium | Keep model lightweight; use ONNX quantization if needed |
| Other teams outpace us | Medium | Medium | Focus on Milestone prizes (guaranteed money); even modest performance could place |

## Resource Requirements

- **Compute:** Kaggle GPU (T4 or P100) for notebook inference
- **API:** OpenRouter credits for vision LLM calls (~$50-100/month estimated)
- **ARC API:** Free (600 RPM)
- **Storage:** GitHub free tier for open-source repo
- **Time:** ~2-3 hours/day of Hermes autonomous development

## Hermes Management Plan

1. **Daily development cron:** Run agent development tasks automatically
2. **Weekly benchmark:** Test on all public games, track scores
3. **Progress dashboard:** Track which tasks are done, scores per game
4. **Auto-commit:** Push improvements to GitHub daily
5. **Milestone alerts:** Notify user before submission deadlines

## User Actions Needed

1. ✅ Register on [arcprize.org/platform](https://arcprize.org/platform) → Get ARC_API_KEY
2. ✅ Register on [kaggle.com](https://kaggle.com) → Accept competition rules
3. ✅ Provide OpenRouter API key (or other LLM provider key)
4. ✅ Create GitHub repo for open-source submission
5. ✅ (Optional) Set up Kaggle API credentials for notebook submission
