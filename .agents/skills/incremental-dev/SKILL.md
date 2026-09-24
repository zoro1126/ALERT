---
name: incremental-dev
description: >-
  Enforces a strict "one small commit per response" development discipline.
  Activate this skill whenever the user asks to implement, build, add, or code
  anything in this project. The agent must scope each response to a single,
  atomic, committable unit of work — no more. Use this to keep PRs small,
  avoid hitting output token limits, and maintain a clean git history.
---

# Incremental Development Discipline

This project is developed in **small, atomic, git-committable increments**. Every response that produces code must follow this protocol exactly.

---

## Core Rule

**One response = one commit's worth of work.**

A single response must implement **exactly one** of the following:
- One new file (e.g., a single module, class, or script)
- One focused set of changes to an existing file (one feature/fix)
- One configuration or setup artifact (e.g., `requirements.txt`, `config.py`)
- One test file for a previously written module

Never implement more than this in a single response, even if the user asks for multiple things at once.

---

## Response Structure (Follow Every Time)

Every coding response must follow this exact structure:

### 1. Scope Statement (2–3 lines max)

State what this single increment implements and why it's the right next step. Reference the build order from the project plan if applicable.

Format:
```
📦 This increment: [what you're building]
🔗 Depends on: [what must already exist, or "nothing yet"]
✅ After this: [what becomes possible next]
```

### 2. Implementation

Write the code. Keep it focused. A single file or a tightly related pair (e.g., a module + its `__init__.py` update).

Hard limits per response:
- Maximum 1–2 files created or modified
- Maximum ~150 lines of new code (excluding docstrings and comments)
- No placeholder functions — every function written must be complete and working

### 3. Suggested Git Commit Message

End every response with a ready-to-copy commit message following Conventional Commits format:

```
<type>(<scope>): <short description>

<optional body: what and why, not how>
```

Types to use:
- `feat` — new feature or module
- `fix` — bug fix
- `refactor` — restructuring without behavior change
- `test` — adding tests
- `chore` — setup, config, tooling
- `docs` — documentation only

### 4. What Comes Next (1 bullet per next step)

List 2–3 possible next increments so the user can pick the next task.

---

## What to NEVER Do in a Single Response

- Do NOT implement an entire module AND its tests AND wire it into main
- Do NOT create the full project skeleton in one go
- Do NOT write more than ~150 lines of new logic
- Do NOT leave TODO stubs or NotImplementedError placeholders
- Do NOT skip the commit message
- Do NOT implement "the whole training pipeline" because the user asked nicely

If the user asks for something too big, break it down and ask which part to do first.

---

## Handling "Do Everything" Requests

If the user says something like "implement the whole detector" or "set up the full project":

1. Do not comply in full. Acknowledge the request.
2. List the increments it would take (numbered list, one line each).
3. Ask: "Which of these should I start with?" — or start with increment #1 and say so explicitly.

---

## Build Order Reference (ALERT Project)

Use this as the canonical sequence. Each item = one increment.

```
[1]  chore: project skeleton (dirs, .gitignore, requirements.txt, setup.py)
[2]  chore: config.py — central Config class with all hyperparameters
[3]  feat: feature_extractor.py — EAR and MAR from MediaPipe landmarks
[4]  feat: feature_extractor.py — head pose estimation (solvePnP)
[5]  feat: dataset.py — UTA-RLDD video scanner and label parser
[6]  feat: dataset.py — frame extraction and 8-feature vector computation
[7]  feat: dataset.py — sliding window and .npz cache writer
[8]  feat: model.py — BiGRU + Attention architecture (PyTorch)
[9]  feat: trainer.py — training loop with AdamW + cosine scheduler
[10] feat: trainer.py — validation loop + early stopping + metric logging
[11] feat: augment.py — temporal augmentation functions
[12] feat: evaluate.py — confusion matrix + per-class F1 reporting
[13] feat: calibrator.py — adaptive EAR/MAR threshold calibration
[14] feat: classifier.py — sliding window buffer + model inference
[15] feat: alert_manager.py — alert state machine (Warning / Critical)
[16] feat: alert_manager.py — audio alerts via pygame.mixer
[17] feat: overlay.py — OpenCV HUD drawing functions
[18] feat: dashboard.py — main real-time detection loop
[19] feat: run_alert.py — entry point wiring everything together
[20] test: tests/ — unit tests for feature_extractor and model
[21] docs: README.md — final documentation with usage and architecture
```

When asked to start building, begin at the lowest uncompleted increment.

---

## Token Budget Awareness

If you notice your response is getting long (approaching the output limit):
1. Stop at the nearest logical boundary (end of a function or class).
2. Note where you stopped.
3. Say: "Pausing here — this is a clean stopping point. Next response will continue from [X]."
