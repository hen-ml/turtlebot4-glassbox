# Laptop harness — LLM mission compiler, robot as tool calls

_Status: design agreed 2026-11-22 (branch `arena/9611c2b5-turtlebot4-glassbox`). Phase 1 code lives in `harness/`._

## TL;DR

A laptop (e.g. an ASUS Zenbook with a Snapdragon X 16 GB) acts as an **agent
harness**: it turns a free-text mission into a validated, ordered plan of tool
calls, hands the plan to the robot, verifies each leg, re-plans **once** on
failure, and stays out of the way otherwise. The robot executes with its own
fast, embedded reflexes — nothing about its motion changes. The LLM never sits
in the control loop, never blocks a drive cycle, and is never load-bearing for
safety.

This is deliberately **not** a new robot brain and **not** a learned policy.
It is the pattern the pragmatic part of the field converged on after the
research lineage below: **LLM-as-mission-compiler over classical skills**, with
a human gate once per plan.

## Why this, why now

The repo's behaviors are classical planning (A* + pure pursuit) sequenced by a
small DSL (`toyscript.py`), triggered today by a keyword matcher
(`match_program`). That matcher is fine for triggers the authors anticipated
and brittle for anything else. An LLM compiled plan removes that brittleness
without adding a single dependency to the robot:

- Pi: **zero** new pip packages; the stock HTTP API is all the harness uses.
- Laptop: one local model server (LM Studio — see below) + this repo's
  `harness/` package, also stdlib-only.
- No cloud account, no API keys, no map, no persistent state.

## Architecture

```
[ Laptop ]                                   [ TurtleBot 4 — unchanged ]
 mission text
     │
     ▼
 LLM compile (local, JSON mode)               ┌─ POST /run {"program": name}
     │  strict schema, few-shot               ├─ POST /goto_xy {"x","y","mode"}
     ▼                                        ├─ POST /run_stop
 validated plan ──human confirm──► executor ─►│  POST /dock, /undock
     │                                        └─ GET  /state   (poll)
     ▼                                                    │
 leg-boundary verifier ◄───────── run_active, cmd_history, status, nav_active
     │ ok → next leg                  (the existing glass-box telemetry)
     │ fail → ONE re-plan with fresh state → retry
     │ fail again → STOP + report
```

Deliberate rejections, recorded so we don't re-litigate:

- **LLM in the control loop / per-tick control** — no. Latency, cost, and a
  stopped model must never stop a safe robot. Plan once per mission, verify at
  leg boundaries.
- **LFG / frontier heuristics** — dropped by decision (no persistent map, and
  map-less robot is a hard constraint).
- **Phone as controller or LLM host** — dropped by decision; simpler with the
  computer browser UI + local LLM on the same machine.
- **LLM on the Pi** — Pi 4 can't run a useful model. No.
- **Persistent occupancy/topological map** — not required for tool-call
  missions; may come back later only if exploration over large spaces hurts.

## Safety model

1. **Once-per-plan human gate.** The compiled plan is printed (with the model's
   rationale — the glass-box rule applies to the harness too); nothing moves
   until the human taps yes. After that, no mid-mission intervention in the
   happy path.
2. **Allowlist, never free-form.** Only these leg kinds exist:
   `run_program` (name validated against `programs/*.toy`), `goto` (clamped
   ±30 m, mode `move|push`), `dock`, `undock`, `wait` (≤ 600 s, sleeps on the
   harness side). Unknown verbs/out-of-range args are rejected before any
   motor moves.
3. **Bounded re-plan.** At most one re-plan per failure, and it receives the
   unmodified mission, the completed legs, the fresh `/state`, and the error.
   A second failure stops the robot and reports.
4. **Hard stop is always local.** `/run_stop`, the dock button, the wall
   socket — none of these involve the LLM or the harness. Ctrl-C in the
   harness calls `run_stop` too.
5. **Budgets.** Mission deadline (default 15 min), per-leg timeout, and a step
   budget (≤ 12 legs) cap everything the LLM can summon.
6. **Robot fails safe on laptop loss.** Worst case: the laptop sleeps/dies →
   current leg finishes (skills self-abort on their own timeouts) → next
   checkpoint never happens. The robot does not wander off to find you.
   Keep the laptop on AC and awake for the mission.

## Where robustness actually comes from

Not from the LLM. From three things that already exist or are trivial to add:

1. The skills' internal re-fix loops (ball-push re-aims every round; follow
   re-detects every cycle; scan-sweep before approach).
2. Leg-boundary verification (this harness) — cheap, deterministic, and it
   catches "the ball is gone" before the next leg compounds the error.
3. The human gate at the start and the hard stop at the end.

The LLM contributes **coverage** (missions you didn't hand-write a trigger
for) and **composition** (chain existing behaviors), nothing more. Treat any
bigger claim with suspicion.

## Local model recommendation for the Zenbook (Snapdragon X, 16 GB)

Install **LM Studio** (Windows-on-ARM build, no WSL involved):

- Model: **Qwen2.5-7B-Instruct Q4_K_M** (or Llama-3.1-8B-Instruct Q4_K_M).
  16 GB RAM runs these comfortably alongside a browser.
- Start the local server (default `http://localhost:1234/v1`). The harness
  speaks the OpenAI `/v1/chat/completions` shape, so pointing it at Ollama,
  llama.cpp server, or a Mac later is just an env var change.
- Set `TB4_LLM_URL=http://localhost:1234/v1` and
  `TB4_LLM_MODEL=<whatever id LM Studio shows>`.
- Expect modest token rates (the Snapdragon NPU is not usable by these
  runtimes; llama.cpp's ARM path on Oryon is what you get). Irrelevant here:
  one structured completion per mission, plus rare re-plans.
- Headless alternative: **Ollama via WSL2** (ARM64 Linux) — fine, but LM
  Studio's GUI is the lower-friction start and equally good behind the same
  OpenAI-compatible endpoint.
- Phase 3 (optional) adds a **vision** model (Qwen2.5-VL-7B Q4) for
  non-COCO landmarks. That is the one place the Zenbook will feel slow —
  seconds per image, once per leg. YOLO stays the primary grounding source.

## Phases

1. **Phase 1 — mission compiler (this repo now).** `harness/` turns mission
   text into a validated plan, prints it, runs it against the Pi's existing
   HTTP API, verifies legs, re-plans once, stops. No Pi changes at all.
2. **Phase 2 — richer legs.** Compose existing programs into multi-leg
   missions (`SCAN_FOR` → approach → `PUSH_TO_WALL` …) and, if needed, one or
   two tiny Pi endpoints only (e.g. structured skill-result query). Still no
   map, still no LLM on the robot.
3. **Phase 3 — open-vocabulary landmarks.** Harness fetches `/camera.jpg`,
   asks the local VLM "is X visible, which side?", uses it to bias a scan leg.
   COCO stays the default path.
4. **Phase 4 (optional) — robot-as-MCP-server.** Wrap the same HTTP API as a
   Model Context Protocol tool server so any MCP-capable agent (Claude
   Desktop, a second robot's harness, MicroDuck's) can drive the TurtleBot.

## MicroDuck / future robots

The harness is split into a **robot-agnostic core** (compile → confirm →
execute → verify → re-plan) and a **robot adapter** (`harness/robot_api.py`
for the TurtleBot 4; a `MicroDuckAdapter` later maps the same logical legs
onto that robot's API). The plan schema and the safety model are
robot-independent; only the transport and the skill catalog differ.

## Relation to the literature — is this "after LM-Nav"?

Honest positioning, because it shapes what we claim:

- **LM-Nav (CoRL 2022, Shah/Osiński/Ichter/Levine)** — LLM parses an
  instruction into landmarks → VLM grounds them → a *learned* visuomotor
  policy (ViNG) drives between them. Zero-shot, but rests on a navigation
  model trained on unlabeled robot video. We have no such model, and don't
  need one: our skills are explicit and verified.
- **LFG (CoRL 2023, Shah/Equi/Osiński/Xia/Ichter/Levine)** — LLM as a
  heuristic for frontier-based exploration. Requires a map; dropped here.
- **SayCan (2022, Google)** — the real ancestor of "robot as tool calls":
  LLM proposes skills, affordance functions ground feasibility. Our
  allowlist + pre-flight `docked`/battery checks are a deterministic,
  repo-native echo of affordance grounding.
- **Code as Policies (2022/23, Google)** — LLM emits code that calls robot
  APIs. Our `.toy` programs are exactly that, and the harness is the
  higher-level compiler one level up.
- **ReAct (2022, Princeton/Google)** — interleaved reasoning/acting. We
  deliberately use a *bounded* version: plan up front, verify at boundaries,
  one re-plan — not an open-ended reasoning loop.
- **ECoT (2024, Zawalski et al. / Levine)** — chain-of-thought *inside* a
  vision-language-action policy. Different layer entirely (learned
  end-to-end); not applicable to a classical stack and not needed for it.
- **The current mainstream idiom (2024–2026)** — "agent harness + tool server"
  (MCP-style): an LLM orchestrator over narrow, well-typed tools, with humans
  gating mission start. That is what this repo is adopting, minus the
  (unnecessary here) MCP server packaging until Phase 4.

So: not *after* LM-Nav in a timeline sense, but squarely in the lineage the
field converged on for constrained, real robots — and more robust than
LM-Nav's pipeline for this codebase because every leg is a verified skill with
a known failure surface, instead of a learned policy whose failure modes we
can't enumerate. The "modern / advanced" part of this design is not the LLM;
it's the combination of (a) compiled constrained plans, (b) local everything,
(c) leg-boundary verification with a bounded re-plan, and (d) a glass-box UI
that shows the plan and the rationale before and during motion.
