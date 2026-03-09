# Tracer — Product & Development Plan

Tracer is an open-source local-first developer tool for AI agents. Its purpose is to record agent runs, reconstruct what the model actually saw, and replay failures deterministically. The opportunity is not "another observability dashboard." The opportunity is a debugging primitive for non-deterministic agent systems, where existing tools still leave developers unable to reconstruct effective prompts or reliably replay failures. That specific gap shows up repeatedly in developer complaints and in the competitive analysis.

---

## 1. Product thesis

Tracer should be built around a single belief: **AI agent debugging is a deterministic reproduction problem, not a dashboard problem.** Existing tools already cover logging, token tracking, and trace dashboards well enough that competing there as a solo founder would be a bad strategy. What remains underbuilt is reproducing an agent failure exactly enough that a developer can inspect it, understand it, and fix it.

**The product metaphor is: Crash dumps for AI agents.**

Short description: *Record agent runs. Inspect decision steps. Replay failures locally.*

This means Tracer is not trying to be Langfuse, Phoenix, LangSmith, or AgentOps. Those products already cover broad observability surfaces. Tracer starts one layer lower and narrower: capture the actual execution state needed to debug a failure.

---

## 2. Core product definition

Tracer produces a portable artifact called an **`.rpack`** file.

An `.rpack` is the debugging artifact for an agent run. It should contain enough information to answer:

- What exact prompt/context did the model see at this step?
- What tool calls happened, with what inputs and outputs?
- What model parameters were used?
- What was returned?
- Can I replay this failure without hitting the provider again?

For v0.1, the MVP should only support **recording, inspection, and replay**. Dashboards, cloud storage, hosted collaboration, analytics, metrics, and team workflows are explicitly out of scope.

---

## 3. Product principles

These principles apply to all development work.

- **Small surface area.** Tracer should remain a narrow tool until developers clearly adopt the artifact format. No platform creep.
- **Local first.** Everything in v0.1 should run on a developer machine with no servers, databases, or background services. The MVP spec explicitly calls for a local JSON-backed artifact and CLI-only workflow.
- **Replay fidelity over breadth.** The most important thing is capturing the exact model input and returning the exact recorded output. Missing fidelity makes replay misleading, which is worse than having no replay at all. Both the market research and MVP spec emphasize that exact prompt reconstruction is the differentiator.
- **Provider-agnostic artifact schema.** The system may start with an OpenAI adapter, but `.rpack` is not OpenAI-specific. There is one artifact format and many adapters.
- **Adapters are not the product.** The product is the artifact and the replay/debugging workflow. Providers and frameworks are just capture mechanisms.
- **Code must feel human-written.** The repository will be judged by experienced engineers. Avoid generic AI scaffolding, bloated abstractions, and speculative architecture.

---

## 4. What the market actually needs

The most repeated pain points from developer research are:

- No step-by-step visibility into multi-step agent runs
- Silent failures where the agent returns nonsense without an error
- Difficulty reconstructing the exact context window at the failure point
- Difficulty replaying a bad run deterministically
- Tool call and context compaction issues
- Cost blind spots in loops and long runs

The competitive analysis shows that broad logging and trace capture are already mature. The underbuilt wedge is **replay-first debugging** centered on effective prompt reconstruction and portable replay artifacts.

This means Tracer should not try to beat incumbents on dashboards. It should beat them on one thing: **turning a bad run into a faithful, inspectable, replayable artifact.**

---

## 5. Naming and positioning

- **Tool name:** Tracer
- **Artifact extension:** `.rpack`

Use this language consistently:

- "Crash dumps for AI agents"
- "Replay-first debugging for agent failures"
- "Record, inspect, and replay agent runs locally"

Do not over-position Tracer as observability, monitoring, or AgentOps. That pulls the project into the wrong market narrative too early.

---

## 6. Version 0.1 scope

Version 0.1 should implement exactly this:

### Required

- A CLI called `tracer`
- Record a Python script execution and intercept supported provider calls
- Write a single `.rpack` artifact to disk
- Open and inspect an `.rpack`
- Replay a step deterministically without calling the provider
- Capture exact prompt/messages payload, model, parameters, response text, timestamps
- Store tool calls if they are present and interceptable
- Keep everything local

### Explicitly excluded

- Web UI
- Team features
- Cloud sync
- Search across runs
- Analytics dashboards
- Metrics aggregation
- Hosted platform
- Long-term storage infrastructure
- Multi-provider support beyond the first adapter, unless trivial to add cleanly

---

## 7. Architecture

Tracer must enforce three strict layers.

### Layer 1: Provider adapter layer

Adapters intercept provider SDK calls and normalize them into the generic schema.

v0.1 can start with one official adapter for the OpenAI Python SDK because it is the fastest path to a usable prototype, not because the product is OpenAI-specific. The artifact schema remains provider-neutral. Future providers are added by writing new adapters, not by changing replay logic.

Adapters should live in a structure like:

```
tracer/adapters/openai_adapter.py
```

Responsibilities:

- Capture exact request payload as sent
- Capture model and parameters
- Capture response text and metadata
- Normalize into schema events

### Layer 2: Provider-agnostic artifact schema

`.rpack` is the core product primitive. The schema must stay stable and provider-neutral.

The schema should represent concepts like:

- `run_id`
- `provider`
- `model`
- `input.messages`
- `parameters`
- `output.text`
- `timestamps`
- `tool calls`
- `final_output`

It should not mirror any provider's raw request or response structures directly. Those should be normalized by adapters before they enter the artifact.

### Layer 3: Replay engine

The replay engine reads `.rpack` and returns stored outputs. It must never import or depend on any provider SDK. Replay is deterministic by definition. It is not a re-query feature.

---

## 8. Effective prompt reconstruction

This is the most important technical and product requirement.

Tracer must capture the **effective prompt** at the moment of the model call. That means whatever the provider actually received, after any prompt assembly, context compaction, summarization, or tool schema injection.

This matters because developers care about what the model actually saw, not the fragmented history that led up to it. The research specifically identifies retrospective effective prompt reconstruction as an underbuilt need and a promising wedge.

In practice, the final request payload should be treated as the source of truth and persisted exactly.

---

## 9. Replay philosophy

Replay in v0.1 is intentionally simple:

1. Read `.rpack`
2. Select step N
3. Show exact input/output
4. Return recorded output instead of calling any model

This is enough to make the artifact useful.

Later, replay can expand to:

- Stubbed tool calls
- "Rerun this step with modified prompt" mode
- Replay packs attached to CI or bug reports
- Diffing between runs

But those are not v0.1.

---

## 10. CLI design

The CLI is the primary UX. The MVP research was clear that CLI-first is the lowest-complexity and highest-leverage interface for the first release.

### Required commands

```
tracer record my_agent.py
```
Runs the target script with recording enabled and writes an `.rpack`.

```
tracer open run.rpack
```
Prints a readable summary of the run and each step.

```
tracer replay run.rpack --step N
```
Shows the exact input/output for that step and returns the stored result.

### Optional but likely useful

```
tracer validate run.rpack
```
Checks schema integrity.

Output should be readable, concise, and deterministic.

---

## 11. Artifact design

Use `.rpack` as the file extension.

For v0.1, `.rpack` can simply contain JSON content written directly to disk. Keep it obvious and inspectable. A developer should be able to open the file and understand it.

A reasonable schema shape:

```json
{
  "run_id": "trace_2026-03-10T21:14:03Z",
  "provider": "openai",
  "steps": [
    {
      "step": 1,
      "model": "gpt-4.1",
      "input": {
        "messages": [
          {"role": "system", "content": "..."},
          {"role": "user", "content": "..."}
        ]
      },
      "parameters": {
        "temperature": 0.2
      },
      "output": {
        "text": "..."
      },
      "tool_calls": [],
      "timestamp": "2026-03-10T21:14:04Z"
    }
  ],
  "final_output": "..."
}
```

Do not overcomplicate v0.1 with compression or bundle formats. If the product grows, `.rpack` can later evolve into a container while keeping the extension stable.

---

## 12. Initial integrations

For version 0.1, the initial integrations should be deliberately narrow.

### Must have

- OpenAI Python SDK interception

### Optional if clean and fast

- Basic Anthropic adapter
- Minimal LangChain callback support

The MVP research specifically concluded that focusing on OpenAI and optionally LangChain is the right leverage point, while trying to support every framework or a generic proxy too early would be a mistake.

---

## 13. Current implementation status

The first implementation already produced:

- Repo layout
- Schema dataclasses
- Recorder
- Replay engine
- CLI
- OpenAI adapter
- Example
- Tests
- Zero runtime dependencies in the core
- 13 tests passing

This is a strong base. The next work should not be a rewrite. It should be focused refinement.

---

## 14. Immediate refinement plan

The following priorities should be executed in order using the current implementation as the base.

### Priority 1: Verify provider interception coverage

Ensure the adapter intercepts all relevant OpenAI SDK paths used by real-world scripts, not just one code path. Modern SDKs may use different entry points.

### Priority 2: Preserve exact structured input

Do not flatten messages into a single string if the SDK call included structured messages. The artifact must preserve the full structured payload.

### Priority 3: Tool call capture

If tool/function calling is present in the intercepted response or request, record it explicitly in the schema rather than burying it in raw text.

### Priority 4: Safer artifact naming

Default output should not always be `run.rpack`. Use timestamped names like `trace_2026-03-10_21-14-03.rpack`.

### Priority 5: Improve open output readability

`tracer open` should format the run in a way that feels immediately useful to a developer, without being noisy.

### Priority 6: Real-world smoke testing

Run the tool against:

- A simple OpenAI script
- A LangChain flow if possible
- A multi-step tool-calling example

The goal is to confirm actual usefulness, not just passing unit tests.

---

## 15. Development roadmap

### Phase 1: Harden v0.1

**Goal:** Make the existing MVP reliable enough for a public GitHub release.

Tasks:

- Verify adapter coverage
- Improve artifact naming
- Preserve exact payload structure
- Improve CLI ergonomics
- Clean code quality and docs
- Confirm real-world recording/replay flows

### Phase 2: Public OSS release

**Goal:** Put Tracer on GitHub with a strong README and examples.

Must include:

- README with the "crash dumps for AI agents" framing
- Quickstart install
- One minimal example
- One realistic example
- Clear CLI usage
- A sample `.rpack`

### Phase 3: Developer feedback cycle

**Goal:** Learn whether developers actually use `.rpack` in their debugging workflow.

Track:

- GitHub stars
- Issues and PRs
- Installs
- Whether people attach `.rpack` files to bug reports
- What they complain is missing

### Phase 4: Narrow expansion

Only after real adoption, consider adding:

- Replay diffing
- Tool-output stubbing
- Replay pack validation
- "Rerun this step with modified prompt" mode
- Better framework adapters

---

## 16. What not to do

This is important.

Tracer should not:

- Turn into a broad observability platform
- Add dashboards
- Add networking or cloud services
- Introduce unnecessary abstractions
- Create a "platform" architecture before the artifact is validated
- Optimize for enterprise features yet
- Chase feature parity with Langfuse, Phoenix, or LangSmith

**Tracer wins by being the cleanest path from agent failure → reproducible artifact → deterministic inspection.**

---

## 17. Open-source distribution plan

GitHub is the distribution engine.

The repo should be structured simply and read well. The README should explain the value in one screen.

Suggested repo layout:

```
pyproject.toml
README.md
tracer/
examples/
tests/
```

README opening:

> **Tracer**
> Crash dumps for AI agents.

Then immediately:

> - Record agent runs
> - Inspect decision steps
> - Replay failures locally

The project should be announced where agent developers already spend time:

- Hacker News
- GitHub
- LangChain / agent communities
- Relevant subreddits and dev communities when appropriate

---

## 18. Monetization direction

Do not optimize for monetization now.

The intended model remains:

- Free open-source core
- Later hosted or team features
- Later enterprise features if adoption warrants it

But that is downstream. Right now the only thing that matters is whether `.rpack` becomes useful enough that developers reach for it during debugging.

If `.rpack` becomes a standard reproducibility artifact, monetization opportunities naturally emerge around storage, collaboration, CI, and governance. The defensible wedge is the replay/debug artifact, not broad tracing or dashboards.

---

## 19. Success criteria

Development should optimize around these early success signals:

- Developers can install and use Tracer in under a few minutes
- `.rpack` files are created reliably
- `tracer open` makes failures easier to understand
- `tracer replay` is deterministic and useful
- At least some developers choose to share `.rpack` in issues or debugging conversations

Those behaviors matter more than vanity features.

---

## 20. Development directive

Build Tracer as a narrow, local-first, replay-first debugging tool for AI agents. Preserve the three-layer architecture: provider adapters, provider-agnostic `.rpack` schema, and replay engine. Optimize for exact prompt fidelity, deterministic replay, and low developer friction. Do not broaden scope into observability or platform features. Use the current implementation as the base and focus next on reliability, real-world compatibility, artifact quality, and public-release readiness.
