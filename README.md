# RLM

Hybrid Recursive Language Model routing for oversized external context in Agent Zero.

## How Agent Zero Uses RLM Context

This plugin gives Agent Zero a second execution path for turns where the hard part is not the visible conversation itself, but a very large amount of external material attached to that turn.

Agent Zero remains the controller. The normal loop still decides what to do next, which tools to call, and what answer to send. The plugin only steps in when large external payloads would otherwise crowd out the useful visible prompt.

In practice, the plugin:

1. Measures prompt pressure for the active turn.
2. Finds large eligible external blocks such as attachment text or oversized tool output.
3. Replaces those blocks with placeholders in the visible prompt.
4. Passes the removed blocks into upstream RLM as structured context.
5. Lets RLM inspect, decompose, and recursively call itself over that context.
6. Returns a normal assistant answer or exact JSON tool call back into the Agent Zero loop.

## Automatic Routing In Practice

Automatic routing is meant to feel mostly invisible to the user.

Good examples:

- A user pastes a huge log dump and asks for the root cause.
- A turn includes many markdown notes or design documents and asks for synthesis.
- A tool returns a large payload that needs cross-block comparison.
- A question requires evidence scattered across many long blocks.

Automatic routing is skipped when:

- The visible prompt is still comfortably small.
- No eligible large external blocks are available to offload.
- Offloading would not reduce pressure enough to help.
- The runtime, provider mapping, or dependency state is not ready.

## Manual Tool In Practice

The `rlm_context` tool is the explicit path for recursive long-context analysis.

Use it when the agent wants to say, in effect:

- "Take the recent history and inspect it with recursive long-context reasoning."
- "Analyze this oversized payload deliberately, even if auto-routing would not have triggered."

The manual tool still returns a normal Agent Zero response. It is not meant to expose raw RLM internals as the final user-visible answer.

## What Users Notice

Users should notice four things:

1. Long external payloads stop overwhelming the normal prompt.
2. The agent can still answer based on large evidence sets without switching to a different workflow.
3. The explorer shows when RLM was used, what happened during the run, and what final answer came back.
4. Dependency, provider, and environment readiness are visible before users trust the feature.

## How To Explain It In Agent Zero

The simplest accurate explanation is:

> Agent Zero uses RLM when a turn contains more external context than should fit inline. It offloads the bulky material, lets RLM reason over it recursively, and then brings the result back as a normal assistant answer.

That framing is important because this plugin is not trying to replace the main model for every turn. It is a selective long-context worker inside the broader Agent Zero framework.

## Key UI Surfaces

- `webui/config.html`: settings and first-use guidance
- `webui/main.html`: readiness, run explorer, and trajectory inspection
- `api/status.py`: dependency and practical readiness summary
- `tools/rlm_context.py`: manual explicit analysis path

## Operational Notes

- The plugin uses the configured main chat model for the root RLM call when mapping is supported.
- It prefers the configured utility model for recursive subcalls when available.
- It can run in `local` or `docker` mode, with `auto` preferring Docker only when usable.
- If RLM is unavailable or unhealthy, Agent Zero falls back to the normal model path.
