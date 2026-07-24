# RLM

Recursive long-context analysis for Agent Zero, powered by
[`alexzhang13/rlm`](https://github.com/alexzhang13/rlm).

The plugin keeps Agent Zero in control of the conversation while selectively
offloading oversized external material to an RLM worker. Version 2.0.0 targets
the stable upstream `rlms==0.1.3` package and its `answer["content"]` /
`answer["ready"]` completion contract. A narrow compatibility adapter treats
nullable token-usage values from OpenAI-compatible OAuth proxies as unknown
instead of allowing upstream cost tracking to abort an otherwise valid run.

## What It Does

Automatic routing:

1. Estimates prompt pressure against the active chat model's context length.
2. Finds eligible large message fields and text attachments.
3. Replaces those fields with compact placeholders in the visible prompt.
4. Gives the structured visible context and offloaded blocks to upstream RLM.
5. Returns the completed assistant response or exact Agent Zero tool-call JSON.

Short conversations, prompts without eligible external blocks, and prompts that
do not shrink enough stay on Agent Zero's normal model path. If dependency,
provider, or execution readiness is blocked, automatic routing also falls back
to the normal model without modifying Agent Zero core code.

The `rlm` tool provides an explicit route for recursive analysis over recent
Agent Zero history.

## Installation

Copy or clone this repository to:

```text
usr/plugins/rlm
```

Enable the plugin in Agent Zero. Its install hook automatically installs
`rlms==0.1.3` into the Agent Zero framework Python environment, so there is no
second setup command. In the standard dual-venv Docker image the hook targets
`/opt/venv-a0/bin/python3`; it does not accidentally install only into the
separate agent execution environment.

There is no separate Execute step. Installation, update setup, module cleanup,
and removal cleanup are owned by `hooks.py`. If automatic setup is interrupted,
the Context Explorer exposes **Retry Setup**, which invokes the same install
hook and refreshes readiness.

Restart Agent Zero after upgrading an RLM dependency that was already loaded.

## Execution Safety

`auto` and `docker` modes require Docker. They do not silently downgrade to the
upstream LocalREPL because LocalREPL executes model-generated Python inside the
Agent Zero framework process and does not provide process, filesystem, or
network isolation.

When Agent Zero itself runs in Docker, the RLM Docker environment needs access
to an external Docker daemon, commonly through a carefully controlled Docker
socket mount. Docker socket access is highly privileged; use a hardened proxy or
equivalent isolation appropriate to your deployment.

The **RLM Context Explorer** separates Docker CLI, endpoint, daemon, and sandbox
probe status instead of treating a mounted socket as sufficient. For the
standard `agent-zero` Compose service, the plugin ships a host-side helper:

```bash
./usr/plugins/rlm/setup/enable-docker-access.sh --apply
```

The helper starts with read-only validation, asks before changing anything,
builds a small Agent Zero-derived image containing the official Docker CLI, and
recreates only that Compose service with the selected Unix socket mounted. Use
`--help` to select a Compose file, base image, socket, or derived image name.

Upstream RLM starts a nested sandbox and calls back to a short-lived proxy in
the Agent Zero process. The plugin-owned Docker CLI shim joins that sandbox to
Agent Zero's existing Docker network, maps `host.docker.internal` to the Agent
Zero container, and translates the temporary workspace from its in-container
path to the corresponding host bind path. Other Docker commands pass through
unchanged. The Explorer's **Run Sandbox Probe** action verifies the CLI, daemon,
image, bind mount, and callback path together before a real model run.

An already-running container cannot gain a new bind mount. Non-Compose or
custom service layouts therefore require an equivalent host-side recreation:
provide a Docker CLI in the Agent Zero image, mount the Docker endpoint at
`/var/run/docker.sock` (or configure `DOCKER_HOST`), and keep `/a0` backed by a
host bind mount so nested workspaces can be shared.

`local` mode remains available as an explicit informed opt-in for trusted,
development-only workloads. Do not use it with untrusted prompts or documents.

Attachment ingestion is restricted to Agent Zero's `usr/uploads` directory and
rejects symlink escapes. Other local paths are never read as RLM attachments.

## Configuration

- `auto_enabled`: allow selective automatic routing.
- `manual_tool_enabled`: expose the `rlm` tool.
- `trigger_threshold_pct`: context-pressure threshold for automatic routing.
- `min_block_chars`: smallest message field eligible for offloading.
- `attachment_max_chars`: per-text-attachment ingestion cap.
- `environment_mode`: `auto`, `docker`, or explicit `local`.
- `docker_image`: image used by the upstream Docker REPL.
- `max_depth`, `max_iterations`, `max_timeout`, `max_budget`, `max_tokens`,
  `max_errors`: upstream execution limits.
- `max_concurrent_subcalls`: bound parallel recursive fan-out.
- `subcall_model_source`: use Agent Zero's utility or root model for subcalls.
- `persistence_enabled`: save run summaries and trajectories locally.
- `retention_count`: number of persisted runs to retain.

Trajectory persistence is off by default because stored trajectories may
contain prompt, model-output, or tool-output excerpts. When enabled, data stays
under the plugin-owned `data/runs` directory and is ignored by Git.

## Supported Provider Mapping

The integration maps Agent Zero model settings to upstream RLM clients for
OpenAI-compatible providers (including OpenRouter), Anthropic, Azure OpenAI,
Gemini, and Portkey. The explorer reports the active mapping and any readiness
blocker before a run.

## Verification

The repository's regression suite uses only the Python standard library:

```bash
python3 -m unittest discover -s tests -v
node --check webui/rlm-context-store.js
```

An end-to-end RLM completion additionally requires the upstream dependency, a
configured provider credential, and the selected execution environment.
For Docker mode, run the Explorer sandbox probe before treating the route as
deployment-ready.

## Upstream and License

This Agent Zero integration is independently maintained by
`a0-community-plugins`. It depends on
[`alexzhang13/rlm`](https://github.com/alexzhang13/rlm), created by Alex Zhang
and distributed under the MIT License. This plugin is also distributed under
the MIT License; see [LICENSE](LICENSE).
