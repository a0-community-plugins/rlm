# RLM

Recursive long-context analysis for Agent Zero, powered by
[`alexzhang13/rlm`](https://github.com/alexzhang13/rlm).

The plugin keeps Agent Zero in control of the conversation while selectively
offloading oversized external material to an RLM worker. Version 2.0.1 targets
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

## Installation — Docker Desktop

1. In Agent Zero, open **Plugins → Install Plugin → Git**, paste
   `https://github.com/a0-community-plugins/rlm`, and install/enable **RLM**.
2. Open **RLM Context Explorer**. Under **Set up Docker Desktop**, select
   **macOS / Linux** or **Windows**, then click **Copy Setup Command**.
3. Paste it into **your computer's Terminal or PowerShell** and confirm.
   Keep Docker Desktop running. Do **not** use Agent Zero's terminal or the
   container's Exec tab.
4. Wait for **RLM Docker setup passed**, then reopen Agent Zero at the same
   address. The sandbox check runs automatically.

**No Compose file, host Python installation, repository checkout, or hand-edited
Docker paths are required.** The copied command gets the setup files directly
from your installed plugin. First setup downloads the Docker CLI and a small
sandbox image; later runs reuse them.

Setup briefly restarts the selected container. It preserves its web port, data
mounts, environment, networks, and installed software. It supports both
`/a0/usr` and older `/a0` mounts, including Docker named volumes. If data is only
inside the container, setup preserves it in a new persistent Docker volume.
The original container is kept stopped as a rollback copy, and a failed setup
restores it automatically. See [setup details and rollback](setup/README.md).

Docker access lets Agent Zero manage Docker containers and access shared host
files. The setup command explains this before asking you to continue. Docker
cannot add a mount to an existing container from the plugin UI, which is why
this one host-terminal step is necessary. Local snapshot images can contain
private data; **never publish them**.

The plugin install hook automatically installs `rlms==0.1.3` into Agent Zero's
framework Python (`/opt/venv-a0/bin/python3` in the standard image). If that step
was interrupted, click **Retry Setup** in the Explorer. This button repairs the
Python dependency; it cannot grant Docker access. Restart Agent Zero after
upgrading a dependency that was already loaded.
There is no separate Execute step.

For an existing installation, update RLM first to get these setup controls.
Advanced users who manage Compose can use the optional
[Dockerfile and Compose override](setup/README.md#optional-compose-setup).

## Execution Safety

`auto` and `docker` modes require Docker and never silently switch to `local`.
Local mode executes generated Python in Agent Zero's own framework process and
is an explicit opt-in for trusted development workloads only.

The Explorer reports Docker CLI, endpoint, daemon, and sandbox-probe status
separately. A passing probe verifies shared files and the callback connection
from an actual sandbox; a complete model run also needs a configured provider.
RLM's shim joins the sandbox to Agent Zero's network, directs its callback to
Agent Zero, and translates the workspace to the daemon's shared data path.
Only the run's workspace is mounted in the sandbox, not the entire data volume.

Attachment ingestion is restricted to Agent Zero's `usr/uploads` directory
and rejects symlink escapes. Other local paths are not read as attachments.

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
