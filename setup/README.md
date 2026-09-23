# Docker setup

Most users should use **Copy Setup Command** in RLM Context Explorer. The command
copies this folder from the running Agent Zero container to `./rlm-setup` on your
computer. It does not assume that an Agent Zero checkout exists on your host.

## Run manually

Copy the **container name** from Docker Desktop. With the downloaded setup folder
in your current directory, run:

```bash
sh ./rlm-setup/enable-docker-access.sh --container agent-zero --apply
```

On Windows, use PowerShell:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File ./rlm-setup/enable-docker-access.ps1 -Container agent-zero -Apply
```

Replace `agent-zero` with your container's name. Omitting `--apply` / `-Apply`
checks the configuration without changing Agent Zero. `--yes` / `-Yes` skips the
confirmation for unattended setup. Windows must use Docker Desktop's **Linux
containers** mode. No Python installation, WSL, or Compose is needed on the host.

## What setup does

- Checks the selected container before stopping anything; downloads are also
  completed before the restart.
- Saves a **local** snapshot of the original container filesystem to retain
  installed software and data not already in mounts.
- Adds the official Docker CLI to a local derived image and recreates Agent Zero
  with its original configuration plus Docker access.
- Reuses existing data volumes, including anonymous volumes. If `/a0/usr` is not
  persisted, it creates a named volume populated from the snapshot.
- Keeps the original as `<name>-rlm-backup-<id>`, stopped with automatic restart
  disabled. It reconnects the replacement to the original networks and keeps
  the assigned web port, including ports originally allocated dynamically.
- Runs the plugin's file-sharing and callback sandbox probe in the framework
  Python. Rerunning setup on a configured, running container just repeats the
  probe. It does not create another backup.

The short-lived installer container uses the selected container's existing image
for Python and the Docker Engine API. It mounts the daemon socket and has no
ordinary network access; image downloads go through Docker. Neither container
settings nor credentials are printed or saved to host-side configuration files.
Snapshot images can contain credentials and other private data: do not push,
export, or share them. The copied `rlm-setup` folder contains only installer code.

The default socket `/var/run/docker.sock` is a **daemon-side** path. On Docker
Desktop it lives inside the Linux VM; your macOS client socket under
`~/.docker/run` or Windows named pipe must not be substituted as the bind source.
For a native rootless Linux daemon, pass `--socket /run/user/1000/docker.sock`
(or `-Socket` on PowerShell) with the actual daemon-side path. The helper follows
the Docker CLI's selected context; use the context containing Agent Zero.

Automatic migration rejects `--rm`, read-only/non-root containers, Swarm,
container-shared namespaces, and host/disabled networking before changing the
container. Those layouts need deployment-specific configuration. Docker Desktop
Enhanced Container Isolation or an organization policy may prohibit socket
access; the helper does not bypass those policies.

## Rollback and cleanup

If installation or the sandbox check fails, setup removes its replacement and
restores the original container's name, networks, restart policy and previous
running state. It does not delete user volumes.
If Docker originally assigned a random web port, restarting that original can
assign a new port. Setup prints the restored container's actual web address.

After successful setup, check your chats/settings before removing the stopped
backup in Docker Desktop. **Do not start both copies together:** they share data
and ports. For a later manual rollback, stop and remove the replacement (keep its
volumes), rename the backup to the original name, restore its restart policy,
and start it. The backup retains its original networks, IP settings, and aliases.
For a container that originally used `unless-stopped`:

```bash
docker stop agent-zero
docker rm agent-zero
docker rename agent-zero-rlm-backup-REPLACE_WITH_ID agent-zero
docker update --restart unless-stopped agent-zero
docker start agent-zero
```

Use the backup name printed by setup, your actual container name, and your
original restart policy. Remove only RLM setup images that are no longer
used, through Docker Desktop's Images screen. Deleting/disabling the RLM plugin
does not undo container-level socket access; rollback or recreate the container
without that mount to revoke it.

A launcher or Compose deployment can later recreate a container from its saved
settings and discard these Docker-access additions. In that case rerun setup,
or persist equivalent settings in your deployment definition. Ordinary Docker
Desktop stop/start and restarts retain setup.

## Optional Compose setup

These files are **for users already maintaining a Compose deployment**:

| File | Purpose |
| --- | --- |
| `Dockerfile.rlm` | Adds Docker CLI to an Agent Zero image. Builds from this folder alone. |
| `docker-compose.rlm.yml` | Override for an existing service named `agent-zero`; not a standalone deployment. |

The Dockerfile's `/usr/local/bin/docker` source is inside the official Docker CLI
image, not a directory on your computer. `/a0/usr` is inside Agent Zero. The
Compose override intentionally contains no host data directory or web port;
those must come from your existing Compose file. Rename the override's service
if your service has a different name.

From the directory containing your existing `docker-compose.yml` and downloaded
`rlm-setup` folder:

```bash
docker build --build-arg A0_BASE_IMAGE=agent0ai/agent-zero:latest \
  -t agent-zero-rlm:local -f ./rlm-setup/Dockerfile.rlm ./rlm-setup
docker compose -f docker-compose.yml -f ./rlm-setup/docker-compose.rlm.yml config --quiet
docker compose -f docker-compose.yml -f ./rlm-setup/docker-compose.rlm.yml up -d --no-deps agent-zero
```

Choose the base image/version you actually use. This optional build extends that
image, not the writable layer of an existing container: software installed only
in the old container needs reinstalling. Keep `/a0/usr` (or legacy `/a0`) persisted
in your existing Compose service and run the Explorer's sandbox probe afterwards.
Continue using both Compose files on future updates. No helper guesses the first
image in a multi-service Compose project.
