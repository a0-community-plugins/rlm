from __future__ import annotations

from datetime import UTC, datetime
import json
import os
from pathlib import Path
import secrets
import shlex
import socket
import subprocess
import tempfile
from typing import Any

from usr.plugins.rlm.helpers.docker_shim import (
    detect_containerized_runtime,
    find_real_docker_cli,
)


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
DOCKER_SHIM_DIR = PLUGIN_ROOT / "bin"
DOCKER_SHIM_PATH = DOCKER_SHIM_DIR / "docker"
PROBE_ROOT = PLUGIN_ROOT / "data" / "docker-probes"
PROBE_RECORD = PLUGIN_ROOT / "data" / "docker-probe.json"
SOCKET_RISK = (
    "A raw Docker socket is effectively root-level control of the Docker host. "
    "Only expose it to a trusted Agent Zero deployment, or use a hardened Docker "
    "API proxy with the container and image operations RLM requires."
)


def activate_docker_cli_shim() -> str:
    # Upstream 0.1.3 defaults to /a0/.rlm_workspace, outside the data mount in
    # current Docker Desktop/launcher installations that persist only /a0/usr.
    os.environ.setdefault("RLM_DOCKER_WORKSPACE_DIR", str(PLUGIN_ROOT / "data" / "workspaces"))
    if not DOCKER_SHIM_PATH.is_file():
        return ""
    shim_dir = str(DOCKER_SHIM_DIR)
    entries = [
        entry
        for entry in os.getenv("PATH", "").split(os.pathsep)
        if entry and entry != shim_dir
    ]
    os.environ["PATH"] = os.pathsep.join([shim_dir, *entries])
    return str(DOCKER_SHIM_PATH)


def deactivate_docker_cli_shim() -> None:
    shim_dir = str(DOCKER_SHIM_DIR)
    entries = [
        entry
        for entry in os.getenv("PATH", "").split(os.pathsep)
        if entry and entry != shim_dir
    ]
    os.environ["PATH"] = os.pathsep.join(entries)


def docker_info_available(timeout: float = 3.0) -> bool:
    real_docker = find_real_docker_cli(DOCKER_SHIM_PATH)
    if not real_docker:
        return False
    try:
        result = subprocess.run(
            [real_docker, "info"],
            capture_output=True,
            check=False,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def has_docker_endpoint() -> bool:
    docker_host = str(os.getenv("DOCKER_HOST", "") or "").strip()
    if docker_host:
        if docker_host.startswith("unix://"):
            return Path(docker_host.removeprefix("unix://")).exists()
        return True
    return Path("/var/run/docker.sock").exists()


def get_docker_setup_status() -> dict[str, Any]:
    activate_docker_cli_shim()
    real_docker = find_real_docker_cli(DOCKER_SHIM_PATH)
    containerized = detect_containerized_runtime()
    endpoint_kind, endpoint_present = _endpoint_summary()
    daemon_reachable = docker_info_available()
    probe = read_probe_record()
    return {
        "containerized_runtime": containerized,
        "cli_available": bool(real_docker),
        "cli_source": _cli_source(real_docker),
        "endpoint_kind": endpoint_kind,
        "endpoint_present": endpoint_present,
        "daemon_reachable": daemon_reachable,
        "shim_active": _shim_is_active(),
        "setup_required": not (real_docker and endpoint_present and daemon_reachable),
        "requires_container_recreate": bool(
            containerized and (not real_docker or not endpoint_present)
        ),
        "setup_command": setup_commands()["shell"],
        "setup_commands": setup_commands(),
        "risk": SOCKET_RISK,
        "last_probe": probe,
    }


def setup_commands() -> dict[str, str]:
    """Copy setup from the installed plugin; no host checkout path is assumed."""
    target = os.getenv("RLM_AGENT_ZERO_CONTAINER") or os.getenv("HOSTNAME") or "YOUR_AGENT_ZERO_CONTAINER"
    source = f"{target}:{PLUGIN_ROOT}/setup/."
    shell = (
        "mkdir -p ./rlm-setup && "
        f"docker cp {shlex.quote(source)} ./rlm-setup/ && "
        f"sh ./rlm-setup/enable-docker-access.sh --container {shlex.quote(target)} --apply"
    )
    ps_quote = lambda value: "'" + value.replace("'", "''") + "'"
    powershell = (
        "New-Item -ItemType Directory -Force ./rlm-setup | Out-Null; "
        f"docker cp {ps_quote(source)} ./rlm-setup/; "
        "if ($LASTEXITCODE -eq 0) { powershell -NoProfile -ExecutionPolicy Bypass "
        f"-File ./rlm-setup/enable-docker-access.ps1 -Container {ps_quote(target)} -Apply }}"
    )
    return {"shell": shell, "powershell": powershell}


def run_docker_sandbox_probe(
    image: str,
    *,
    timeout: float = 180.0,
) -> dict[str, Any]:
    activate_docker_cli_shim()
    checked_at = datetime.now(UTC).isoformat()
    image = str(image or "python:3.11-slim").strip() or "python:3.11-slim"
    if not docker_info_available():
        return _save_probe_record(
            {
                "success": False,
                "checked_at": checked_at,
                "image": image,
                "message": "Docker CLI or daemon is unavailable.",
            }
        )

    PROBE_ROOT.mkdir(parents=True, exist_ok=True)
    token = secrets.token_urlsafe(24)
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("0.0.0.0", 0))
    listener.listen(1)
    listener.settimeout(min(timeout, 30.0))
    port = int(listener.getsockname()[1])

    try:
        with tempfile.TemporaryDirectory(prefix="probe-", dir=PROBE_ROOT) as temp_dir:
            workspace = Path(temp_dir)
            (workspace / "probe.txt").write_text(token, encoding="utf-8")
            script = (
                "from pathlib import Path; import socket; "
                "token=Path('/workspace/probe.txt').read_text().strip(); "
                f"s=socket.create_connection(('host.docker.internal',{port}),10); "
                "s.sendall(token.encode()); s.close(); print(token)"
            )
            command = [
                str(DOCKER_SHIM_PATH),
                "run",
                "--rm",
                "-v",
                f"{workspace}:/workspace:ro",
                "--add-host",
                "host.docker.internal:host-gateway",
                image,
                "python",
                "-c",
                script,
            ]
            result = subprocess.run(
                command,
                capture_output=True,
                check=False,
                text=True,
                timeout=timeout,
            )
            received = b""
            if result.returncode == 0:
                connection, _ = listener.accept()
                with connection:
                    connection.settimeout(5)
                    received = connection.recv(256)
            if (
                result.returncode != 0
                or token not in result.stdout
                or received.decode("utf-8", errors="replace") != token
            ):
                detail = (result.stderr or result.stdout or "probe callback failed").strip()
                raise RuntimeError(detail[:1200])
    except Exception as exc:
        return _save_probe_record(
            {
                "success": False,
                "checked_at": checked_at,
                "image": image,
                "message": f"{type(exc).__name__}: {exc}"[:1400],
            }
        )
    finally:
        listener.close()

    return _save_probe_record(
        {
            "success": True,
            "checked_at": checked_at,
            "image": image,
            "message": (
                "Sandbox container, bind-mounted workspace, and callback network "
                "path all passed."
            ),
        }
    )


def read_probe_record() -> dict[str, Any] | None:
    try:
        payload = json.loads(PROBE_RECORD.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _save_probe_record(record: dict[str, Any]) -> dict[str, Any]:
    PROBE_RECORD.parent.mkdir(parents=True, exist_ok=True)
    temporary = PROBE_RECORD.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(record, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(PROBE_RECORD)
    return record


def _endpoint_summary() -> tuple[str, bool]:
    docker_host = str(os.getenv("DOCKER_HOST", "") or "").strip()
    if docker_host.startswith("unix://"):
        return "Unix socket", Path(docker_host.removeprefix("unix://")).exists()
    if docker_host.startswith(("tcp://", "http://", "https://", "ssh://")):
        return "Remote endpoint", True
    socket_path = Path("/var/run/docker.sock")
    if socket_path.exists():
        return "Default Unix socket", True
    return "Not detected", False


def _cli_source(real_docker: str | None) -> str:
    if not real_docker:
        return "Missing"
    if real_docker == "/usr/local/libexec/rlm-docker/docker":
        return "RLM derived image"
    return "System Docker CLI"


def _shim_is_active() -> bool:
    path_entries = os.getenv("PATH", "").split(os.pathsep)
    return bool(path_entries and path_entries[0] == str(DOCKER_SHIM_DIR))
