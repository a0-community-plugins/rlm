from __future__ import annotations

import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
from typing import Any


RLM_HOST_ALIAS = "host.docker.internal:host-gateway"
RLM_REAL_DOCKER_ENV = "RLM_REAL_DOCKER_BIN"
RLM_CONTAINER_ENV = "RLM_AGENT_ZERO_CONTAINER"
RLM_NETWORK_ENV = "RLM_DOCKER_NETWORK"


def find_real_docker_cli(shim_path: Path | None = None) -> str | None:
    shim = (shim_path or Path(sys.argv[0])).resolve()
    candidates = [
        os.getenv(RLM_REAL_DOCKER_ENV, ""),
        "/usr/local/libexec/rlm-docker/docker",
        "/usr/bin/docker",
        "/usr/local/bin/docker",
        "/opt/homebrew/bin/docker",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if path.is_file() and os.access(path, os.X_OK) and path.resolve() != shim:
            return str(path)

    shim_dir = str(shim.parent)
    search_path = os.pathsep.join(
        entry for entry in os.getenv("PATH", "").split(os.pathsep) if entry != shim_dir
    )
    candidate = shutil.which("docker", path=search_path)
    if candidate and Path(candidate).resolve() != shim:
        return candidate
    return None


def detect_containerized_runtime() -> bool:
    if Path("/.dockerenv").exists():
        return True
    cgroup = Path("/proc/1/cgroup")
    if not cgroup.exists():
        return False
    try:
        text = cgroup.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    return any(marker in text for marker in ("docker", "containerd", "kubepods"))


def inspect_outer_container(real_docker: str) -> dict[str, Any]:
    target = str(os.getenv(RLM_CONTAINER_ENV, "") or os.getenv("HOSTNAME", "")).strip()
    if not target:
        raise RuntimeError(
            f"Cannot identify the Agent Zero container. Set {RLM_CONTAINER_ENV}."
        )
    result = subprocess.run(
        [real_docker, "inspect", target],
        capture_output=True,
        check=False,
        text=True,
        timeout=10,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "container not found").strip()
        raise RuntimeError(
            f"Cannot inspect Agent Zero container {target!r}: {detail}"
        )
    payload = json.loads(result.stdout)
    if not isinstance(payload, list) or not payload or not isinstance(payload[0], dict):
        raise RuntimeError(f"Docker returned no inspection data for {target!r}.")
    return payload[0]


def rewrite_rlm_run_args(
    args: list[str],
    inspection: dict[str, Any],
    *,
    requested_network: str = "",
) -> list[str]:
    if not _is_rlm_sandbox_run(args):
        return list(args)

    networks = (
        inspection.get("NetworkSettings", {}).get("Networks", {})
        if isinstance(inspection.get("NetworkSettings"), dict)
        else {}
    )
    network_name, outer_address = _select_network(networks, requested_network)
    if not network_name or not outer_address:
        raise RuntimeError(
            "The Agent Zero container has no reachable bridge-network address. "
            f"Set {RLM_NETWORK_ENV} to a connected Docker network."
        )

    rewritten = _rewrite_host_alias(args, outer_address)
    rewritten = _rewrite_workspace_mounts(
        rewritten,
        inspection.get("Mounts", []),
    )
    if not _has_network_argument(rewritten):
        rewritten[1:1] = ["--network", network_name]
    return rewritten


def _is_rlm_sandbox_run(args: list[str]) -> bool:
    if not args or args[0] != "run":
        return False
    for index, value in enumerate(args):
        if value == "--add-host" and index + 1 < len(args):
            if args[index + 1] == RLM_HOST_ALIAS:
                return True
        if value == f"--add-host={RLM_HOST_ALIAS}":
            return True
    return False


def _select_network(
    networks: dict[str, Any],
    requested_network: str,
) -> tuple[str, str]:
    if requested_network:
        details = networks.get(requested_network)
        if not isinstance(details, dict):
            raise RuntimeError(
                f"Configured Docker network {requested_network!r} is not attached "
                "to the Agent Zero container."
            )
        return requested_network, str(details.get("IPAddress", "") or "").strip()

    for name, details in networks.items():
        if not isinstance(details, dict):
            continue
        address = str(details.get("IPAddress", "") or "").strip()
        if address:
            return str(name), address
    return "", ""


def _rewrite_host_alias(args: list[str], address: str) -> list[str]:
    rewritten = list(args)
    replacement = f"host.docker.internal:{address}"
    for index, value in enumerate(rewritten):
        if value == "--add-host" and index + 1 < len(rewritten):
            if rewritten[index + 1] == RLM_HOST_ALIAS:
                rewritten[index + 1] = replacement
        elif value == f"--add-host={RLM_HOST_ALIAS}":
            rewritten[index] = f"--add-host={replacement}"
    return rewritten


def _rewrite_workspace_mounts(
    args: list[str],
    mounts: Any,
) -> list[str]:
    rewritten = list(args)
    bind_mounts = [
        mount
        for mount in mounts
        if isinstance(mount, dict)
        and mount.get("Type") == "bind"
        and mount.get("Source")
        and mount.get("Destination")
    ]
    for index, value in enumerate(rewritten):
        if value not in ("-v", "--volume") or index + 1 >= len(rewritten):
            continue
        specification = rewritten[index + 1]
        source, separator, remainder = specification.partition(":")
        if not separator or not source.startswith("/"):
            continue
        translated = _translate_bind_source(source, bind_mounts)
        if translated:
            rewritten[index + 1] = f"{translated}:{remainder}"
            continue
        if remainder.startswith("/workspace"):
            raise RuntimeError(
                "The RLM workspace is not inside a bind mount visible to the "
                "external Docker daemon. Bind-mount Agent Zero's /a0 directory "
                "from the host before running the Docker sandbox."
            )
    return rewritten


def _translate_bind_source(source: str, mounts: list[dict[str, Any]]) -> str | None:
    candidate = PurePosixPath(source)
    matches: list[tuple[int, PurePosixPath, PurePosixPath]] = []
    for mount in mounts:
        destination = PurePosixPath(str(mount["Destination"]))
        try:
            relative = candidate.relative_to(destination)
        except ValueError:
            continue
        host_source = PurePosixPath(str(mount["Source"]))
        matches.append((len(destination.parts), host_source, relative))
    if not matches:
        return None
    _, host_source, relative = max(matches, key=lambda item: item[0])
    return str(host_source / relative)


def _has_network_argument(args: list[str]) -> bool:
    return any(
        value in ("--network", "--net")
        or value.startswith("--network=")
        or value.startswith("--net=")
        for value in args
    )


def main() -> int:
    shim_path = Path(sys.argv[0]).resolve()
    real_docker = find_real_docker_cli(shim_path)
    if not real_docker:
        print(
            "RLM Docker setup is incomplete: no real Docker CLI is available. "
            "Run usr/plugins/rlm/setup/enable-docker-access.sh on the host.",
            file=sys.stderr,
        )
        return 127

    args = list(sys.argv[1:])
    if detect_containerized_runtime() and _is_rlm_sandbox_run(args):
        try:
            inspection = inspect_outer_container(real_docker)
            args = rewrite_rlm_run_args(
                args,
                inspection,
                requested_network=str(os.getenv(RLM_NETWORK_ENV, "") or "").strip(),
            )
        except Exception as exc:
            print(f"RLM Docker sandbox setup failed: {exc}", file=sys.stderr)
            return 125

    os.execv(real_docker, [real_docker, *args])
    return 126


if __name__ == "__main__":
    raise SystemExit(main())
