from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import subprocess


@dataclass
class EnvironmentResolution:
    environment: str
    environment_kwargs: dict
    reason: str = ""
    usable: bool = True


def detect_containerized_runtime() -> bool:
    if Path("/.dockerenv").exists():
        return True
    cgroup = Path("/proc/1/cgroup")
    if not cgroup.exists():
        return False
    try:
        text = cgroup.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return False
    return "docker" in text or "containerd" in text or "kubepods" in text


def is_docker_available(timeout: float = 2.0) -> bool:
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=timeout,
            check=False,
            text=True,
        )
    except Exception:
        return False
    return result.returncode == 0


def has_external_docker_access() -> bool:
    docker_host = str(os.getenv("DOCKER_HOST", "") or "").strip()
    if docker_host:
        return True
    return Path("/var/run/docker.sock").exists()


def resolve_environment(config: dict | None) -> EnvironmentResolution:
    config = config or {}
    mode = str(config.get("environment_mode", "auto") or "auto").lower()
    image = str(config.get("docker_image", "python:3.11-slim") or "python:3.11-slim")
    containerized_runtime = detect_containerized_runtime()
    docker_available = is_docker_available()
    docker_accessible_from_container = has_external_docker_access()

    if mode == "local":
        return EnvironmentResolution("local", {}, reason="Configured for local REPL.")

    if mode == "docker":
        if docker_available and (not containerized_runtime or docker_accessible_from_container):
            return EnvironmentResolution(
                "docker",
                {"image": image},
                reason="Configured for Docker REPL.",
            )
        return EnvironmentResolution(
            "docker",
            {"image": image},
            reason="Configured for Docker REPL, but Docker is unavailable.",
            usable=False,
        )

    if docker_available:
        if containerized_runtime and not docker_accessible_from_container:
            return EnvironmentResolution(
                "local",
                {},
                reason=(
                    "Auto mode fell back to local REPL inside a containerized runtime "
                    "because Docker is not externally accessible."
                ),
            )
        return EnvironmentResolution(
            "docker",
            {"image": image},
            reason="Auto mode selected Docker because it is available.",
        )

    if containerized_runtime:
        return EnvironmentResolution(
            "local",
            {},
            reason="Auto mode fell back to local REPL inside a containerized runtime because Docker is unavailable.",
        )

    return EnvironmentResolution(
        "local",
        {},
        reason="Auto mode fell back to local REPL because Docker is unavailable.",
    )
