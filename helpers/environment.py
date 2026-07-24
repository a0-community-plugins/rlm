from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from usr.plugins.rlm.helpers.docker_setup import (
    docker_info_available,
    has_docker_endpoint,
)
from usr.plugins.rlm.helpers.docker_shim import detect_containerized_runtime


@dataclass
class EnvironmentResolution:
    environment: str
    environment_kwargs: dict
    reason: str = ""
    usable: bool = True


def is_docker_available(timeout: float = 2.0) -> bool:
    return docker_info_available(timeout=timeout)


def has_external_docker_access() -> bool:
    return has_docker_endpoint()


def resolve_environment(config: dict | None) -> EnvironmentResolution:
    config = config or {}
    mode = str(config.get("environment_mode", "auto") or "auto").lower()
    image = str(config.get("docker_image", "python:3.11-slim") or "python:3.11-slim")
    containerized_runtime = detect_containerized_runtime()
    docker_available = is_docker_available()
    docker_accessible_from_container = has_external_docker_access()

    if mode == "local":
        return EnvironmentResolution(
            "local",
            {},
            reason=(
                "Local REPL was explicitly selected. It executes model-generated Python "
                "inside the Agent Zero framework process without Docker isolation."
            ),
        )

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
                "docker",
                {},
                reason=(
                    "Auto mode requires an isolated Docker REPL, but this container cannot "
                    "reach an external Docker daemon. Mount the Docker socket or explicitly "
                    "opt into local mode after reviewing its security implications."
                ),
                usable=False,
            )
        return EnvironmentResolution(
            "docker",
            {"image": image},
            reason="Auto mode selected Docker because it is available.",
        )

    if containerized_runtime:
        return EnvironmentResolution(
            "docker",
            {},
            reason=(
                "Auto mode requires an isolated Docker REPL, but Docker is unavailable "
                "inside this Agent Zero container. Mount the Docker socket or explicitly "
                "opt into local mode after reviewing its security implications."
            ),
            usable=False,
        )

    return EnvironmentResolution(
        "docker",
        {},
        reason=(
            "Auto mode requires an isolated Docker REPL, but Docker is unavailable. "
            "Install Docker or explicitly opt into local mode after reviewing its "
            "security implications."
        ),
        usable=False,
    )
