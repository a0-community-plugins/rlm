#!/bin/sh
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
plugin_root=$(CDPATH= cd -- "$script_dir/.." && pwd)
project_root=$(CDPATH= cd -- "$plugin_root/../../.." && pwd)
overlay_file="$script_dir/docker-compose.rlm.yml"
dockerfile="$script_dir/Dockerfile.rlm"
compose_file=""
base_image=""
socket_path="${RLM_DOCKER_SOCKET:-}"
target_image="${RLM_AGENT_ZERO_IMAGE:-agent-zero-rlm:local}"
apply_changes=0
assume_yes=0

usage() {
    printf '%s\n' \
        "Usage: $0 [--check] [--apply] [--yes] [--compose FILE]" \
        "          [--base-image IMAGE] [--socket PATH] [--target-image IMAGE]" \
        "" \
        "Default mode is read-only. --apply builds a small derived Agent Zero" \
        "image with the Docker CLI and recreates the standard agent-zero Compose" \
        "service with the selected Docker socket mounted."
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --check)
            apply_changes=0
            ;;
        --apply)
            apply_changes=1
            ;;
        --yes)
            assume_yes=1
            ;;
        --compose)
            shift
            compose_file=${1:-}
            ;;
        --base-image)
            shift
            base_image=${1:-}
            ;;
        --socket)
            shift
            socket_path=${1:-}
            ;;
        --target-image)
            shift
            target_image=${1:-}
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            printf 'Unknown option: %s\n' "$1" >&2
            usage >&2
            exit 2
            ;;
    esac
    shift
done

if ! command -v docker >/dev/null 2>&1; then
    printf '%s\n' "Docker CLI is not available on the host." >&2
    exit 1
fi
if ! docker info >/dev/null 2>&1; then
    printf '%s\n' "Docker daemon is not reachable from the host." >&2
    exit 1
fi

if [ -z "$socket_path" ]; then
    case "${DOCKER_HOST:-}" in
        unix://*)
            socket_path=${DOCKER_HOST#unix://}
            ;;
    esac
fi
if [ -z "$socket_path" ] && [ -S /var/run/docker.sock ]; then
    socket_path=/var/run/docker.sock
fi
if [ -z "$socket_path" ] && [ -n "${HOME:-}" ] && [ -S "$HOME/.docker/run/docker.sock" ]; then
    socket_path=$HOME/.docker/run/docker.sock
fi
if [ -z "$socket_path" ] || [ ! -S "$socket_path" ]; then
    printf '%s\n' \
        "No Docker Unix socket was found." \
        "Pass --socket PATH, or use a manually configured DOCKER_HOST endpoint." >&2
    exit 1
fi

if [ -z "$compose_file" ]; then
    for candidate in \
        "$project_root/compose.yaml" \
        "$project_root/compose.yml" \
        "$project_root/docker-compose.yml" \
        "$project_root/docker/run/docker-compose.yml"
    do
        if [ -f "$candidate" ]; then
            compose_file=$candidate
            break
        fi
    done
fi

printf 'Docker socket: %s\n' "$socket_path"
printf 'Compose overlay: %s\n' "$overlay_file"
printf '%s\n' \
    "Security note: a raw Docker socket grants root-level control of the Docker host."

if [ "$apply_changes" -eq 0 ]; then
    printf '%s\n' \
        "Readiness check passed on the host." \
        "Re-run with --apply after reviewing the security note."
    exit 0
fi

if [ -z "$compose_file" ] || [ ! -f "$compose_file" ]; then
    printf '%s\n' \
        "A Compose file is required for automated recreation." \
        "Pass --compose /absolute/path/to/docker-compose.yml." >&2
    exit 1
fi

if [ -z "$base_image" ]; then
    base_image=$(docker compose -f "$compose_file" config --images | sed -n '1p')
fi
if [ -z "$base_image" ]; then
    printf '%s\n' \
        "Could not infer the Agent Zero image. Pass --base-image IMAGE." >&2
    exit 1
fi

printf 'Base Agent Zero image: %s\n' "$base_image"
printf 'Derived image: %s\n' "$target_image"
printf 'Compose file: %s\n' "$compose_file"

if [ "$assume_yes" -ne 1 ]; then
    if [ ! -t 0 ]; then
        printf '%s\n' "Interactive confirmation required; rerun with --yes." >&2
        exit 1
    fi
    printf '%s' "Build the derived image and recreate agent-zero? [y/N] "
    read -r answer
    case "$answer" in
        y|Y|yes|YES)
            ;;
        *)
            printf '%s\n' "Cancelled."
            exit 0
            ;;
    esac
fi

docker build \
    --build-arg "A0_BASE_IMAGE=$base_image" \
    --tag "$target_image" \
    --file "$dockerfile" \
    "$script_dir"

RLM_DOCKER_SOCKET=$socket_path \
RLM_AGENT_ZERO_IMAGE=$target_image \
docker compose \
    -f "$compose_file" \
    -f "$overlay_file" \
    up -d --force-recreate agent-zero

RLM_DOCKER_SOCKET=$socket_path \
RLM_AGENT_ZERO_IMAGE=$target_image \
docker compose \
    -f "$compose_file" \
    -f "$overlay_file" \
    exec -T agent-zero docker info >/dev/null

printf '%s\n' \
    "Docker access is ready inside Agent Zero." \
    "Open the RLM Context Explorer and run the sandbox probe."
