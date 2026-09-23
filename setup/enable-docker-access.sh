#!/bin/sh
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
container=""
apply=0
yes=0
socket_path=/var/run/docker.sock
usage() {
    printf '%s\n' \
        "Usage: $0 --container NAME [--check | --apply] [--yes] [--socket PATH]" \
        "Run in your computer's terminal, not Agent Zero's terminal or Docker's Exec tab." \
        "No Compose file or host Python installation is needed." \
        "Default: check only. --apply preserves the original as a stopped rollback copy." \
        "--socket is a path on the Docker daemon's machine (Desktop: /var/run/docker.sock)."
}
while [ "$#" -gt 0 ]; do
    case "$1" in
        --container|--socket)
            option=$1
            shift
            if [ "$#" -eq 0 ] || [ -z "$1" ]; then
                printf 'Missing value for %s\n' "$option" >&2; exit 2
            fi
            case "$option" in
                --container) container=$1 ;;
                --socket) socket_path=$1 ;;
            esac ;;
        --check) apply=0 ;;
        --apply) apply=1 ;;
        --yes) yes=1 ;;
        --help|-h) usage; exit 0 ;;
        *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done
if ! command -v docker >/dev/null 2>&1; then
    printf '%s\n' 'Docker command not found. Start Docker Desktop and open a new terminal.' >&2; exit 1
fi
if ! docker info >/dev/null 2>&1; then
    printf '%s\n' 'Docker is not reachable. Start Docker Desktop and try again.' >&2; exit 1
fi
if [ -z "$container" ]; then
    printf '%s\n' 'Choose your Agent Zero container from Docker Desktop, then pass --container NAME.' >&2
    docker ps -a --format '{{.Names}}'
    exit 2
fi
# Use the exact local image, not a mutable latest tag, to run framework Python.
image=$(docker inspect --type container --format '{{.Image}}' "$container")
if [ "$apply" -eq 1 ]; then
    printf '%s\n' \
        "Set up RLM for: $container" \
        'Agent Zero will restart. Its ports, data and installed software are preserved.' \
        'A stopped rollback copy and local snapshot image are kept; never publish the image.' \
        'This grants Agent Zero control of Docker and its containers through the Docker socket.'
    if [ "$yes" -ne 1 ]; then
        printf '%s' 'Continue? [y/N] '
        read -r answer
        case "$answer" in y|Y|yes|YES) ;; *) printf '%s\n' 'Cancelled.'; exit 0 ;; esac
    fi
fi
set -- --container "$container" --socket "$socket_path"
if [ "$apply" -eq 1 ]; then set -- "$@" --apply --yes; fi
# Desktop resolves this source inside its Linux VM. The macOS client socket and
# Windows named pipe are NOT valid socket bind sources for the Linux container.
# --mount fails if missing instead of creating a directory named docker.sock.
docker run --rm -i --user 0 --network none \
    --mount "type=bind,source=$socket_path,target=/var/run/docker.sock" \
    --entrypoint /opt/venv-a0/bin/python3 \
    "$image" - "$@" < "$script_dir/docker_desktop_setup.py"
