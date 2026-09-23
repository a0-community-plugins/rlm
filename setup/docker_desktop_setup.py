"""Docker Desktop setup, run by the host launchers inside a temporary container.

Standard library only. Never prints container environment or writes inspect output.
The original container stays available as a stopped rollback copy.
"""
from __future__ import annotations

import argparse
import copy
import http.client
import json
import signal
import socket
import sys
from urllib.parse import quote, urlencode

SOCKET = "/var/run/docker.sock"
WORKSPACE = "/a0/usr/plugins/rlm/data/workspaces"
SETUP_LABEL = "org.a0.rlm.setup"


class SetupError(RuntimeError):
    pass


class UnixConnection(http.client.HTTPConnection):
    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(SOCKET)


class Engine:
    def __init__(self):
        self.prefix = ""
        version = self.request("GET", "/version")
        if version.get("Os") != "linux":
            raise SetupError("Switch Docker Desktop to Linux containers, then try again.")
        self.prefix = "/v" + version["ApiVersion"]

    def request(self, method, path, body=None, *, raw=False, content_type="application/json"):
        connection = UnixConnection("localhost", timeout=600)
        data = body if isinstance(body, bytes) else json.dumps(body).encode() if body is not None else None
        try:
            connection.request(method, self.prefix + path, data, {"Content-Type": content_type})
            response = connection.getresponse()
            payload = response.read()
            if response.status >= 400:
                # Do not echo request bodies: container configuration can contain secrets.
                raise SetupError(f"Docker {method} {path.split('?')[0]} failed (HTTP {response.status}).")
            if raw:
                return payload
            return json.loads(payload) if payload else None
        finally:
            connection.close()

    def inspect(self, target):
        return self.request("GET", f"/containers/{quote(target, safe='')}/json")

    def container(self, target, action, body=None, query=None):
        suffix = "?" + urlencode(query) if query else ""
        return self.request("POST", f"/containers/{quote(target, safe='')}/{action}{suffix}", body)

    def create(self, config, name=None):
        query = "?" + urlencode({"name": name}) if name else ""
        return self.request("POST", "/containers/create" + query, config)["Id"]

    def remove(self, target, *, volumes=False):
        return self.request("DELETE", f"/containers/{target}?force=true&v={str(volumes).lower()}")

    def pull(self, image, platform=None):
        try:
            return self.request("GET", f"/images/{quote(image, safe='')}/json")["Id"]
        except SetupError:
            print(f"Downloading {image} (first setup only)...", flush=True)
        query = {"fromImage": image}
        if platform:
            query["platform"] = platform
        payload = self.request("POST", "/images/create?" + urlencode(query), raw=True)
        for line in payload.splitlines():
            if json.loads(line).get("error"):
                raise SetupError(f"Could not download {image}. Check Docker Desktop's network access.")
        return self.request("GET", f"/images/{quote(image, safe='')}/json")["Id"]

    def execute(self, target, command):
        execution = self.container(target, "exec", {
            "AttachStdout": True, "AttachStderr": True, "Cmd": command,
            "WorkingDir": "/a0", "User": "0",
        })["Id"]
        output = self.request("POST", f"/exec/{execution}/start", {"Detach": False, "Tty": False}, raw=True)
        decoded = bytearray()
        while len(output) >= 8:
            size = int.from_bytes(output[4:8], "big")
            decoded.extend(output[8:8 + size])
            output = output[8 + size:]
        status = self.request("GET", f"/exec/{execution}/json")
        return status["ExitCode"], decoded.decode(errors="replace")


def endpoint_config(details, original_id):
    result = {key: copy.deepcopy(details[key]) for key in ("IPAMConfig", "DriverOpts", "GwPriority") if details.get(key) is not None}
    aliases = [name for name in details.get("Aliases") or [] if name not in (original_id, original_id[:12])]
    if aliases:
        result["Aliases"] = aliases
    return result


def replacement_config(original, image, socket_path=SOCKET):
    """Preserve Engine create settings, including anonymous volumes and fixed ports."""
    host = copy.deepcopy(original["HostConfig"])
    config = copy.deepcopy(original["Config"])
    name = original["Name"].lstrip("/")
    if host.get("AutoRemove"):
        raise SetupError("This container uses --rm. Recreate it without automatic removal before setup.")
    if host.get("NetworkMode", "") in ("host", "none") or str(host.get("NetworkMode", "")).startswith("container:"):
        raise SetupError("RLM setup requires a bridge network. This container uses a custom network mode.")
    if host.get("Links") or host.get("VolumesFrom") or any(str(host.get(key, "")).startswith("container:") for key in ("PidMode", "IpcMode", "UTSMode")):
        raise SetupError("Linked containers or shared namespaces need manual setup; nothing was changed.")
    if any(key.startswith("com.docker.swarm.") for key in config.get("Labels", {}) or {}):
        raise SetupError("This container is managed by Docker Swarm. Configure its service instead.")
    if host.get("ReadonlyRootfs") or config.get("User") not in (None, "", "0", "root", "0:0", "root:root"):
        raise SetupError("This hardened container needs manual Docker socket permissions and CLI installation.")
    networks = original.get("NetworkSettings", {}).get("Networks", {})
    if not networks:
        raise SetupError("Connect Agent Zero to a bridge network before setup.")

    # Mounts, unlike Config.Volumes, identify the existing anonymous volume names.
    # Keep detailed user-supplied mount options and add implicit image volumes.
    binds = host.get("Binds") or []
    mounts = host.get("Mounts") or []
    existing = {m["Target"] for m in mounts}
    existing.update(b.split(":")[1] for b in binds if ":" in b)
    persistent_usr = False
    for mount in original.get("Mounts", []):
        destination = mount["Destination"]
        if destination in ("/a0", "/a0/usr"):
            if not mount.get("RW", True) or mount["Type"] not in ("bind", "volume"):
                raise SetupError("Agent Zero's data directory must be a writable bind mount or Docker volume.")
            persistent_usr = True
        if destination not in existing and mount["Type"] == "volume":
            mounts.append({"Type": "volume", "Source": mount["Name"], "Target": destination, "ReadOnly": not mount.get("RW", True)})
    # New named volume is populated from the local snapshot, including existing usr data.
    if not persistent_usr:
        mounts.append({"Type": "volume", "Source": f"rlm-{original['Id'][:12]}-usr", "Target": "/a0/usr"})
    binds = [b for b in binds if b.split(":")[1] != SOCKET]
    mounts = [m for m in mounts if m["Target"] != SOCKET]
    mounts.append({"Type": "bind", "Source": socket_path, "Target": SOCKET})
    host["Binds"], host["Mounts"] = binds, mounts

    # HostPort=0 means allocate a new port; keep the actual published port instead.
    for port, bindings in (host.get("PortBindings") or {}).items():
        if any(binding.get("HostPort") in (None, "", "0") for binding in bindings or []):
            resolved = original.get("NetworkSettings", {}).get("Ports", {}).get(port)
            if not resolved:
                raise SetupError("Start this container once so Docker can assign its web port, then retry setup.")
            host["PortBindings"][port] = copy.deepcopy(resolved)
    if host.get("PublishAllPorts"):
        host.setdefault("PortBindings", {}).update(copy.deepcopy(original["NetworkSettings"].get("Ports") or {}))
        host["PublishAllPorts"] = False
    env = [item for item in config.get("Env", []) if item.split("=", 1)[0] not in (
        "DOCKER_HOST", "DOCKER_CONTEXT", "DOCKER_TLS_VERIFY", "DOCKER_CERT_PATH",
        "RLM_AGENT_ZERO_CONTAINER", "RLM_DOCKER_WORKSPACE_DIR", "RLM_REAL_DOCKER_BIN",
    )]
    env.extend([f"DOCKER_HOST=unix://{SOCKET}", f"RLM_AGENT_ZERO_CONTAINER={name}", f"RLM_DOCKER_WORKSPACE_DIR={WORKSPACE}", "RLM_REAL_DOCKER_BIN=/usr/local/bin/docker"])
    config.update(Image=image, Env=env)
    config["Labels"] = {**(config.get("Labels") or {}), SETUP_LABEL: "1"}
    if config.get("Hostname") in (original["Id"], original["Id"][:12]):
        config["Hostname"] = ""
    for key in ("MacAddress", "Domainname"):
        if not config.get(key):
            config.pop(key, None)
    config["HostConfig"] = host
    config["NetworkingConfig"] = {"EndpointsConfig": {
        network: endpoint_config(details, original["Id"]) for network, details in networks.items()
    }}
    return config


def probe(engine, target, probe_image):
    script = (
        "import json,sys; from usr.plugins.rlm.helpers.docker_setup import run_docker_sandbox_probe; "
        f"r=run_docker_sandbox_probe({probe_image!r}); "
        "print(json.dumps(r)); sys.exit(0 if r['success'] else 1)"
    )
    code, output = engine.execute(target, ["/opt/venv-a0/bin/python3", "-c", script])
    if code:
        try:
            detail = str(json.loads(output.splitlines()[-1])["message"])[:1400]
        except (ValueError, IndexError, KeyError, TypeError):
            detail = "Framework Python could not run the plugin's sandbox check. Update RLM, then retry."
        raise SetupError(f"The RLM sandbox check failed: {detail}")


def report_web_address(engine, target):
    ports = engine.inspect(target).get("NetworkSettings", {}).get("Ports", {})
    for binding in ports.get("80/tcp") or []:
        host = binding.get("HostIp") or "127.0.0.1"
        if host in ("0.0.0.0", "::"):
            host = "localhost"
        elif ":" in host:
            host = f"[{host}]"
        print(f"Agent Zero address on the Docker host: http://{host}:{binding['HostPort']}", flush=True)


def install(engine, original, *, cli_image="docker:28-cli", probe_image="python:3.11-slim", socket_path=SOCKET):
    config = replacement_config(original, original["Image"], socket_path)
    old_id = original["Id"]
    name = original["Name"].lstrip("/")
    backup_name = f"{name}-rlm-backup-{old_id[:8]}"
    try:
        engine.inspect(backup_name)
    except SetupError:
        pass
    else:
        raise SetupError(f"A backup already exists: {backup_name}. Nothing was changed.")

    image_details = engine.request("GET", f"/images/{original['Image']}/json")
    platform = "linux/" + image_details["Architecture"]
    cli_id = engine.pull(cli_image, platform)
    engine.pull(probe_image, platform)
    donor = engine.create({"Image": cli_id, "Cmd": ["true"]})
    try:
        cli_archive = engine.request("GET", f"/containers/{donor}/archive?path=/usr/local/bin/docker", raw=True)
    finally:
        engine.remove(donor, volumes=True)
    was_running = original["State"]["Running"]
    renamed = False
    replacement = None
    restart_changed = False
    try:
        print("Saving the current container and its installed software locally...", flush=True)
        if was_running:
            engine.container(old_id, "stop", query={"t": 30})
        snapshot = engine.request("POST", "/commit?" + urlencode({"container": old_id, "pause": "false"}))["Id"]
        staging = engine.create({"Image": snapshot, "Cmd": ["true"], "Entrypoint": []})
        try:
            engine.request("PUT", f"/containers/{staging}/archive?path=/usr/local/bin", cli_archive, content_type="application/x-tar")
            # Supply the original image config; the staging command must not replace A0 startup.
            image = engine.request("POST", "/commit?" + urlencode({
                "container": staging, "repo": "rlm-local/agent-zero", "tag": old_id[:12], "pause": "false",
            }), original["Config"])["Id"]
        finally:
            engine.remove(staging, volumes=True)
        config["Image"] = image
        engine.container(old_id, "update", {"RestartPolicy": {"Name": "no"}})
        restart_changed = True
        engine.container(old_id, "rename", query={"name": backup_name})
        renamed = True
        # Stopping releases the active endpoints (including static IPs). Leave
        # the backup's network configuration intact so rollback can just start it.
        print("Starting Agent Zero with the same ports and data...", flush=True)
        replacement = engine.create(config, name)
        engine.container(replacement, "start")
        # The plugin import/probe runs in the framework Python, independently of WebUI startup.
        print("Checking the sandbox, shared files, and callback connection...", flush=True)
        probe(engine, replacement, probe_image)
        if not engine.inspect(replacement)["State"]["Running"]:
            raise SetupError("The replacement container exited during verification.")
    except BaseException:
        print("Setup did not finish. Restoring the original container...", flush=True)
        try:
            if replacement:
                engine.remove(replacement)  # Never delete user volumes on rollback.
            if renamed:
                engine.container(old_id, "rename", query={"name": name})
            if restart_changed:
                engine.container(old_id, "update", {"RestartPolicy": original["HostConfig"]["RestartPolicy"]})
            if was_running:
                engine.container(old_id, "start")
                # Docker may allocate a different random port when restarting
                # the untouched original. Print its actual address for recovery.
                report_web_address(engine, old_id)
        except Exception as recovery_error:
            raise SetupError(
                f"Automatic recovery was interrupted. Original container ID: {old_id[:12]}. "
                f"Keep this container and its volumes; see setup/README.md. ({recovery_error})"
            ) from recovery_error
        raise
    print(f"RLM Docker setup passed. Open Agent Zero at its usual address.\nStopped rollback copy: {backup_name}", flush=True)
    report_web_address(engine, replacement)
    print("Keep the backup until you have checked your chats and settings. Local snapshot images may contain private data; do not publish them.", flush=True)
    if (original["Config"].get("Labels") or {}).get("com.docker.compose.project"):
        print("This container is Compose-managed. Save equivalent RLM settings in Compose before your next Compose recreation; see setup/README.md.", flush=True)


def main():
    def interrupted(signum, frame):
        raise SetupError("Setup was interrupted.")

    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser(description="Configure the selected Agent Zero Docker container without Compose.")
    parser.add_argument("--container", required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--socket", default=SOCKET, help="Socket path on the Docker daemon's machine, not the desktop client socket")
    parser.add_argument("--cli-image", default="docker:28-cli")
    parser.add_argument("--probe-image", default="python:3.11-slim")
    args = parser.parse_args()
    try:
        engine = Engine()
        original = engine.inspect(args.container)
        replacement_config(original, original["Image"], args.socket)
        name = original["Name"].lstrip("/")
        print(f"Agent Zero container: {name}\nData, ports, environment, networks and installed software will be preserved.", flush=True)
        if not args.apply:
            print("Check passed. No changes made to Agent Zero. Run again with --apply to set up RLM.")
            return 0
        if not args.yes:
            raise SetupError("Run the host launcher with --apply and confirm its prompt.")
        if (original["Config"].get("Labels") or {}).get(SETUP_LABEL) == "1":
            was_running = original["State"]["Running"]
            if not was_running:
                engine.container(original["Id"], "start")
            try:
                probe(engine, original["Id"], args.probe_image)
            except BaseException:
                if not was_running:
                    engine.container(original["Id"], "stop", query={"t": 30})
                raise
            print("RLM Docker setup is already installed; sandbox check passed.")
            report_web_address(engine, original["Id"])
            return 0
        install(engine, original, cli_image=args.cli_image, probe_image=args.probe_image, socket_path=args.socket)
        return 0
    except (SetupError, OSError, http.client.HTTPException) as exc:
        print(f"RLM setup: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
