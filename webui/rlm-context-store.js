import { createStore } from "/js/AlpineStore.js";
import { callJsonApi } from "/js/api.js";
import { toastFrontendError, toastFrontendSuccess } from "/components/notifications/notification-store.js";

const STATUS_API = "/plugins/rlm/status";
const RUNS_API = "/plugins/rlm/runs";
const DOCKER_SETUP_API = "/plugins/rlm/docker_setup";
const INSTALL_SETUP_API = "/plugins/rlm/install_setup";
const TITLE = "RLM";

function formatError(error, fallback) {
  return error instanceof Error && error.message ? error.message : fallback;
}

const model = {
  status: null,
  runs: [],
  selectedRun: null,
  selectedRunId: "",
  selectedIterationId: 0,
  loadingStatus: false,
  loadingRuns: false,
  loadingRun: false,
  pruning: false,
  probingDocker: false,
  settingUp: false,
  initialized: false,
  openPromise: null,

  async onOpen() {
    if (this.openPromise) {
      return await this.openPromise;
    }

    this.openPromise = Promise.all([this.loadStatus(), this.loadRuns()]);
    try {
      await this.openPromise;
      this.initialized = true;
    } finally {
      this.openPromise = null;
    }
  },

  async loadStatus() {
    this.loadingStatus = true;
    try {
      const response = await callJsonApi(STATUS_API, {});
      this.status = response;
    } catch (error) {
      this.status = null;
      void toastFrontendError(
        `Failed to load plugin status: ${formatError(error, "Unknown error")}`,
        TITLE,
      );
    } finally {
      this.loadingStatus = false;
    }
  },

  async loadRuns() {
    this.loadingRuns = true;
    try {
      const response = await callJsonApi(RUNS_API, { action: "list" });
      this.runs = Array.isArray(response?.runs) ? response.runs : [];

      if (!this.selectedRunId && this.runs.length > 0) {
        this.selectedRunId = this.runs[0].run_id;
      }

      if (this.selectedRunId) {
        await this.loadRun(this.selectedRunId, true);
      } else {
        this.selectedRun = null;
      }
    } catch (error) {
      this.runs = [];
      this.selectedRun = null;
      void toastFrontendError(
        `Failed to load run history: ${formatError(error, "Unknown error")}`,
        TITLE,
      );
    } finally {
      this.loadingRuns = false;
    }
  },

  async loadRun(runId, silent = false) {
    if (!runId) {
      this.selectedRun = null;
      this.selectedRunId = "";
      this.selectedIterationId = 0;
      return;
    }

    this.selectedRunId = runId;
    this.loadingRun = !silent;
    try {
      const response = await callJsonApi(RUNS_API, { action: "get", run_id: runId });
      this.selectedRun = response?.success ? response.run : null;
      const iterations = this.selectedRunView().iterations || [];
      const stillSelected = iterations.some((item) => item.iteration === this.selectedIterationId);
      this.selectedIterationId = stillSelected
        ? this.selectedIterationId
        : (iterations[0]?.iteration || 0);
    } catch (error) {
      this.selectedRun = null;
      this.selectedIterationId = 0;
      if (!silent) {
        void toastFrontendError(
          `Failed to load run details: ${formatError(error, "Unknown error")}`,
          TITLE,
        );
      }
    } finally {
      this.loadingRun = false;
    }
  },

  async pruneRuns(keep = 0) {
    this.pruning = true;
    try {
      const response = await callJsonApi(RUNS_API, { action: "prune", keep });
      void toastFrontendSuccess(`Removed ${response?.removed || 0} stored runs.`, TITLE);
      this.selectedRun = null;
      this.selectedRunId = "";
      this.selectedIterationId = 0;
      await this.loadRuns();
    } catch (error) {
      void toastFrontendError(
        `Failed to prune runs: ${formatError(error, "Unknown error")}`,
        TITLE,
      );
    } finally {
      this.pruning = false;
    }
  },

  async retrySetup() {
    this.settingUp = true;
    try {
      const response = await callJsonApi(INSTALL_SETUP_API, {});
      this.status = { ...(this.status || {}), ...(response || {}) };
      void toastFrontendSuccess("RLM setup completed through the install hook.", TITLE);
    } catch (error) {
      void toastFrontendError(
        `RLM setup failed: ${formatError(error, "Unknown error")}`,
        TITLE,
      );
    } finally {
      this.settingUp = false;
    }
  },

  async runDockerProbe() {
    this.probingDocker = true;
    try {
      const response = await callJsonApi(DOCKER_SETUP_API, { action: "probe" });
      if (response?.docker_setup && this.status) {
        this.status = { ...this.status, docker_setup: response.docker_setup };
      } else {
        await this.loadStatus();
      }
      if (response?.success) {
        void toastFrontendSuccess("Docker sandbox probe passed.", TITLE);
      } else {
        void toastFrontendError(
          response?.probe?.message || "Docker sandbox probe failed.",
          TITLE,
        );
      }
    } catch (error) {
      void toastFrontendError(
        `Docker sandbox probe failed: ${formatError(error, "Unknown error")}`,
        TITLE,
      );
    } finally {
      this.probingDocker = false;
    }
  },

  async copyDockerSetupCommand() {
    const command = this.dockerSetup().setup_command;
    if (!command) {
      return;
    }
    try {
      await navigator.clipboard.writeText(command);
      void toastFrontendSuccess("Host setup command copied.", TITLE);
    } catch (error) {
      void toastFrontendError(
        `Could not copy setup command: ${formatError(error, "Clipboard unavailable")}`,
        TITLE,
      );
    }
  },

  dependencyStateLabel() {
    if (!this.status?.dependency_installed) {
      return "Missing";
    }
    return this.status?.dependency_satisfied ? "Installed" : "Upgrade Needed";
  },

  dockerSetup() {
    return this.status?.docker_setup || {
      cli_available: false,
      cli_source: "Missing",
      endpoint_kind: "Not detected",
      endpoint_present: false,
      daemon_reachable: false,
      setup_required: true,
      requires_container_recreate: false,
      setup_command: "./usr/plugins/rlm/setup/enable-docker-access.sh --apply",
      risk: "",
      last_probe: null,
    };
  },

  dockerProbeStateLabel() {
    const probe = this.dockerSetup().last_probe;
    if (!probe) {
      return "Not run";
    }
    return probe.success ? "Passed" : "Failed";
  },

  readiness() {
    return this.status?.readiness || {
      auto_enabled: false,
      manual_tool_enabled: false,
      auto_ready: false,
      manual_ready: false,
      blockers: [],
      advisories: [],
      environment: {},
      root_model: {},
      subcall_model: {},
    };
  },

  readinessLabel(ready, enabled) {
    if (!enabled) {
      return "Disabled";
    }
    return ready ? "Ready" : "Blocked";
  },

  readinessToneClass(ready, enabled) {
    if (!enabled) {
      return "is-muted";
    }
    return ready ? "is-success" : "is-warning";
  },

  selectedRunView() {
    return this.selectedRun?.view || {
      metrics: {},
      chart: {},
      iterations: [],
      subcalls: [],
      run_metadata: {},
    };
  },

  selectedIterationView() {
    return this.selectedRunView().iterations.find(
      (item) => item.iteration === this.selectedIterationId,
    ) || null;
  },

  selectIteration(iterationId) {
    this.selectedIterationId = Number(iterationId || 0);
  },

  formatJson(value) {
    try {
      return JSON.stringify(value ?? {}, null, 2);
    } catch {
      return "{}";
    }
  },

  shortValue(value, fallback = "Unknown") {
    if (value === null || value === undefined || value === "") {
      return fallback;
    }
    return `${value}`;
  },

  previewText(value, limit = 180) {
    const text = this.shortValue(value, "").trim();
    if (!text) {
      return "";
    }
    if (text.length <= limit) {
      return text;
    }
    return `${text.slice(0, Math.max(0, limit - 1)).trimEnd()}…`;
  },

  formatSeconds(value) {
    const seconds = Number(value || 0);
    if (!Number.isFinite(seconds)) {
      return "0.0s";
    }
    return `${seconds.toFixed(seconds >= 10 ? 1 : 2)}s`;
  },

  formatCount(value) {
    const count = Number(value || 0);
    if (!Number.isFinite(count)) {
      return "0";
    }
    return count.toLocaleString();
  },

  statusToneClass(status) {
    const normalized = `${status || ""}`.toLowerCase();
    if (normalized.includes("timeout") || normalized.includes("error")) {
      return "is-warning";
    }
    if (normalized.includes("cancel")) {
      return "is-muted";
    }
    return "is-success";
  },

  iterationToneClass(iteration) {
    if (!iteration) {
      return "";
    }
    if (iteration.final_answer) {
      return "is-final";
    }
    if (iteration.had_error) {
      return "is-error";
    }
    return "is-normal";
  },

  barStyle(pct, minPct = 6) {
    const numeric = Number(pct || 0);
    const width = numeric > 0 ? Math.max(minPct, Math.min(100, numeric)) : 0;
    return `width:${width}%`;
  },
};

export const store = createStore("rlmContextDashboard", model);
