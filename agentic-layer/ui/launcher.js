(function () {
  const existing = document.querySelector("[data-ida-agentic-launcher]");
  if (existing) {
    return;
  }

  const scriptOrigin = document.currentScript?.src ? new URL(document.currentScript.src).origin : "";
  const agentUrl = window.IDA_AGENTIC_CORE_URL || scriptOrigin || "/api/agentic/core";

  function getActiveDatasetContext() {
    try {
      const stored = JSON.parse(localStorage.getItem("ai-eda-workspace-v2") || "{}");
      const state = stored?.state || {};
      const activeDataset = state.activeDatasetKey && state.datasets ? state.datasets[state.activeDatasetKey] : null;
      const dataset = activeDataset || state;
      const fileName = dataset.fileName || "";
      const datasetId = dataset.datasetId || "";
      const columns = Array.isArray(dataset.columns) ? dataset.columns : [];
      if (!fileName && !datasetId && !columns.length) {
        return null;
      }

      const columnNames = columns
        .map((column) => column?.name)
        .filter(Boolean)
        .slice(0, 60);
      const numericColumns = columns
        .filter((column) => column?.role === "numeric" || String(column?.dtype || "").toLowerCase().includes("float") || String(column?.dtype || "").toLowerCase().includes("int"))
        .map((column) => column.name)
        .filter(Boolean)
        .slice(0, 40);

      return {
        fileName,
        datasetId,
        totalRows: dataset.totalRows || 0,
        loadedRows: dataset.loadedRowCount || 0,
        columns: columnNames.join(","),
        numericColumns: numericColumns.join(","),
      };
    } catch (error) {
      return null;
    }
  }
  const style = document.createElement("style");
  style.textContent = `
    [data-ida-agentic-launcher] {
      position: fixed;
      right: 24px;
      bottom: 24px;
      z-index: 9999;
      display: inline-flex;
      align-items: center;
      gap: 10px;
      min-height: 48px;
      border: 1px solid rgba(226, 232, 240, 0.9);
      border-radius: 999px;
      padding: 8px 14px 8px 8px;
      background: #ffffff;
      color: #0f172a;
      box-shadow: 0 20px 52px -24px rgba(17, 24, 39, 0.44);
      font: 500 13px/1 Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      cursor: pointer;
      transition: transform 140ms ease, box-shadow 140ms ease;
    }
    [data-ida-agentic-launcher] .ida-agentic-monogram {
      display: grid;
      place-items: center;
      width: 32px;
      height: 32px;
      border-radius: 999px;
      background: linear-gradient(135deg, #234e9e 0%, #2f5fa8 48%, #4cb8f0 100%);
      color: #ffffff;
      font-size: 10px;
      font-weight: 800;
      box-shadow: 0 10px 22px -14px rgba(47, 95, 168, 0.9);
    }
    [data-ida-agentic-launcher] .ida-agentic-ready {
      position: relative;
      display: inline-flex;
      width: 14px;
      height: 14px;
      align-items: center;
      justify-content: center;
    }
    [data-ida-agentic-launcher] .ida-agentic-ready::before {
      content: "";
      position: absolute;
      width: 14px;
      height: 14px;
      border-radius: 999px;
      background: rgba(52, 211, 153, 0.55);
      animation: ida-agentic-ping 1.4s cubic-bezier(0, 0, 0.2, 1) infinite;
    }
    [data-ida-agentic-launcher] .ida-agentic-ready::after {
      content: "";
      position: relative;
      width: 10px;
      height: 10px;
      border-radius: 999px;
      background: #10b981;
      box-shadow: 0 0 0 2px #d1fae5;
    }
    [data-ida-agentic-launcher][data-processing="true"] .ida-agentic-ready::before,
    [data-ida-agentic-launcher][data-processing="true"] .ida-agentic-ready::after {
      display: none;
    }
    [data-ida-agentic-launcher][data-processing="true"] .ida-agentic-ready {
      width: 14px;
      height: 14px;
      border: 2px solid #3b82f6;
      border-top-color: transparent;
      border-radius: 999px;
      animation: ida-agentic-spin 0.8s linear infinite;
    }
    [data-ida-agentic-tooltip] {
      position: fixed;
      right: 24px;
      bottom: 84px;
      z-index: 9999;
      max-width: 260px;
      border: 1px solid #e2e8f0;
      border-radius: 8px;
      background: #ffffff;
      color: #1e293b;
      box-shadow: 0 22px 58px -26px rgba(15, 23, 42, 0.45);
      padding: 12px 40px 12px 16px;
      font: 500 14px/1.45 Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    [data-ida-agentic-tooltip]::after {
      content: "";
      position: absolute;
      right: 32px;
      bottom: -7px;
      width: 12px;
      height: 12px;
      transform: rotate(45deg);
      border-right: 1px solid #e2e8f0;
      border-bottom: 1px solid #e2e8f0;
      background: #ffffff;
    }
    [data-ida-agentic-tooltip] button {
      position: absolute;
      right: 8px;
      top: 8px;
      display: grid;
      place-items: center;
      width: 24px;
      height: 24px;
      border: 0;
      border-radius: 999px;
      background: transparent;
      color: #64748b;
      cursor: pointer;
      font-size: 18px;
      line-height: 1;
    }
    [data-ida-agentic-launcher]:hover {
      transform: translateY(-1px);
      box-shadow: 0 24px 58px -24px rgba(17, 24, 39, 0.54);
    }
    @keyframes ida-agentic-ping {
      75%, 100% {
        transform: scale(1.85);
        opacity: 0;
      }
    }
    @keyframes ida-agentic-spin {
      to {
        transform: rotate(360deg);
      }
    }
  `;

  const button = document.createElement("button");
  button.type = "button";
  button.dataset.idaAgenticLauncher = "true";
  button.dataset.processing = "false";
  button.setAttribute("aria-label", "Open IDA Agentic Core");
  button.innerHTML = '<span class="ida-agentic-monogram">IDA</span><span>Agentic Core</span><span class="ida-agentic-ready" aria-label="Agent ready"></span>';
  button.addEventListener("click", () => {
    const params = new URLSearchParams({ returnUrl: window.location.href });
    const dataset = getActiveDatasetContext();
    if (dataset) {
      params.set("datasetName", dataset.fileName || dataset.datasetId || "uploaded dataset");
      if (dataset.datasetId) params.set("datasetId", dataset.datasetId);
      if (dataset.totalRows) params.set("totalRows", String(dataset.totalRows));
      if (dataset.loadedRows) params.set("loadedRows", String(dataset.loadedRows));
      if (dataset.columns) params.set("columns", dataset.columns);
      if (dataset.numericColumns) params.set("numericColumns", dataset.numericColumns);
      params.set("autoSuggest", "1");
    }
    window.location.assign(`${agentUrl}/?${params.toString()}`);
  });

  document.head.appendChild(style);
  document.body.appendChild(button);

  try {
    if (localStorage.getItem("ida_agent_tooltip_dismissed") !== "true") {
      const tooltip = document.createElement("div");
      tooltip.dataset.idaAgenticTooltip = "true";
      tooltip.innerHTML = 'Your AI agent — click to automate your full workflow <button type="button" aria-label="Dismiss Agentic Core tip">&times;</button>';
      tooltip.querySelector("button")?.addEventListener("click", () => {
        localStorage.setItem("ida_agent_tooltip_dismissed", "true");
        tooltip.remove();
      });
      document.body.appendChild(tooltip);
    }
  } catch (error) {
    return;
  }
})();
