(function () {
  const existing = document.querySelector("[data-ida-agentic-launcher]");
  if (existing) {
    return;
  }

  const scriptOrigin = document.currentScript?.src ? new URL(document.currentScript.src).origin : "";
  const agentUrl = window.IDA_AGENTIC_CORE_URL || scriptOrigin || "http://127.0.0.1:5055";

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
      right: 22px;
      bottom: 22px;
      z-index: 2147483000;
      display: inline-flex;
      align-items: center;
      gap: 10px;
      min-height: 48px;
      border: 1px solid rgba(8, 122, 118, 0.38);
      border-radius: 14px;
      padding: 0 16px 0 10px;
      background:
        linear-gradient(135deg, rgba(255, 255, 255, 0.14), rgba(255, 255, 255, 0)),
        linear-gradient(135deg, #111827, #087a76 58%, #315fce);
      color: #fff;
      box-shadow: 0 18px 44px rgba(17, 24, 39, 0.24);
      font: 800 13px/1.1 Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      cursor: pointer;
      transition: transform 140ms ease, box-shadow 140ms ease;
    }
    [data-ida-agentic-launcher] span:first-child {
      display: grid;
      place-items: center;
      width: 30px;
      height: 30px;
      border-radius: 9px;
      background: rgba(255, 255, 255, 0.18);
      font-size: 10px;
      letter-spacing: 0.04em;
    }
    [data-ida-agentic-launcher]:hover {
      transform: translateY(-1px);
      box-shadow: 0 22px 48px rgba(17, 24, 39, 0.28);
    }
  `;

  const button = document.createElement("button");
  button.type = "button";
  button.dataset.idaAgenticLauncher = "true";
  button.setAttribute("aria-label", "Open IDA Agentic Core");
  button.innerHTML = "<span>IDA</span><span>Agentic Core</span>";
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
})();
