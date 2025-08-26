// static/js/main.js
document.addEventListener("DOMContentLoaded", () => {
  const fileInput = document.getElementById("pdbFile");
  const runBtn    = document.getElementById("runBtn");
  const status    = document.getElementById("status");
  const ramaFrame = document.getElementById("ramaFrame");
  const tbody     = document.getElementById("topBody");
const loadLocalBtn  = document.getElementById("loadLocalOps");
const localOpsBody  = document.getElementById("localOpsBody");
const localOpsPathP = document.getElementById("localOpsPath");

if (loadLocalBtn) {
  loadLocalBtn.addEventListener("click", async () => {
    localOpsBody.innerHTML = "<tr><td colspan='5'>Loading…</td></tr>";
    try {
      const r = await fetch("/api/local_ops_top");
      const js = await r.json();
      if (!r.ok || js.error) throw new Error(js.error || ("HTTP " + r.status));
      localOpsPathP.textContent = "file: " + js.path;
      localOpsBody.innerHTML = js.rows.map(x => `
        <tr>
          <td>${x.resid}</td><td>${x.resname}</td>
          <td>${(+x.local_score_med).toFixed(3)}</td>
          <td>${(+x.rama_disallowed_pct).toFixed(1)}%</td>
          <td>${x.n_frames}</td>
        </tr>`).join("");
    } catch (e) {
      localOpsBody.innerHTML = `<tr><td colspan="5">Error: ${e.message}</td></tr>`;
    }
  });
}
  async function refreshTop() {
    try {
      const r = await fetch("/api/top_anomalies");
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const rows = await r.json();
      tbody.innerHTML = rows.map((row) => `
        <tr>
          <td>${row.resid}</td>
          <td>${(+row.mean_if).toFixed(3)}</td>
          <td>${(+row.pct_disallowed).toFixed(1)}%</td>
          <td>${row.n_frames}</td>
        </tr>
      `).join("");
    } catch (e) {
      console.error(e);
      tbody.innerHTML = `<tr><td colspan="4">Failed to load top anomalies: ${e.message}</td></tr>`;
    }
  }

  runBtn.addEventListener("click", async () => {
    if (!fileInput.files.length) {
      status.textContent = "Pick a .pdb first";
      return;
    }
    status.textContent = "Running…";

    const fd = new FormData();
    fd.append("file", fileInput.files[0]);

    try {
      const resp = await fetch("/api/score_pdb", { method: "POST", body: fd });
      const js   = await resp.json();
      if (!resp.ok || js.error) throw new Error(js.error || `HTTP ${resp.status}`);

      status.textContent = "Done.";
      ramaFrame.src = "/rama?ts=" + Date.now();  // refresh φ/ψ plot
      await refreshTop();                         // refresh table
    } catch (e) {
      console.error(e);
      status.textContent = "Error: " + e.message;
    }
  });

  // Populate table on load if rollup.csv exists
  refreshTop();
});
