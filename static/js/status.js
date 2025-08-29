// static/js/status.js
async function getJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url} -> ${r.status}`);
  return r.json();
}

function badge(el, ok) {
  el.className = 'badge ' + (ok ? 'ok' : 'bad');
  el.textContent = ok ? 'OK' : 'MISSING';
}

function tableFromRows(rows) {
  if (!rows || !rows.length) return '';
  const cols = Object.keys(rows[0]);
  let th = '<tr>' + cols.map(c => `<th>${c}</th>`).join('') + '</tr>';
  let tb = rows.slice(0, 10).map(r =>
    '<tr>' + cols.map(c => `<td>${r[c] ?? ''}</td>`).join('') + '</tr>'
  ).join('');
  return `<table>${th}${tb}</table>`;
}

// --- Health panel ---
(async () => {
  try {
    const h = await j('/api/health').catch(() => ({}));
    const exists = (h && h.exists) ? h.exists : {};
    const items = [];

    if (exists.rama_html) items.push(`Rama HTML:  ${exists.rama_html}`);
    if (exists.rollup_csv) items.push(`IF rollup.csv:  ${exists.rollup_csv}`);
    if (h && h.deep && h.deep.run_dir) items.push(`Deep dir: ${h.deep.run_dir}`);

    $('#files').innerHTML = `<ul>${items.map(m => `<li>${m}</li>`).join('')}</ul>`;
  } catch (e) {
    $('#files').textContent = `Health error: ${e.message || e}`;
  }
})();
  // Isolation Forest (Phase 1)
  const ifBadge = document.getElementById('if-badge');
  const ifPath = document.getElementById('if-path');
  const ifTable = document.getElementById('if-table');
  try {
    const j = await getJSON('/api/top_anomalies');
    badge(ifBadge, true);
    ifPath.innerHTML = `path: <code>${j.path || ''}</code>`;
    ifTable.innerHTML = tableFromRows(j.rows);
  } catch (e) {
    badge(ifBadge, false);
    ifPath.textContent = e.message.includes('404')
      ? 'rollup.csv not found (this table will be empty until a Phase-1 run writes it)'
      : ('error: ' + e.message);
    ifTable.innerHTML = '';
  }

  // Local-Ops (domain rules)
  const loBadge = document.getElementById('local-badge');
  const loPath = document.getElementById('local-path');
  const loTable = document.getElementById('local-table');
  try {
    const j = await getJSON('/api/local_ops_top');
    badge(loBadge, true);
    loPath.innerHTML = `path: <code>${j.path || ''}</code>`;
    loTable.innerHTML = tableFromRows(j.rows);
  } catch (e) {
    badge(loBadge, false);
    loPath.textContent = 'error: ' + e.message;
    loTable.innerHTML = '';
  }

  // Deep (AE + clustering)
  const dBadge = document.getElementById('deep-badge');
  const dRun = document.getElementById('deep-run');
  const dLinks = document.getElementById('deep-links');
  const dImgs = document.getElementById('deep-images');
  const dHeads = document.getElementById('deep-heads');
  try {
    const j = await getJSON('/api/deep_latest');
    badge(dBadge, true);
    dRun.textContent = 'run: ' + (j.run_dir || '—');

    const links = [];
    if (j.per_window) links.push(`<a href="/file/${j.per_window}" target="_blank">per_window.csv</a>`);
    if (j.latent?.path) links.push(`<a href="/file/${j.latent.path}" target="_blank">latent.csv</a>`);
    if (j.recon?.path) links.push(`<a href="/file/${j.recon.path}" target="_blank">recon_error.csv</a>`);
    if (j.clusters?.path) links.push(`<a href="/file/${j.clusters.path}" target="_blank">latent_clusters.csv</a>`);
    if (j.anomalies?.path) links.push(`<a href="/file/${j.anomalies.path}" target="_blank">anomalies.csv</a>`);
    if (j.hybrid?.path) links.push(`<a href="/file/${j.hybrid.path}" target="_blank">hybrid_scores.csv</a>`);
    if (j.hotspots?.path) links.push(`<a href="/file/${j.hotspots.path}" target="_blank">residue_hotspots.csv</a>`);
    dLinks.innerHTML = links.length ? ('Links: ' + links.join(' · ')) : 'No deep CSVs found.';

    dImgs.innerHTML = (j.images || []).map(src => `<div><img class="thumb" src="${src}"></div>`).join('');

    const headBlocks = [];
    function headBlock(name, obj) {
      if (!obj) return;
      headBlocks.push(`<h4>${name}</h4>${tableFromRows(obj.head || [])}`);
    }
    headBlock('latent.csv (head)', j.latent);
    headBlock('recon_error.csv (head)', j.recon);
    headBlock('latent_clusters.csv (head)', j.clusters);
    headBlock('anomalies.csv (head)', j.anomalies);
    headBlock('hybrid_scores.csv (head)', j.hybrid);
    headBlock('residue_hotspots.csv (head)', j.hotspots);
    dHeads.innerHTML = headBlocks.join('');
  } catch (e) {
    badge(dBadge, false);
    dRun.textContent = 'deep_latest error: ' + e.message;
    dLinks.textContent = '';
    dImgs.innerHTML = '';
    dHeads.innerHTML = '';
  }
})();
