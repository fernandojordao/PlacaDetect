const state = {
  status: "all",
  page: 1,
  pageSize: 60,
  polling: null,
};

const STATUS_LABELS = {
  pending: "Pendente",
  processing: "Processando",
  success: "Sucesso",
  no_plate: "Sem placa",
  error: "Erro",
};

const STATUS_ICON = {
  pending: "⏳",
  processing: "⚙️",
  success: "✅",
  no_plate: "⚠️",
  error: "❌",
};

async function api(path, options) {
  const res = await fetch(path, options);
  if (!res.ok) {
    let msg = res.statusText;
    try {
      const data = await res.json();
      msg = data.detail || msg;
    } catch (e) {}
    throw new Error(msg);
  }
  const ct = res.headers.get("content-type") || "";
  return ct.includes("application/json") ? res.json() : res;
}

// -------------------------------------------------------------- stats ---

async function refreshStats() {
  const stats = await api("/api/stats");
  const el = document.getElementById("stats");
  el.innerHTML = `
    <div class="stat"><b>${stats.total}</b>total</div>
    <div class="stat pending"><b>${stats.pending}</b>pendente</div>
    <div class="stat success"><b>${stats.success}</b>sucesso</div>
    <div class="stat no_plate"><b>${stats.no_plate}</b>sem placa</div>
    <div class="stat error"><b>${stats.error}</b>erro</div>
  `;
  return stats;
}

// -------------------------------------------------------------- tabs ----

document.querySelectorAll(".tab-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".tab-content").forEach((c) => c.classList.add("hidden"));
    btn.classList.add("active");
    document.getElementById(`tab-${btn.dataset.tab}`).classList.remove("hidden");
  });
});

// ----------------------------------------------------------- import -----

document.getElementById("btn-scan-folder").addEventListener("click", async () => {
  const folder = document.getElementById("folder-path").value.trim();
  const recursive = document.getElementById("folder-recursive").checked;
  const resultEl = document.getElementById("folder-result");
  if (!folder) {
    resultEl.textContent = "Informe um caminho de pasta.";
    return;
  }
  resultEl.textContent = "Escaneando...";
  try {
    const res = await api("/api/scan-folder", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ folder, recursive }),
    });
    resultEl.textContent = `Encontradas ${res.found} imagens, ${res.added} novas adicionadas (${res.skipped_existing} já estavam na lista).`;
    await refreshAll();
  } catch (e) {
    resultEl.textContent = `Erro: ${e.message}`;
  }
});

const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("file-input");

dropzone.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropzone.classList.add("drag");
});
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("drag"));
dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.classList.remove("drag");
  uploadFiles(e.dataTransfer.files);
});
fileInput.addEventListener("change", () => uploadFiles(fileInput.files));

async function uploadFiles(fileList) {
  if (!fileList || fileList.length === 0) return;
  const resultEl = document.getElementById("upload-result");
  const formData = new FormData();
  for (const f of fileList) formData.append("files", f);
  resultEl.textContent = `Enviando ${fileList.length} arquivo(s)...`;
  try {
    const res = await api("/api/upload", { method: "POST", body: formData });
    resultEl.textContent = `${res.added} foto(s) adicionada(s).` +
      (res.errors.length ? ` ${res.errors.length} com erro.` : "");
    await refreshAll();
  } catch (e) {
    resultEl.textContent = `Erro: ${e.message}`;
  }
}

// ---------------------------------------------------------- processing --

document.getElementById("btn-process").addEventListener("click", () => startProcess("pending"));
document.getElementById("btn-reprocess-errors").addEventListener("click", () => startProcess("errors"));
document.getElementById("btn-cancel").addEventListener("click", async () => {
  await api("/api/process/cancel", { method: "POST" });
});

document.getElementById("redaction-style").addEventListener("change", saveSettings);
document.getElementById("conf-thresh").addEventListener("change", saveSettings);

async function saveSettings() {
  await api("/api/settings", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      redaction_style: document.getElementById("redaction-style").value,
      conf_thresh: parseFloat(document.getElementById("conf-thresh").value),
    }),
  });
}

async function startProcess(scope) {
  await saveSettings();
  try {
    const res = await api("/api/process", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ scope }),
    });
    if (!res.started) {
      alert(res.reason || "Nada para processar.");
      return;
    }
    startPolling();
  } catch (e) {
    alert(`Erro ao iniciar processamento: ${e.message}`);
  }
}

function startPolling() {
  document.getElementById("progress-wrap").classList.remove("hidden");
  document.getElementById("btn-cancel").classList.remove("hidden");
  document.getElementById("btn-process").disabled = true;

  if (state.polling) clearInterval(state.polling);
  state.polling = setInterval(pollStatus, 900);
  pollStatus();
}

async function pollStatus() {
  const res = await api("/api/process/status");
  const pct = res.total > 0 ? Math.round((res.done / res.total) * 100) : 0;
  document.getElementById("progress-fill").style.width = `${pct}%`;
  document.getElementById("progress-text").textContent =
    `${res.done} / ${res.total} processadas (${pct}%)`;

  await refreshStats();
  if (state.status !== "pending") await loadGallery();

  if (!res.active) {
    clearInterval(state.polling);
    state.polling = null;
    document.getElementById("btn-cancel").classList.add("hidden");
    document.getElementById("btn-process").disabled = false;
    document.getElementById("progress-text").textContent += " — concluído.";
    await refreshAll();
  }
}

// -------------------------------------------------------------- filters -

document.querySelectorAll(".filter-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".filter-btn").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    state.status = btn.dataset.status;
    state.page = 1;
    loadGallery();
  });
});

document.getElementById("btn-refresh").addEventListener("click", refreshAll);
document.getElementById("btn-download-success").addEventListener("click", () => {
  window.location = `/api/download-zip?status=success`;
});

// -------------------------------------------------------------- gallery -

async function loadGallery() {
  const params = new URLSearchParams({
    page: state.page,
    page_size: state.pageSize,
  });
  if (state.status !== "all") params.set("status", state.status);

  const res = await api(`/api/photos?${params.toString()}`);
  const gallery = document.getElementById("gallery");

  if (res.photos.length === 0) {
    gallery.innerHTML = `<div class="empty-state">Nenhuma foto nesse filtro ainda. Importe fotos acima e clique em "Processar pendentes".</div>`;
  } else {
    gallery.innerHTML = res.photos.map(cardHtml).join("");
    gallery.querySelectorAll(".card").forEach((card) => {
      card.addEventListener("click", () => openModal(parseInt(card.dataset.id, 10)));
    });
  }

  renderPagination(res.total);
}

function cardHtml(p) {
  const thumb = p.thumb_output_url || p.thumb_input_url;
  const img = thumb
    ? `<img src="${thumb}" loading="lazy" alt="${p.filename}" />`
    : `<div style="height:130px;display:flex;align-items:center;justify-content:center;color:#555">sem preview</div>`;
  return `
    <div class="card" data-id="${p.id}">
      ${img}
      <div class="card-body">
        <div class="filename" title="${p.filename}">${p.filename}</div>
        <span class="badge ${p.status}">${STATUS_ICON[p.status] || ""} ${STATUS_LABELS[p.status] || p.status}</span>
      </div>
    </div>`;
}

function renderPagination(total) {
  const pages = Math.max(1, Math.ceil(total / state.pageSize));
  const el = document.getElementById("pagination");
  if (pages <= 1) {
    el.innerHTML = "";
    return;
  }
  let html = "";
  for (let i = 1; i <= pages; i++) {
    html += `<button class="${i === state.page ? "active" : ""}" data-page="${i}">${i}</button>`;
  }
  el.innerHTML = html;
  el.querySelectorAll("button").forEach((btn) => {
    btn.addEventListener("click", () => {
      state.page = parseInt(btn.dataset.page, 10);
      loadGallery();
    });
  });
}

// --------------------------------------------------------------- modal --

async function openModal(id) {
  const p = await api(`/api/photos/${id}`);
  const modal = document.getElementById("modal");
  const body = document.getElementById("modal-body");

  const detectionsHtml = p.detections.length
    ? `<ul>${p.detections.map((d) => `<li>placa detectada com ${(d.confidence * 100).toFixed(1)}% de confiança</li>`).join("")}</ul>`
    : `<p>Nenhuma placa detectada nesta imagem.</p>`;

  const errorHtml = p.error_message ? `<p style="color:var(--error)">Erro: ${p.error_message}</p>` : "";

  body.innerHTML = `
    <h2>${p.filename}</h2>
    <span class="badge ${p.status}">${STATUS_ICON[p.status] || ""} ${STATUS_LABELS[p.status] || p.status}</span>
    ${errorHtml}
    <div class="compare">
      <figure>
        <img src="${p.thumb_input_url || ""}" alt="original" />
        <figcaption>Original (miniatura)</figcaption>
      </figure>
      <figure>
        <img src="${p.thumb_output_url || p.thumb_input_url || ""}" alt="processada" />
        <figcaption>Processada</figcaption>
      </figure>
    </div>
    <div class="detections-list">${detectionsHtml}</div>
    <div class="modal-actions">
      ${p.output_url ? `<button class="primary" onclick="window.location='/api/photos/${p.id}/download'">Baixar em resolução original</button>` : ""}
    </div>
  `;
  modal.classList.remove("hidden");
}

document.getElementById("modal-close").addEventListener("click", () => {
  document.getElementById("modal").classList.add("hidden");
});
document.getElementById("modal").addEventListener("click", (e) => {
  if (e.target.id === "modal") document.getElementById("modal").classList.add("hidden");
});

// -------------------------------------------------------------- init ----

async function refreshAll() {
  await refreshStats();
  await loadGallery();
}

async function loadSettingsIntoUI() {
  const s = await api("/api/settings");
  document.getElementById("redaction-style").value = s.redaction_style;
  document.getElementById("conf-thresh").value = String(s.conf_thresh);
}

(async function init() {
  await loadSettingsIntoUI();
  await refreshAll();
  const status = await api("/api/process/status");
  if (status.active) startPolling();
})();
