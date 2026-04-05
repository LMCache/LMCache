// MultiProcess Dashboard JavaScript

let baseUrl = "";
let statusData = null;
let autoRefreshTimer = null;

// Initialize after DOM is loaded
window.addEventListener("DOMContentLoaded", () => {
    updateCurrentTime();
    setInterval(updateCurrentTime, 1000);

    // Base URL is the current page origin
    const protocol = window.location.protocol;
    const host = window.location.hostname;
    const port = window.location.port
        || (protocol === "https:" ? "443" : "80");
    baseUrl = protocol + "//" + host + ":" + port;

    // Refresh button
    document.getElementById("refreshAllBtn")
        .addEventListener("click", refreshAll);

    // Auto-refresh toggle
    document.getElementById("autoRefreshToggle")
        .addEventListener("change", toggleAutoRefresh);

    // Tab switching
    document.querySelectorAll(".nav-link").forEach(function(tab) {
        tab.addEventListener("shown.bs.tab", function() {
            if (statusData) {
                renderAll(statusData);
            }
        });
    });

    // JSON search
    document.getElementById("jsonSearchInput")
        .addEventListener("input", filterJson);

    // Initial load
    refreshAll();
});

function updateCurrentTime() {
    var now = new Date();
    var timeStr = now.toLocaleTimeString("en-US", {
        hour12: false,
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit"
    });
    document.getElementById("currentTime").textContent = timeStr;
}

function toggleAutoRefresh() {
    var toggle = document.getElementById("autoRefreshToggle");
    if (toggle.checked) {
        autoRefreshTimer = setInterval(refreshAll, 5000);
    } else {
        if (autoRefreshTimer) {
            clearInterval(autoRefreshTimer);
            autoRefreshTimer = null;
        }
    }
}

async function refreshAll() {
    var statusEl = document.getElementById("connectionStatus");
    try {
        statusEl.textContent = "Refreshing...";
        statusEl.className = "badge bg-warning";

        // Fetch both endpoints in parallel
        var results = await Promise.all([
            fetch(baseUrl + "/api/status"),
            fetch(baseUrl + "/api/healthcheck")
        ]);

        var statusResp = results[0];
        var healthResp = results[1];

        if (!statusResp.ok) {
            throw new Error(
                "Status API returned " + statusResp.status
            );
        }

        statusData = await statusResp.json();
        var healthData = healthResp.ok
            ? await healthResp.json()
            : { status: "unknown" };

        statusData._health = healthData;

        statusEl.textContent = "Connected";
        statusEl.className = "badge bg-success";

        renderAll(statusData);
    } catch (err) {
        console.error("Refresh error:", err);
        statusEl.textContent = "Error: " + err.message;
        statusEl.className = "badge bg-danger";
    }
}

function renderAll(data) {
    renderOverview(data);
    renderStorage(data);
    renderGpuContexts(data);
    renderRawJson(data);
}

// ---------------------------------------------------------------
// Overview Tab
// ---------------------------------------------------------------
function renderOverview(data) {
    var container = document.getElementById("overviewCards");
    var isHealthy = data.is_healthy;
    var healthClass = isHealthy ? "healthy" : "unhealthy";
    var healthText = isHealthy ? "Healthy" : "Unhealthy";

    var sm = data.storage_manager || {};
    var l1 = sm.l1_manager || {};
    var l1Capacity = l1.capacity || 0;
    var l1Used = l1.used || 0;
    var l1Pct = l1Capacity > 0
        ? Math.round((l1Used / l1Capacity) * 100)
        : 0;

    var barColor = l1Pct > 90
        ? "#dc3545"
        : l1Pct > 70
            ? "#ffc107"
            : "#198754";

    var gpuIds = data.registered_gpu_ids || [];
    var sessions = data.active_sessions || 0;
    var engineType = data.engine_type || "Unknown";
    var chunkSize = data.chunk_size || "N/A";
    var hashAlgo = data.hash_algorithm || "N/A";
    var numAdapters = sm.num_l2_adapters || 0;

    var html = "";

    // Row 1: Health + Engine Info
    html += '<div class="col-md-4 mb-3">';
    html += '  <div class="card stat-card">';
    html += '    <div class="card-body">';
    html += '      <div class="stat-label">Health</div>';
    html += '      <div class="mt-2">';
    html += '        <span class="health-dot ' + healthClass;
    html += '"></span>';
    html += '        <span class="fs-4 fw-bold">';
    html += healthText + "</span>";
    html += "      </div>";
    html += "    </div>";
    html += "  </div>";
    html += "</div>";

    html += '<div class="col-md-4 mb-3">';
    html += '  <div class="card stat-card">';
    html += '    <div class="card-body">';
    html += '      <div class="stat-label">Engine Type</div>';
    html += '      <div class="stat-value fs-4">';
    html += engineType + "</div>";
    html += '      <small class="text-muted">Chunk: ';
    html += chunkSize + " | Hash: " + hashAlgo + "</small>";
    html += "    </div>";
    html += "  </div>";
    html += "</div>";

    html += '<div class="col-md-4 mb-3">';
    html += '  <div class="card stat-card">';
    html += '    <div class="card-body">';
    html += '      <div class="stat-label">Active Sessions</div>';
    html += '      <div class="stat-value">' + sessions + "</div>";
    html += "    </div>";
    html += "  </div>";
    html += "</div>";

    // Row 2: GPU + L1 + L2
    html += '<div class="col-md-4 mb-3">';
    html += '  <div class="card stat-card">';
    html += '    <div class="card-body">';
    html += '      <div class="stat-label">GPU Workers</div>';
    html += '      <div class="stat-value">';
    html += gpuIds.length + "</div>";
    html += '      <small class="text-muted">IDs: ';
    html += (gpuIds.length > 0
        ? gpuIds.join(", ")
        : "none") + "</small>";
    html += "    </div>";
    html += "  </div>";
    html += "</div>";

    html += '<div class="col-md-4 mb-3">';
    html += '  <div class="card stat-card">';
    html += '    <div class="card-body">';
    html += '      <div class="stat-label">L1 Cache Usage</div>';
    html += '      <div class="memory-bar mt-2">';
    html += '        <div class="bar-fill" style="width:';
    html += l1Pct + "%;background-color:" + barColor + '">';
    html += l1Pct + "%</div>";
    html += "      </div>";
    html += '      <small class="text-muted mt-1 d-block">';
    html += l1Used + " / " + l1Capacity + " objects</small>";
    html += "    </div>";
    html += "  </div>";
    html += "</div>";

    html += '<div class="col-md-4 mb-3">';
    html += '  <div class="card stat-card">';
    html += '    <div class="card-body">';
    html += '      <div class="stat-label">L2 Adapters</div>';
    html += '      <div class="stat-value">' + numAdapters + "</div>";
    html += "    </div>";
    html += "  </div>";
    html += "</div>";

    container.innerHTML = html;
}

// ---------------------------------------------------------------
// Storage Tab
// ---------------------------------------------------------------
function renderStorage(data) {
    var container = document.getElementById("storageContent");
    var sm = data.storage_manager;
    if (!sm) {
        container.innerHTML = '<div class="alert alert-warning">'
            + "No storage manager data available</div>";
        return;
    }

    var html = "";

    // L1 Manager
    html += renderSection(
        "L1 Manager", sm.l1_manager, "l1-section"
    );

    // Store Controller
    html += renderSection(
        "Store Controller",
        sm.store_controller,
        "store-section"
    );

    // Prefetch Controller
    html += renderSection(
        "Prefetch Controller",
        sm.prefetch_controller,
        "prefetch-section"
    );

    // L1 Eviction Controller
    html += renderSection(
        "L1 Eviction Controller",
        sm.l1_eviction_controller,
        "l1-evict-section"
    );

    // L2 Eviction Controller
    html += renderSection(
        "L2 Eviction Controller",
        sm.l2_eviction_controller,
        "l2-evict-section"
    );

    // L2 Adapters
    var adapters = sm.l2_adapters || [];
    for (var i = 0; i < adapters.length; i++) {
        html += renderSection(
            "L2 Adapter #" + i,
            adapters[i],
            "l2-adapter-" + i
        );
    }

    container.innerHTML = html;

    // Attach toggle listeners
    container.querySelectorAll(".section-header")
        .forEach(function(header) {
            header.addEventListener("click", function() {
                var targetId = this.dataset.target;
                var body = document.getElementById(targetId);
                if (body) {
                    body.classList.toggle("d-none");
                    this.classList.toggle("collapsed");
                }
            });
        });
}

function renderSection(title, obj, sectionId) {
    if (!obj) {
        return "";
    }
    var isHealthy = obj.is_healthy;
    var dotClass = isHealthy === true
        ? "healthy"
        : isHealthy === false
            ? "unhealthy"
            : "";

    var html = "";
    html += '<div class="section-header" data-target="';
    html += sectionId + '-body">';
    html += "  <span>";
    if (dotClass) {
        html += '<span class="health-dot ' + dotClass;
        html += '"></span>';
    }
    html += "    <strong>" + title + "</strong>";
    html += "  </span>";
    html += '  <i class="bi bi-chevron-down toggle-icon"></i>';
    html += "</div>";
    html += '<div id="' + sectionId + '-body" class="mb-3">';
    html += '  <div class="card"><div class="card-body">';
    html += renderObjectTable(obj);
    html += "  </div></div>";
    html += "</div>";
    return html;
}

function renderObjectTable(obj) {
    if (obj === null || obj === undefined) {
        return '<span class="text-muted">N/A</span>';
    }
    if (typeof obj !== "object") {
        return "<span>" + escapeHtml(String(obj)) + "</span>";
    }
    if (Array.isArray(obj)) {
        if (obj.length === 0) {
            return '<span class="text-muted">[]</span>';
        }
        var html = '<ul class="list-group list-group-flush">';
        for (var i = 0; i < obj.length; i++) {
            html += "<li class=\"list-group-item\">";
            html += renderObjectTable(obj[i]);
            html += "</li>";
        }
        html += "</ul>";
        return html;
    }

    var keys = Object.keys(obj);
    if (keys.length === 0) {
        return '<span class="text-muted">{}</span>';
    }

    var html = '<table class="table table-sm table-bordered '
        + 'mb-0"><tbody>';
    for (var k = 0; k < keys.length; k++) {
        var key = keys[k];
        var val = obj[key];
        html += "<tr>";
        html += '<td class="fw-bold" style="width:30%">';
        html += escapeHtml(key) + "</td>";
        html += "<td>";
        if (typeof val === "object" && val !== null) {
            html += renderObjectTable(val);
        } else if (typeof val === "boolean") {
            html += val
                ? '<span class="badge bg-success">true</span>'
                : '<span class="badge bg-danger">false</span>';
        } else {
            html += escapeHtml(String(val));
        }
        html += "</td></tr>";
    }
    html += "</tbody></table>";
    return html;
}

// ---------------------------------------------------------------
// GPU Contexts Tab
// ---------------------------------------------------------------
function renderGpuContexts(data) {
    var container = document.getElementById("gpuContent");
    var meta = data.gpu_context_meta;
    if (!meta || Object.keys(meta).length === 0) {
        container.innerHTML = '<div class="alert alert-info">'
            + "No GPU contexts registered yet. "
            + "Workers will register when they connect."
            + "</div>";
        return;
    }

    var html = "";
    var gpuIds = Object.keys(meta);
    for (var i = 0; i < gpuIds.length; i++) {
        var gpuId = gpuIds[i];
        var ctx = meta[gpuId];
        var layout = ctx.kv_cache_layout || {};

        html += '<div class="card mb-3">';
        html += '  <div class="card-header bg-light">';
        html += '    <i class="bi bi-gpu-card"></i> ';
        html += "    <strong>GPU Worker " + gpuId + "</strong>";
        html += '    <span class="badge bg-primary ms-2">';
        html += (ctx.model_name || "unknown") + "</span>";
        html += "  </div>";
        html += '  <div class="card-body">';

        // Basic info
        html += '  <div class="row mb-3">';
        html += '    <div class="col-md-6">';
        html += '      <table class="table table-sm mb-0">';
        html += "        <tbody>";
        html += "          <tr><td class=\"fw-bold\">";
        html += "Model</td><td>";
        html += escapeHtml(ctx.model_name || "N/A");
        html += "</td></tr>";
        html += "          <tr><td class=\"fw-bold\">";
        html += "World Size</td><td>";
        html += ctx.world_size + "</td></tr>";
        html += "        </tbody></table>";
        html += "    </div>";
        html += '    <div class="col-md-6">';

        if (Object.keys(layout).length > 0) {
            html += '<table class="table table-sm mb-0">';
            html += "  <tbody>";
            var layoutKeys = Object.keys(layout);
            for (var j = 0; j < layoutKeys.length; j++) {
                var lk = layoutKeys[j];
                var lv = layout[lk];
                html += "  <tr><td class=\"fw-bold\">";
                html += escapeHtml(lk) + "</td><td>";
                if (typeof lv === "object" && lv !== null) {
                    html += "<code>";
                    html += escapeHtml(JSON.stringify(lv));
                    html += "</code>";
                } else if (typeof lv === "boolean") {
                    html += lv
                        ? '<span class="badge bg-success">'
                          + "true</span>"
                        : '<span class="badge bg-secondary">'
                          + "false</span>";
                } else {
                    html += escapeHtml(String(lv));
                }
                html += "</td></tr>";
            }
            html += "  </tbody></table>";
        } else {
            html += '<span class="text-muted">';
            html += "No layout info</span>";
        }

        html += "    </div>";
        html += "  </div>";
        html += "  </div>";
        html += "</div>";
    }

    container.innerHTML = html;
}

// ---------------------------------------------------------------
// Raw JSON Tab
// ---------------------------------------------------------------
function renderRawJson(data) {
    var el = document.getElementById("rawJsonContent");
    el.textContent = JSON.stringify(data, null, 2);
}

function filterJson() {
    var input = document.getElementById("jsonSearchInput");
    var el = document.getElementById("rawJsonContent");
    var term = input.value.toLowerCase();

    if (!statusData) {
        return;
    }

    var fullText = JSON.stringify(statusData, null, 2);
    if (!term) {
        el.textContent = fullText;
        return;
    }

    var lines = fullText.split("\n");
    var filtered = lines.filter(function(line) {
        return line.toLowerCase().indexOf(term) !== -1;
    });
    el.textContent = filtered.join("\n") || "No matches found";
}

// ---------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------
function escapeHtml(str) {
    var div = document.createElement("div");
    div.appendChild(document.createTextNode(str));
    return div.innerHTML;
}
