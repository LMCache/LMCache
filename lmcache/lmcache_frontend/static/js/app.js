// Global variables
let currentNode = null;
let allNodes = []; // Store all nodes for filtering

// Initialize after DOM is loaded
window.addEventListener('DOMContentLoaded', () => {
    // Initialize node selector
    loadNodes();

    // Node search input event
    const nodeSearchInput = document.getElementById('nodeSearchInput');
    const nodeDropdown = document.getElementById('nodeDropdown');

    nodeSearchInput.addEventListener('focus', () => {
        filterNodes();
        nodeDropdown.classList.add('show');
    });

    nodeSearchInput.addEventListener('input', () => {
        filterNodes();
    });

    // Close dropdown when clicking outside
    document.addEventListener('click', (e) => {
        if (!nodeSearchInput.contains(e.target) && !nodeDropdown.contains(e.target)) {
            nodeDropdown.classList.remove('show');
        }
    });

    // Tab switching event
    document.querySelectorAll('.nav-link').forEach(tab => {
        tab.addEventListener('shown.bs.tab', () => {
            if (currentNode) {
                refreshActiveTab();
            }
        });
    });

    // Set log level button
    document.getElementById('setLogLevelBtn').addEventListener('click', setLogLevel);

    // Config management buttons
    document.getElementById('getConfigBtn').addEventListener('click', getConfig);
    document.getElementById('setConfigBtn').addEventListener('click', setConfig);

    // Refresh page button
    document.getElementById('refreshPageBtn').addEventListener('click', refreshCurrentPage);

    // Refresh nodes button
    document.getElementById('refreshNodesBtn').addEventListener('click', refreshNodes);

    // Environment search input
    document.getElementById('envSearchInput').addEventListener('input', filterEnvVariables);

    // Environment filter buttons
    document.getElementById('filterNodesBtn').addEventListener('click', filterNodesByEnv);
    document.getElementById('clearFilterBtn').addEventListener('click', clearEnvFilter);

    // Refresh current page function
    function refreshCurrentPage() {
        if (currentNode) {
            refreshActiveTab();
        } else {
            alert('Please select a target node first');
        }
    }

    // Load node management list
    document.getElementById('node-management-tab').addEventListener('shown.bs.tab', () => {
        loadNodeListForManagement();
    });
});

// Load node list
async function loadNodes() {
    try {
        const response = await fetch('/api/nodes');
        const data = await response.json();

        allNodes = data.nodes || [];

        // Initialize dropdown with all nodes
        filterNodes();
    } catch (error) {
        console.error('Failed to load nodes:', error);
    }
}

// Filter nodes based on search input
function filterNodes() {
    const searchInput = document.getElementById('nodeSearchInput');
    const dropdown = document.getElementById('nodeDropdown');
    const searchTerm = searchInput.value.toLowerCase();

    dropdown.innerHTML = '';

    const filteredNodes = allNodes.filter(node => {
        const nodeText = `${node.name} (${node.host}:${node.port})`.toLowerCase();
        return nodeText.includes(searchTerm);
    });

    if (filteredNodes.length === 0) {
        const noResultItem = document.createElement('div');
        noResultItem.className = 'dropdown-item disabled';
        noResultItem.textContent = 'No matching nodes found';
        dropdown.appendChild(noResultItem);
    } else {
        filteredNodes.forEach(node => {
            const item = document.createElement('a');
            item.className = 'dropdown-item';
            item.href = '#';
            item.textContent = `${node.name} (${node.host}:${node.port})`;
            item.addEventListener('click', (e) => {
                e.preventDefault();
                selectNode(node);
            });
            dropdown.appendChild(item);
        });
    }
}

// Select a node
function selectNode(node) {
    const searchInput = document.getElementById('nodeSearchInput');
    const dropdown = document.getElementById('nodeDropdown');

    // Update search input display
    searchInput.value = `${node.name} (${node.host}:${node.port})`;

    // Close dropdown
    dropdown.classList.remove('show');

    // Set current node and refresh the active tab
    currentNode = node;
    document.getElementById('currentNode').textContent =
        `${node.name} (${node.host}:${node.port})`;
    refreshActiveTab();
}

// Refresh the node list from the coordinator-backed registry
async function refreshNodes() {
    await loadNodes();
}

// ==== Node Management (read-only) ====
async function loadNodeListForManagement() {
    try {
        const response = await fetch('/api/nodes');
        const data = await response.json();

        const tableBody = document.getElementById('nodeListBody');
        tableBody.innerHTML = '';

        data.nodes.forEach(node => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${node.name}</td>
                <td>${node.host}</td>
                <td>${node.port}</td>
            `;
            tableBody.appendChild(row);
        });
    } catch (error) {
        console.error('Failed to load nodes for management:', error);
        alert('Failed to load nodes: ' + error.message);
    }
}

// Refresh active tab
function refreshActiveTab() {
    const activeTab = document.querySelector('.tab-pane.active');
    if (!activeTab) return;

    switch (activeTab.id) {
        case 'overview':
            loadOverview();
            break;
        case 'metrics':
            loadMetrics();
            break;
        case 'threads':
            loadThreads();
            break;
        case 'loglevel':
            loadLogLevel();
            break;
        case 'config':
            loadConfig();
            break;
        case 'meta':
            loadMeta();
            break;
        case 'inference':
            loadInference();
            break;
        case 'env':
            loadEnvironment();
            break;
        case 'node-management':
            loadNodeListForManagement();
            break;
    }
}

// Load overview information
async function loadOverview() {
    if (!currentNode) return;

    const contentDiv = document.getElementById('overviewContent');
    contentDiv.innerHTML = '<div class="spinner-border" role="status">'
        + '<span class="visually-hidden">Loading...</span></div>';

    // Try multiProcess /api/status first; fall back to basic info.
    try {
        const statusResp = await fetch(transformPath('api/status'));
        if (statusResp.ok) {
            const statusData = await statusResp.json();
            renderMpOverview(contentDiv, statusData);
            return;
        }
    } catch (_) {
        // not a multiProcess node – fall through
    }

    // Basic overview for inProcess / legacy nodes
    try {
        const response = await fetch(transformPath('version'));
        const versionInfo = await response.text();

        contentDiv.innerHTML = `
            <div class="card">
                <div class="card-body">
                    <h5 class="card-title">Node Information</h5>
                    <p><strong>Name:</strong> ${escapeHtmlStr(currentNode.name)}</p>
                    <p><strong>Host:</strong> ${escapeHtmlStr(currentNode.host)}</p>
                    <p><strong>Port:</strong> ${escapeHtmlStr(String(currentNode.port))}</p>
                </div>
            </div>
            <div class="card mt-3">
                <div class="card-body">
                    <h5 class="card-title">Version Information</h5>
                    <pre>${escapeHtmlStr(versionInfo)}</pre>
                </div>
            </div>
        `;
    } catch (error) {
        contentDiv.innerHTML = '<div class="alert alert-danger">'
            + 'Failed to load overview: '
            + escapeHtmlStr(error.message) + '</div>';
    }
}

// ---------------------------------------------------------------
// MultiProcess Overview renderer (ported from mp_app.js)
// ---------------------------------------------------------------
function renderMpOverview(container, data) {
    var isHealthy = data.is_healthy;
    var healthClass = isHealthy ? "healthy" : "unhealthy";
    var healthText  = isHealthy ? "Healthy" : "Unhealthy";

    var sm = data.storage_manager || {};
    var l1 = sm.l1_manager || {};
    var l1TotalBytes = l1.memory_total_bytes || 0;
    var l1UsedBytes  = l1.memory_used_bytes  || 0;
    var l1Pct = l1TotalBytes > 0
        ? Math.round((l1UsedBytes / l1TotalBytes) * 100) : 0;
    var l1Objects = l1.total_object_count || 0;
    var barColor = l1Pct > 90 ? "#dc3545"
        : l1Pct > 70 ? "#ffc107" : "#198754";

    var gpuIds      = data.registered_gpu_ids || [];
    var sessions    = data.active_sessions    || 0;
    var engineType  = data.engine_type        || "Unknown";
    var chunkSize   = data.chunk_size         || "N/A";
    var hashAlgo    = data.hash_algorithm     || "N/A";
    var numAdapters = sm.num_l2_adapters      || 0;

    var html = '<div class="row">';

    // Row 1: Health / Engine / Sessions
    html += '<div class="col-md-4 mb-3"><div class="card stat-card">'
        + '<div class="card-body"><div class="stat-label">Health</div>'
        + '<div class="mt-2"><span class="health-dot ' + healthClass + '"></span>'
        + '<span class="fs-4 fw-bold">' + healthText + '</span>'
        + '</div></div></div></div>';

    html += '<div class="col-md-4 mb-3"><div class="card stat-card">'
        + '<div class="card-body"><div class="stat-label">Engine Type</div>'
        + '<div class="stat-value fs-4">' + escapeHtmlStr(engineType) + '</div>'
        + '<small class="text-muted">Chunk: ' + escapeHtmlStr(String(chunkSize))
        + ' | Hash: ' + escapeHtmlStr(hashAlgo) + '</small>'
        + '</div></div></div>';

    html += '<div class="col-md-4 mb-3"><div class="card stat-card">'
        + '<div class="card-body"><div class="stat-label">Active Sessions</div>'
        + '<div class="stat-value">' + sessions + '</div>'
        + '</div></div></div>';

    // Row 2: GPU / L1 / L2
    html += '<div class="col-md-4 mb-3"><div class="card stat-card">'
        + '<div class="card-body"><div class="stat-label">GPU Workers</div>'
        + '<div class="stat-value">' + gpuIds.length + '</div>'
        + '<small class="text-muted">IDs: '
        + escapeHtmlStr(gpuIds.length > 0 ? gpuIds.join(", ") : "none")
        + '</small></div></div></div>';

    html += '<div class="col-md-4 mb-3"><div class="card stat-card">'
        + '<div class="card-body"><div class="stat-label">L1 Cache Usage</div>'
        + '<div class="memory-bar mt-2"><div class="bar-fill" style="width:'
        + l1Pct + '%;background-color:' + barColor + '">' + l1Pct + '%</div></div>'
        + '<small class="text-muted mt-1 d-block">'
        + formatBytesStr(l1UsedBytes) + ' / ' + formatBytesStr(l1TotalBytes)
        + ' (' + l1Objects + ' objects)</small>'
        + '</div></div></div>';

    html += '<div class="col-md-4 mb-3"><div class="card stat-card">'
        + '<div class="card-body"><div class="stat-label">L2 Adapters</div>'
        + '<div class="stat-value">' + numAdapters + '</div>'
        + '</div></div></div>';

    // Row 3: Pending & Prefetch
    var pendingLookups  = data.pending_lookup_count    || 0;
    var nextJobId       = data.next_prefetch_job_id    || 0;
    var prefetchJobIds  = data.prefetch_job_ids        || [];
    var pendingReqIds   = data.pending_request_ids     || [];

    html += '<div class="col-12 mt-2 mb-2"><h5 class="text-muted">'
        + '<i class="bi bi-hourglass-split"></i> Pending &amp; Prefetch</h5></div>';

    html += '<div class="col-md-4 mb-3"><div class="card stat-card">'
        + '<div class="card-body"><div class="stat-label">Active Prefetch Jobs</div>'
        + '<div class="stat-value">' + prefetchJobIds.length + '</div>'
        + '<small class="text-muted">next ID: ' + nextJobId;
    if (prefetchJobIds.length > 0) {
        html += ' &middot; IDs: '
            + escapeHtmlStr(prefetchJobIds.slice(0, 5).join(", "));
        if (prefetchJobIds.length > 5) {
            html += ' +' + (prefetchJobIds.length - 5) + ' more';
        }
    }
    html += '</small></div></div></div>';

    html += '<div class="col-md-4 mb-3"><div class="card stat-card">'
        + '<div class="card-body"><div class="stat-label">Pending Lookups</div>'
        + '<div class="stat-value">' + pendingLookups + '</div>'
        + '</div></div></div>';

    html += '<div class="col-md-4 mb-3"><div class="card stat-card">'
        + '<div class="card-body"><div class="stat-label">Pending Requests</div>'
        + '<div class="stat-value">' + pendingReqIds.length + '</div>';
    if (pendingReqIds.length > 0) {
        html += '<small class="text-muted">'
            + escapeHtmlStr(pendingReqIds.slice(0, 3).join(", "));
        if (pendingReqIds.length > 3) {
            html += ' +' + (pendingReqIds.length - 3) + ' more';
        }
        html += '</small>';
    }
    html += '</div></div></div>';

    // Row 4: Periodic Threads summary
    var pt       = data.periodic_threads || {};
    var ptTotal   = pt.total_count   || 0;
    var ptRunning = pt.running_count || 0;
    var ptActive  = pt.active_count  || 0;

    if (ptTotal > 0) {
        html += '<div class="col-12 mt-2 mb-2"><h5 class="text-muted">'
            + '<i class="bi bi-arrow-repeat"></i> Periodic Threads</h5></div>';

        html += '<div class="col-md-4 mb-3"><div class="card stat-card">'
            + '<div class="card-body"><div class="stat-label">Registered</div>'
            + '<div class="stat-value">' + ptTotal + '</div>'
            + '</div></div></div>';

        var runColor = ptRunning === ptTotal ? "#198754" : "#ffc107";
        html += '<div class="col-md-4 mb-3"><div class="card stat-card">'
            + '<div class="card-body"><div class="stat-label">Running</div>'
            + '<div class="stat-value" style="color:' + runColor + '">'
            + ptRunning + ' / ' + ptTotal + '</div>'
            + '</div></div></div>';

        var actColor = ptActive === ptRunning ? "#198754" : "#dc3545";
        html += '<div class="col-md-4 mb-3"><div class="card stat-card">'
            + '<div class="card-body"><div class="stat-label">Active</div>'
            + '<div class="stat-value" style="color:' + actColor + '">'
            + ptActive + ' / ' + ptRunning + '</div>'
            + '</div></div></div>';
    }

    // Row 5: Hit Statistics
    html += renderMpHitStats(data.hit_stats);

    html += '</div>'; // close .row
    container.innerHTML = html;
}

function renderMpHitStats(stats) {
    if (!stats) return "";

    var hitRate  = stats.hit_rate || 0;
    var hitPct   = Math.round(hitRate * 100);
    var hitColor = hitPct >= 80 ? "#198754"
        : hitPct >= 50 ? "#ffc107" : "#dc3545";

    var totalReqs       = stats.total_requests        || 0;
    var totalTokens     = stats.total_tokens          || 0;
    var retrievedTokens = stats.total_retrieved_tokens || 0;

    var html = '<div class="col-12 mt-2 mb-2"><h5 class="text-muted">'
        + '<i class="bi bi-bullseye"></i> Hit Statistics</h5></div>';

    html += '<div class="col-md-4 mb-3"><div class="card stat-card">'
        + '<div class="card-body"><div class="stat-label">GPU Hit Rate</div>'
        + '<div class="stat-value" style="color:' + hitColor + '">'
        + hitPct + '%</div>'
        + '<div class="memory-bar mt-2"><div class="bar-fill" style="width:'
        + hitPct + '%;background-color:' + hitColor + '">' + hitPct + '%</div></div>'
        + '<small class="text-muted mt-1 d-block">'
        + formatTokenCountStr(retrievedTokens) + ' / '
        + formatTokenCountStr(totalTokens) + ' tokens</small>'
        + '</div></div></div>';

    html += '<div class="col-md-4 mb-3"><div class="card stat-card">'
        + '<div class="card-body"><div class="stat-label">Total Requests</div>'
        + '<div class="stat-value">' + totalReqs + '</div>'
        + '<small class="text-muted">'
        + formatTokenCountStr(totalTokens) + ' tokens total</small>'
        + '</div></div></div>';

    html += '<div class="col-md-4 mb-3"><div class="card stat-card">'
        + '<div class="card-body"><div class="stat-label">GPU Retrieved</div>'
        + '<div class="stat-value">'
        + formatTokenCountStr(retrievedTokens) + '</div>'
        + '<small class="text-muted">tokens written to GPU</small>'
        + '</div></div></div>';

    return html;
}

function formatBytesStr(bytes) {
    if (bytes >= 1073741824) return (bytes / 1073741824).toFixed(2) + " GB";
    if (bytes >= 1048576)    return (bytes / 1048576).toFixed(1)    + " MB";
    if (bytes >= 1024)       return (bytes / 1024).toFixed(1)       + " KB";
    return bytes + " B";
}

function formatTokenCountStr(count) {
    if (count >= 1000000) return (count / 1000000).toFixed(1) + "M";
    if (count >= 1000)    return (count / 1000).toFixed(1)    + "K";
    return String(count);
}

function escapeHtmlStr(str) {
    var div = document.createElement("div");
    div.appendChild(document.createTextNode(str));
    return div.innerHTML;
}

// Load metrics information
async function loadMetrics() {
    if (!currentNode) return;

    const contentDiv = document.getElementById('metricsContent');
    contentDiv.textContent = 'Loading...';

    try {
        const response = await fetch(transformPath('metrics'));
        const metrics = await response.text();
        contentDiv.textContent = metrics;
    } catch (error) {
        contentDiv.textContent = `Failed to load metrics: ${error.message}`;
    }
}

// Load threads information
async function loadThreads() {
    if (!currentNode) return;
    const contentDiv = document.getElementById('threadsContent');
    contentDiv.textContent = 'Loading...';

    try {
        const response = await fetch(transformPath('threads'));
        const threads = await response.text();
        contentDiv.textContent = threads;
    } catch (error) {
        contentDiv.textContent = `Failed to load threads: ${error.message}`;
    }
}

// Load log level
async function loadLogLevel() {
    if (!currentNode) return;

    const contentDiv = document.getElementById('logLevelContent');
    const loggerInput = document.getElementById('loggerInput');

    contentDiv.textContent = 'Loading...';
    loggerInput.value = '';

    try {
        const response = await fetch(transformPath('loglevel'));

        const text = await response.text();

        contentDiv.textContent = text;
    } catch (error) {
        contentDiv.textContent = `Failed to load log levels: ${error.message}`;
    }
}

// Set log level
async function setLogLevel() {
    if (!currentNode) return;

    const loggerInput = document.getElementById('loggerInput');
    const levelSelector = document.getElementById('logLevelSelector');

    const loggerName = loggerInput.value.trim();
    const level = levelSelector.value;

    try {
        let url;
        // Encode socket path if needed
        const portOrSocket = encodeURIComponent(encodeURIComponent(currentNode.port));

        if (!level) {
            // Read log level if no level is selected
            url = transformPath('loglevel');
            if (loggerName) {
                url += `?logger_name=${encodeURIComponent(loggerName)}`;
            }
            const response = await fetch(url);
            const text = await response.text();
            alert(text);
        } else {
            // Set log level if level is selected
            if (!loggerName) {
                alert('Please enter a Logger name');
                return;
            }
            url = transformPath('loglevel');
            url += `?logger_name=${encodeURIComponent(loggerName)}&level=${level}`;
            const response = await fetch(url, { method: 'GET' });

            const text = await response.text();
            alert(text);

            if (response.ok) {
                loadLogLevel();
            }
        }
    } catch (error) {
        alert(`Failed to manage log level: ${error.message}`);
    }
}

// Load configuration
async function loadConfig() {
    if (!currentNode) return;

    const contentDiv = document.getElementById('configContent');
    const configKeyInput = document.getElementById('configKeyInput');
    const configValueInput = document.getElementById('configValueInput');

    contentDiv.textContent = 'Loading...';
    configKeyInput.value = '';
    configValueInput.value = '';

    try {
        const response = await fetch(transformPath('conf'));
        const text = await response.text();
        contentDiv.textContent = text;
    } catch (error) {
        contentDiv.textContent = `Failed to load configuration: ${error.message}`;
    }
}

// Get configuration
async function getConfig() {
    if (!currentNode) return;

    const configKeyInput = document.getElementById('configKeyInput');
    const configKey = configKeyInput.value.trim();

    try {
        let url = transformPath('conf');
        if (configKey) {
            url += `?key=${encodeURIComponent(configKey)}`;
        }
        const response = await fetch(url);
        const text = await response.text();
        alert(text);
    } catch (error) {
        alert(`Failed to get configuration: ${error.message}`);
    }
}

// Set configuration
async function setConfig() {
    if (!currentNode) return;

    const configKeyInput = document.getElementById('configKeyInput');
    const configValueInput = document.getElementById('configValueInput');

    const configKey = configKeyInput.value.trim();
    const configValue = configValueInput.value.trim();

    if (!configKey) {
        alert('Please enter a configuration key');
        return;
    }

    if (!configValue) {
        alert('Please enter a configuration value');
        return;
    }

    try {
        const url = transformPath('conf') + `?key=${encodeURIComponent(configKey)}&value=${encodeURIComponent(configValue)}`;
        const response = await fetch(url, { method: 'GET' });
        const text = await response.text();
        alert(text);

        if (response.ok) {
            loadConfig();
        }
    } catch (error) {
        alert(`Failed to set configuration: ${error.message}`);
    }
}

// Load meta information
async function loadMeta() {
    if (!currentNode) return;

    const contentDiv = document.getElementById('metaContent');
    contentDiv.textContent = 'Loading...';

    try {
        const response = await fetch(transformPath('meta'));
        const text = await response.text();
        contentDiv.textContent = text;
    } catch (error) {
        contentDiv.textContent = `Failed to load meta information: ${error.message}`;
    }
}

// Load inference information
async function loadInference() {
    if (!currentNode) return;

    const contentDiv = document.getElementById('inferenceContent');
    contentDiv.textContent = 'Loading...';

    try {
        const response = await fetch(transformPath('inference_info'));
        const text = await response.text();
        contentDiv.textContent = text;
    } catch (error) {
        contentDiv.textContent = `Failed to load inference information: ${error.message}`;
    }
}

// Load environment variables
let envVariablesData = null; // Store as object instead of string
async function loadEnvironment() {
    if (!currentNode) return;

    const contentDiv = document.getElementById('envContent');
    const searchInput = document.getElementById('envSearchInput');
    contentDiv.textContent = 'Loading...';
    searchInput.value = '';

    try {
        const response = await fetch(transformPath('env'));
        const text = await response.text();
        
        // Parse JSON and format for display
        try {
            envVariablesData = JSON.parse(text);
            // Format as KEY=VALUE lines for display
            const formattedText = Object.entries(envVariablesData)
                .map(([key, value]) => `${key}=${value}`)
                .join('\n');
            contentDiv.textContent = formattedText;
        } catch (e) {
            // Fallback to plain text if not JSON
            envVariablesData = text;
            contentDiv.textContent = text;
        }
    } catch (error) {
        contentDiv.textContent = `Failed to load environment variables: ${error.message}`;
        envVariablesData = null;
    }
}

// Filter environment variables based on search input
function filterEnvVariables() {
    const searchInput = document.getElementById('envSearchInput');
    const contentDiv = document.getElementById('envContent');
    const searchTerm = searchInput.value.toLowerCase();

    if (!envVariablesData) {
        return;
    }

    if (!searchTerm) {
        // Show all variables
        if (typeof envVariablesData === 'object') {
            const formattedText = Object.entries(envVariablesData)
                .map(([key, value]) => `${key}=${value}`)
                .join('\n');
            contentDiv.textContent = formattedText;
        } else {
            contentDiv.textContent = envVariablesData;
        }
        return;
    }

    // Filter based on search term
    if (typeof envVariablesData === 'object') {
        const filteredEntries = Object.entries(envVariablesData).filter(([key, value]) => {
            const line = `${key}=${value}`;
            return line.toLowerCase().includes(searchTerm);
        });
        const formattedText = filteredEntries
            .map(([key, value]) => `${key}=${value}`)
            .join('\n');
        contentDiv.textContent = formattedText;
    } else {
        // Fallback for plain text
        const lines = envVariablesData.split('\n');
        const filteredLines = lines.filter(line => 
            line.toLowerCase().includes(searchTerm)
        );
        contentDiv.textContent = filteredLines.join('\n');
    }
}

// Clear all tab contents
function clearAllTabs() {
    document.getElementById('overviewContent').innerHTML = 'Please select a target node first';
    document.getElementById('metricsContent').textContent = 'Please select a target node first';
    document.getElementById('threadsContent').textContent = 'Please select a target node first';
    document.getElementById('logLevelContent').textContent = 'Please select a target node first';
    document.getElementById('configContent').textContent = 'Please select a target node first';
    document.getElementById('metaContent').textContent = 'Please select a target node first';
    document.getElementById('inferenceContent').textContent = 'Please select a target node first';
    document.getElementById('envContent').textContent = 'Please select a target node first';
    document.getElementById('loggerInput').value = '';
    document.getElementById('configKeyInput').value = '';
    document.getElementById('configValueInput').value = '';
    document.getElementById('envSearchInput').value = '';
    envVariablesData = null;
}

function transformPath(path) {
    if (!currentNode) return path;
    return `/proxy2/${currentNode.name}/${path}`;
}

// Filter nodes by environment variable and keep only matching nodes
async function filterNodesByEnv() {
    const envFilter = document.getElementById('envFilterInput').value.trim();
    if (!envFilter) {
        alert('Please enter an environment variable filter (e.g., TAG or TAG=TaijiDS)');
        return;
    }

    // Parse filter condition
    let filterKey, filterValue;
    const filterParts = envFilter.split('=');
    if (filterParts.length === 1) {
        filterKey = filterParts[0].trim();
        filterValue = null;
    } else if (filterParts.length === 2) {
        filterKey = filterParts[0].trim();
        filterValue = filterParts[1].trim();
    } else {
        alert('Invalid filter format. Please use KEY or KEY=VALUE format (e.g., TAG or TAG=TaijiDS)');
        return;
    }

    const searchInput = document.getElementById('nodeSearchInput');
    const dropdown = document.getElementById('nodeDropdown');
    searchInput.value = 'Filtering nodes...';
    searchInput.disabled = true;
    dropdown.classList.remove('show');

    try {
        const response = await fetch('/api/nodes');
        const data = await response.json();
        const nodes = data.nodes || [];

        const checks = nodes.map(async (node) => {
            try {
                const envResponse = await fetch(`/proxy2/${node.name}/env`);
                if (!envResponse.ok) return null;
                const envData = JSON.parse(await envResponse.text());
                if (filterValue === null) {
                    return (filterKey in envData) ? node : null;
                }
                return (filterKey in envData && envData[filterKey] === filterValue) ? node : null;
            } catch (error) {
                console.error(`Error checking node ${node.name}:`, error);
                return null;
            }
        });

        const matchingNodes = (await Promise.all(checks)).filter(n => n !== null);

        searchInput.value = '';
        searchInput.disabled = false;

        const filterDesc = filterValue === null ? filterKey : `${filterKey}=${filterValue}`;
        if (matchingNodes.length === 0) {
            alert(`No nodes found with ${filterDesc}`);
            loadNodes();
        } else {
            allNodes = matchingNodes;
            filterNodes();
            alert(`Found ${matchingNodes.length} matching node(s) for ${filterDesc}`);
        }
    } catch (error) {
        console.error('Failed to filter nodes:', error);
        alert('Failed to filter nodes: ' + error.message);
        searchInput.disabled = false;
        loadNodes();
    }
}

// Clear environment filter
function clearEnvFilter() {
    document.getElementById('envFilterInput').value = '';
    loadNodes();
}
