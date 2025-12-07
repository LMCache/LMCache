// Controller Dashboard JavaScript
// Global variables
let controllerBaseUrl = "";
let isConnected = false;
let currentInstances = [];
let currentWorkers = [];
let currentKeyPool = [];
let envVariablesData = null;

// Initialize after DOM is loaded
window.addEventListener('DOMContentLoaded', () => {
    // Initialize current time display
    updateCurrentTime();
    setInterval(updateCurrentTime, 1000);

    // Connect button event
    document.getElementById('connectControllerBtn').addEventListener('click', connectToController);

    // Refresh all button
    document.getElementById('refreshAllBtn').addEventListener('click', refreshAllData);

    // Tab switching event
    document.querySelectorAll('.nav-link').forEach(tab => {
        tab.addEventListener('shown.bs.tab', (event) => {
            if (isConnected) {
                const tabId = event.target.getAttribute('data-bs-target').replace('#', '');
                loadTabData(tabId);
            }
        });
    });

    // Instance management
    document.getElementById('refreshInstancesBtn').addEventListener('click', loadInstances);
    document.getElementById('saveInstanceBtn').addEventListener('click', addInstance);

    // Worker management
    document.getElementById('refreshWorkersBtn').addEventListener('click', loadWorkers);
    document.getElementById('instanceFilter').addEventListener('change', loadWorkers);

    // Key pool management
    document.getElementById('refreshKeyPoolBtn').addEventListener('click', loadKeyPool);
    document.getElementById('clearKeyPoolBtn').addEventListener('click', clearKeyPool);
    document.getElementById('keySearchInput').addEventListener('input', filterKeyPool);

    // Metrics
    document.getElementById('refreshMetricsBtn').addEventListener('click', loadMetrics);

    // Log level management
    document.getElementById('setLogLevelBtn').addEventListener('click', setLogLevel);

    // Config management
    document.getElementById('getConfigBtn').addEventListener('click', getConfig);
    document.getElementById('setConfigBtn').addEventListener('click', setConfig);

    // Threads
    document.getElementById('refreshThreadsBtn').addEventListener('click', loadThreads);

    // Environment
    document.getElementById('envSearchInput').addEventListener('input', filterEnvVariables);

    // Script execution
    document.getElementById('executeScriptBtn').addEventListener('click', executeScript);
    document.getElementById('uploadScriptBtn').addEventListener('click', uploadScript);

    // Auto-connect if parameters are set
    const urlParams = new URLSearchParams(window.location.search);
    const autoHost = urlParams.get('host') || 'localhost';
    const autoPort = urlParams.get('port') || '9000';
    
    if (urlParams.has('host') || urlParams.has('port')) {
        document.getElementById('controllerHostInput').value = autoHost;
        document.getElementById('controllerPortInput').value = autoPort;
        setTimeout(() => connectToController(), 1000);
    }
});

// Update current time display
function updateCurrentTime() {
    const now = new Date();
    const timeString = now.toLocaleTimeString('en-US', {
        hour12: false,
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit'
    });
    document.getElementById('currentTime').textContent = timeString;
}

// Connect to Controller
async function connectToController() {
    const host = document.getElementById('controllerHostInput').value.trim();
    const port = document.getElementById('controllerPortInput').value.trim();
    
    if (!host || !port) {
        alert('Please enter controller host and port');
        return;
    }

    controllerBaseUrl = `http://${host}:${port}`;
    const statusElement = document.getElementById('connectionStatus');
    
    try {
        statusElement.textContent = 'Connecting...';
        statusElement.className = 'badge bg-warning';
        
        // Test connection with a simple health check
        const response = await fetch(`${controllerBaseUrl}/health`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ instance_id: 'test' })
        });
        
        if (response.ok) {
            isConnected = true;
            statusElement.textContent = 'Connected';
            statusElement.className = 'badge bg-success';
            
            // Load initial data
            loadOverview();
            loadInstances();
            loadWorkers();
            loadKeyPool();
            
            // Update URL with connection parameters
            const newUrl = new URL(window.location);
            newUrl.searchParams.set('host', host);
            newUrl.searchParams.set('port', port);
            window.history.replaceState({}, '', newUrl);
            
        } else {
            throw new Error('Connection failed');
        }
    } catch (error) {
        console.error('Connection error:', error);
        statusElement.textContent = 'Connection Failed';
        statusElement.className = 'badge bg-danger';
        isConnected = false;
        alert('Failed to connect to controller: ' + error.message);
    }
}

// Refresh all data
async function refreshAllData() {
    if (!isConnected) {
        alert('Please connect to controller first');
        return;
    }

    const activeTab = document.querySelector('.tab-pane.active').id;
    loadTabData(activeTab);
}

// Load data for active tab
function loadTabData(tabId) {
    if (!isConnected) return;

    switch (tabId) {
        case 'overview':
            loadOverview();
            break;
        case 'instances':
            loadInstances();
            break;
        case 'workers':
            loadWorkers();
            break;
        case 'keypool':
            loadKeyPool();
            break;
        case 'metrics':
            loadMetrics();
            break;
        case 'loglevel':
            loadLogLevel();
            break;
        case 'config':
            loadConfig();
            break;
        case 'threads':
            loadThreads();
            break;
        case 'env':
            loadEnvironment();
            break;
        case 'script':
            // Nothing to load for script tab
            break;
    }
}

// Load overview data
async function loadOverview() {
    if (!isConnected) return;

    const systemStatusElement = document.getElementById('systemStatus');
    const quickStatsElement = document.getElementById('quickStats');
    const recentActivitiesElement = document.getElementById('recentActivities');

    try {
        // Load system status
        const healthResponse = await fetch(`${controllerBaseUrl}/health`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ instance_id: 'system' })
        });

        if (healthResponse.ok) {
            const healthData = await healthResponse.json();
            systemStatusElement.innerHTML = `
                <div class="text-success">
                    <i class="bi bi-check-circle-fill fs-1"></i>
                    <p class="mt-2">Controller is running</p>
                    <small class="text-muted">Event ID: ${healthData.event_id}</small>
                </div>
            `;
        }

        // Load quick stats (instance count, worker count, key count)
        const instancesResponse = await fetch(`${controllerBaseUrl}/query_worker_info`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ instance_id: 'all' })
        });

        if (instancesResponse.ok) {
            const instancesData = await instancesResponse.json();
            const instanceCount = new Set(instancesData.worker_infos.map(w => w.instance_id)).size;
            const workerCount = instancesData.worker_infos.length;
            
            // For key count, we would need to implement a separate endpoint
            quickStatsElement.innerHTML = `
                <div class="row">
                    <div class="col-6">
                        <div class="card bg-light mb-2">
                            <div class="card-body p-2">
                                <h6 class="card-title mb-0">Instances</h6>
                                <h3 class="mb-0">${instanceCount}</h3>
                            </div>
                        </div>
                    </div>
                    <div class="col-6">
                        <div class="card bg-light mb-2">
                            <div class="card-body p-2">
                                <h6 class="card-title mb-0">Workers</h6>
                                <h3 class="mb-0">${workerCount}</h3>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        }

        // Recent activities (placeholder)
        recentActivitiesElement.innerHTML = `
            <div class="list-group">
                <div class="list-group-item">
                    <small class="text-muted">Just now</small>
                    <p class="mb-1">Controller dashboard loaded</p>
                </div>
                <div class="list-group-item">
                    <small class="text-muted">2 minutes ago</small>
                    <p class="mb-1">Health check performed</p>
                </div>
            </div>
        `;

    } catch (error) {
        console.error('Error loading overview:', error);
        systemStatusElement.innerHTML = `<div class="alert alert-danger">Error: ${error.message}</div>`;
    }
}

// Load instances
async function loadInstances() {
    if (!isConnected) return;

    const tableBody = document.getElementById('instancesTableBody');
    tableBody.innerHTML = '<tr><td colspan="6" class="text-center"><div class="spinner-border" role="status"></div></td></tr>';

    try {
        const response = await fetch(`${controllerBaseUrl}/query_worker_info`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ instance_id: 'all' })
        });

        if (!response.ok) {
            throw new Error('Failed to fetch instances');
        }

        const data = await response.json();
        currentInstances = data.worker_infos;
        
        // Group workers by instance
        const instancesMap = new Map();
        currentInstances.forEach(worker => {
            if (!instancesMap.has(worker.instance_id)) {
                instancesMap.set(worker.instance_id, {
                    instance_id: worker.instance_id,
                    ip: worker.ip,
                    workers: [],
                    last_heartbeat: worker.last_heartbeat_time
                });
            }
            instancesMap.get(worker.instance_id).workers.push(worker);
            // Update latest heartbeat
            if (worker.last_heartbeat_time > instancesMap.get(worker.instance_id).last_heartbeat) {
                instancesMap.get(worker.instance_id).last_heartbeat = worker.last_heartbeat_time;
            }
        });

        // Update instance filter dropdown
        const instanceFilter = document.getElementById('instanceFilter');
        instanceFilter.innerHTML = '<option value="">All Instances</option>';
        instancesMap.forEach((instance, instanceId) => {
            const option = document.createElement('option');
            option.value = instanceId;
            option.textContent = instanceId;
            instanceFilter.appendChild(option);
        });

        // Populate table
        tableBody.innerHTML = '';
        instancesMap.forEach((instance, instanceId) => {
            const row = document.createElement('tr');
            const now = Math.floor(Date.now() / 1000);
            const timeDiff = now - instance.last_heartbeat;
            const status = timeDiff < 60 ? 'Active' : timeDiff < 300 ? 'Warning' : 'Inactive';
            const statusClass = timeDiff < 60 ? 'status-active' : timeDiff < 300 ? 'status-warning' : 'status-inactive';
            
            const lastHeartbeat = new Date(instance.last_heartbeat * 1000).toLocaleTimeString();
            
            row.innerHTML = `
                <td><strong>${instanceId}</strong></td>
                <td>${instance.ip}</td>
                <td><span class="${statusClass}">${status}</span></td>
                <td>${instance.workers.length}</td>
                <td>${lastHeartbeat}</td>
                <td>
                    <button class="btn btn-sm btn-info view-instance" data-instance="${instanceId}">View</button>
                    <button class="btn btn-sm btn-danger remove-instance" data-instance="${instanceId}">Remove</button>
                </td>
            `;
            tableBody.appendChild(row);
        });

        // Add event listeners to buttons
        document.querySelectorAll('.view-instance').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const instanceId = e.target.dataset.instance;
                alert(`Viewing instance: ${instanceId}`);
                // In a real implementation, this would navigate to instance details
            });
        });

        document.querySelectorAll('.remove-instance').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const instanceId = e.target.dataset.instance;
                if (confirm(`Are you sure you want to remove instance ${instanceId}?`)) {
                    removeInstance(instanceId);
                }
            });
        });

    } catch (error) {
        console.error('Error loading instances:', error);
        tableBody.innerHTML = `<tr><td colspan="6" class="text-center text-danger">Error: ${error.message}</td></tr>`;
    }
}

// Add instance (placeholder - would need backend implementation)
async function addInstance() {
    const instanceId = document.getElementById('newInstanceId').value.trim();
    const instanceIp = document.getElementById('newInstanceIp').value.trim();

    if (!instanceId || !instanceIp) {
        alert('Please fill all fields');
        return;
    }

    // This is a placeholder - in reality, you would need a backend endpoint to add instances
    alert(`Would add instance: ${instanceId} with IP: ${instanceIp}`);
    
    // Close modal
    const modal = bootstrap.Modal.getInstance(document.getElementById('addInstanceModal'));
    modal.hide();
    
    // Clear form
    document.getElementById('newInstanceId').value = '';
    document.getElementById('newInstanceIp').value = '';
    
    // Refresh instances list
    loadInstances();
}

// Remove instance (placeholder)
async function removeInstance(instanceId) {
    // This is a placeholder - in reality, you would need a backend endpoint to remove instances
    alert(`Would remove instance: ${instanceId}`);
    loadInstances();
}

// Load workers
async function loadWorkers() {
    if (!isConnected) return;

    const tableBody = document.getElementById('workersTableBody');
    const instanceFilter = document.getElementById('instanceFilter').value;
    
    tableBody.innerHTML = '<tr><td colspan="7" class="text-center"><div class="spinner-border" role="status"></div></td></tr>';

    try {
        const requestBody = instanceFilter ? 
            { instance_id: instanceFilter } : 
            { instance_id: 'all' };
        
        const response = await fetch(`${controllerBaseUrl}/query_worker_info`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(requestBody)
        });

        if (!response.ok) {
            throw new Error('Failed to fetch workers');
        }

        const data = await response.json();
        currentWorkers = data.worker_infos;
        
        // Populate table
        tableBody.innerHTML = '';
        currentWorkers.forEach(worker => {
            const row = document.createElement('tr');
            const now = Math.floor(Date.now() / 1000);
            const timeDiff = now - worker.last_heartbeat_time;
            const status = timeDiff < 60 ? 'Active' : timeDiff < 300 ? 'Warning' : 'Inactive';
            const statusClass = timeDiff < 60 ? 'status-active' : timeDiff < 300 ? 'status-warning' : 'status-inactive';
            
            const lastHeartbeat = new Date(worker.last_heartbeat_time * 1000).toLocaleTimeString();
            
            row.innerHTML = `
                <td>${worker.instance_id}</td>
                <td>${worker.worker_id}</td>
                <td>${worker.ip}</td>
                <td>${worker.port}</td>
                <td><span class="${statusClass}">${status}</span></td>
                <td>${lastHeartbeat}</td>
                <td>
                    <button class="btn btn-sm btn-info view-worker" data-instance="${worker.instance_id}" data-worker="${worker.worker_id}">View</button>
                </td>
            `;
            tableBody.appendChild(row);
        });

    } catch (error) {
        console.error('Error loading workers:', error);
        tableBody.innerHTML = `<tr><td colspan="7" class="text-center text-danger">Error: ${error.message}</td></tr>`;
    }
}

// Load key pool (placeholder - would need backend implementation)
async function loadKeyPool() {
    if (!isConnected) return;

    const tableBody = document.getElementById('keyPoolTableBody');
    tableBody.innerHTML = '<tr><td colspan="6" class="text-center"><div class="spinner-border" role="status"></div></td></tr>';

    try {
        // This is a placeholder - in reality, you would need a backend endpoint to get key pool
        // Simulating data for demonstration
        currentKeyPool = [
            { key: 'key_001', instance_id: 'instance_1', location: 'memory', seq_num: 1, status: 'active' },
            { key: 'key_002', instance_id: 'instance_1', location: 'disk', seq_num: 2, status: 'active' },
            { key: 'key_003', instance_id: 'instance_2', location: 'memory', seq_num: 1, status: 'pinned' },
            { key: 'key_004', instance_id: 'instance_2', location: 'memory', seq_num: 3, status: 'evicted' },
            { key: 'key_005', instance_id: 'instance_1', location: 'disk', seq_num: 4, status: 'active' }
        ];
        
        // Populate table
        tableBody.innerHTML = '';
        currentKeyPool.forEach(key => {
            const row = document.createElement('tr');
            const statusClass = key.status === 'active' ? 'status-active' : 
                               key.status === 'pinned' ? 'status-info' : 
                               'status-inactive';
            
            row.innerHTML = `
                <td><code>${key.key}</code></td>
                <td>${key.instance_id}</td>
                <td>${key.location}</td>
                <td>${key.seq_num}</td>
                <td><span class="${statusClass}">${key.status}</span></td>
                <td>
                    <button class="btn btn-sm btn-info view-key" data-key="${key.key}">View</button>
                    <button class="btn btn-sm btn-danger remove-key" data-key="${key.key}">Remove</button>
                </td>
            `;
            tableBody.appendChild(row);
        });

        // Add event listeners
        document.querySelectorAll('.view-key').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const key = e.target.dataset.key;
                alert(`Viewing key: ${key}`);
            });
        });

        document.querySelectorAll('.remove-key').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const key = e.target.dataset.key;
                if (confirm(`Are you sure you want to remove key ${key}?`)) {
                    removeKey(key);
                }
            });
        });

    } catch (error) {
        console.error('Error loading key pool:', error);
        tableBody.innerHTML = `<tr><td colspan="6" class="text-center text-danger">Error: ${error.message}</td></tr>`;
    }
}

// Filter key pool
function filterKeyPool() {
    const searchTerm = document.getElementById('keySearchInput').value.toLowerCase();
    const rows = document.querySelectorAll('#keyPoolTableBody tr');
    
    rows.forEach(row => {
        const text = row.textContent.toLowerCase();
        row.style.display = text.includes(searchTerm) ? '' : 'none';
    });
}

// Clear key pool (placeholder)
async function clearKeyPool() {
    if (!confirm('Are you sure you want to clear all keys from the key pool? This cannot be undone.')) {
        return;
    }

    // This is a placeholder - in reality, you would need a backend endpoint to clear key pool
    alert('Would clear key pool');
    loadKeyPool();
}

// Remove key (placeholder)
async function removeKey(key) {
    // This is a placeholder - in reality, you would need a backend endpoint to remove keys
    alert(`Would remove key: ${key}`);
    loadKeyPool();
}

// Load metrics
async function loadMetrics() {
    if (!isConnected) return;

    const contentDiv = document.getElementById('metricsContent');
    contentDiv.textContent = 'Loading...';

    try {
        // Note: This endpoint might not exist in the current controller
        // You would need to implement a metrics endpoint
        const response = await fetch(`${controllerBaseUrl}/metrics`);
        
        if (response.ok) {
            const metrics = await response.text();
            contentDiv.textContent = metrics;
        } else {
            contentDiv.textContent = 'Metrics endpoint not available. Would need to implement /metrics endpoint.';
        }
    } catch (error) {
        contentDiv.textContent = `Failed to load metrics: ${error.message}`;
    }
}

// Load log level
async function loadLogLevel() {
    if (!isConnected) return;

    const contentDiv = document.getElementById('logLevelContent');
    contentDiv.textContent = 'Loading...';

    try {
        // Note: This endpoint might not exist in the current controller
        // You would need to implement a log level endpoint
        contentDiv.textContent = 'Log level management would require implementing /loglevel endpoint in controller.';
    } catch (error) {
        contentDiv.textContent = `Failed to load log levels: ${error.message}`;
    }
}

// Set log level
async function setLogLevel() {
    if (!isConnected) return;

    const loggerInput = document.getElementById('loggerInput');
    const levelSelector = document.getElementById('logLevelSelector');

    const loggerName = loggerInput.value.trim();
    const level = levelSelector.value;

    if (!level) {
        alert('Please select a log level');
        return;
    }

    if (!loggerName) {
        alert('Please enter a logger name');
        return;
    }

    try {
        // This is a placeholder - you would need to implement this endpoint
        alert(`Would set log level for ${loggerName} to ${level}`);
        loadLogLevel();
    } catch (error) {
        alert(`Failed to set log level: ${error.message}`);
    }
}

// Load configuration
async function loadConfig() {
    if (!isConnected) return;

    const contentDiv = document.getElementById('configContent');
    contentDiv.textContent = 'Loading...';

    try {
        // Note: This endpoint might not exist in the current controller
        // You would need to implement a config endpoint
        contentDiv.textContent = 'Configuration management would require implementing /conf endpoint in controller.';
    } catch (error) {
        contentDiv.textContent = `Failed to load configuration: ${error.message}`;
    }
}

// Get configuration
async function getConfig() {
    if (!isConnected) return;

    const configKeyInput = document.getElementById('configKeyInput');
    const configKey = configKeyInput.value.trim();

    try {
        // This is a placeholder - you would need to implement this endpoint
        alert(`Would get configuration for key: ${configKey || 'all'}`);
    } catch (error) {
        alert(`Failed to get configuration: ${error.message}`);
    }
}

// Set configuration
async function setConfig() {
    if (!isConnected) return;

    const configKeyInput = document.getElementById('configKeyInput');
    const configValueInput = document.getElementById('configValueInput');

    const configKey = configKeyInput.value.trim();
    const configValue = configValueInput.value.trim();

    if (!configKey || !configValue) {
        alert('Please enter both key and value');
        return;
    }

    try {
        // This is a placeholder - you would need to implement this endpoint
        alert(`Would set configuration ${configKey}=${configValue}`);
        loadConfig();
    } catch (error) {
        alert(`Failed to set configuration: ${error.message}`);
    }
}

// Load threads
async function loadThreads() {
    if (!isConnected) return;

    const contentDiv = document.getElementById('threadsContent');
    contentDiv.textContent = 'Loading...';

    try {
        // Note: This endpoint might not exist in the current controller
        // You would need to implement a threads endpoint
        contentDiv.textContent = 'Thread information would require implementing /threads endpoint in controller.';
    } catch (error) {
        contentDiv.textContent = `Failed to load threads: ${error.message}`;
    }
}

// Load environment variables
async function loadEnvironment() {
    if (!isConnected) return;

    const contentDiv = document.getElementById('envContent');
    const searchInput = document.getElementById('envSearchInput');
    contentDiv.textContent = 'Loading...';
    searchInput.value = '';

    try {
        // Note: This endpoint might not exist in the current controller
        // You would need to implement an env endpoint
        // For now, simulate some data
        envVariablesData = {
            'PYTHONPATH': '/Users/msy/projects/LMCache',
            'PATH': '/usr/local/bin:/usr/bin:/bin',
            'LANG': 'en_US.UTF-8',
            'HOME': '/Users/msy',
            'USER': 'msy',
            'SHELL': '/bin/zsh',
            'PWD': '/Users/msy/projects/LMCache',
            'LMCACHE_CONFIG': 'config.yaml'
        };
        
        // Format for display
        const formattedText = Object.entries(envVariablesData)
            .map(([key, value]) => `${key}=${value}`)
            .join('\n');
        contentDiv.textContent = formattedText;
    } catch (error) {
        contentDiv.textContent = `Failed to load environment variables: ${error.message}`;
        envVariablesData = null;
    }
}

// Filter environment variables
function filterEnvVariables() {
    const searchInput = document.getElementById('envSearchInput');
    const contentDiv = document.getElementById('envContent');
    const searchTerm = searchInput.value.toLowerCase();

    if (!envVariablesData) return;

    if (typeof envVariablesData === 'object') {
        const filteredEntries = Object.entries(envVariablesData).filter(([key, value]) => {
            const line = `${key}=${value}`;
            return line.toLowerCase().includes(searchTerm);
        });
        const formattedText = filteredEntries
            .map(([key, value]) => `${key}=${value}`)
            .join('\n');
        contentDiv.textContent = formattedText;
    }
}

// Execute script
async function executeScript() {
    if (!isConnected) return;

    const scriptName = document.getElementById('scriptNameInput').value.trim();
    const scriptContent = document.getElementById('scriptContent').value.trim();
    const allowedImports = document.getElementById('allowedImportsInput').value.trim();

    if (!scriptContent) {
        alert('Please enter script content');
        return;
    }

    const resultsDiv = document.getElementById('scriptResults');
    resultsDiv.textContent = 'Executing script...';

    try {
        // This is a placeholder - you would need to implement a script execution endpoint
        // Based on test_run_script.py, the endpoint might be something like /run_script
        
        // Simulate execution for demonstration
        setTimeout(() => {
            resultsDiv.textContent = `Script execution results:\n`;
            resultsDiv.textContent += `Script: ${scriptName || 'Untitled'}\n`;
            resultsDiv.textContent += `Allowed imports: ${allowedImports || 'None'}\n`;
            resultsDiv.textContent += `Execution time: 0.5s\n`;
            resultsDiv.textContent += `Result: Script executed successfully\n`;
            resultsDiv.textContent += `Output: Hello from LMCache Controller!`;
        }, 1000);

    } catch (error) {
        resultsDiv.textContent = `Failed to execute script: ${error.message}`;
    }
}

// Upload script
function uploadScript() {
    const fileInput = document.createElement('input');
    fileInput.type = 'file';
    fileInput.accept = '.py,.txt';
    
    fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (!file) return;

        const reader = new FileReader();
        reader.onload = (event) => {
            document.getElementById('scriptContent').value = event.target.result;
            document.getElementById('scriptNameInput').value = file.name.replace(/\.[^/.]+$/, '');
        };
        reader.readAsText(file);
    });
    
    fileInput.click();
}