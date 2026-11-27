// SPDX-License-Identifier: Apache-2.0
const assert = require('node:assert/strict');
const test = require('node:test');
const {dashboard} = require('./dashboard_test_utils.cjs');

test('Clear resets selectors, node badge and tab contents, and stops requests to the previous target', async () => {
    const ui = dashboard(() => Response.json({proxies: []}));
    ui.window.DOMContentLoaded();
    await new Promise(resolve => setImmediate(resolve));
    ui.run("currentProxy = {name: 'proxy'}; currentNode = {name: 'cache'}");
    for (const id of ['proxySelector', 'proxySearchInput', 'targetSelector', 'envFilterInput']) ui.field(id, 'old selection');
    ui.document.getElementById('overviewContent').innerHTML = 'old overview';
    ui.document.getElementById('metricsContent').textContent = 'old metrics';
    ui.run('clearEnvFilter()');
    await new Promise(resolve => setImmediate(resolve));
    const before = ui.requests.length;
    await ui.run('loadMetrics()');
    ui.document.getElementById('refreshPageBtn').listeners.click();
    assert.equal(ui.requests.length, before);
    assert.equal(ui.run('currentNode'), null);
    assert.equal(ui.run('currentProxy'), null);
    assert.equal(ui.document.getElementById('currentNode').textContent, 'No Node Selected');
    assert.equal(ui.document.getElementById('proxySelector').value, '');
    assert.equal(ui.document.getElementById('proxySearchInput').value, '');
    assert.equal(ui.document.getElementById('targetSelector').disabled, true);
    assert.equal(ui.document.getElementById('overviewContent').innerHTML, 'Please select a target node first');
    assert.equal(ui.document.getElementById('metricsContent').textContent, 'Please select a target node first');
});

for (const [loader, content] of [
    ['loadOverview', 'overviewContent'], ['loadMetrics', 'metricsContent'],
    ['loadThreads', 'threadsContent'], ['loadLogLevel', 'logLevelContent'],
    ['loadConfig', 'configContent'], ['loadMeta', 'metaContent'],
    ['loadInference', 'inferenceContent'], ['loadEnvironment', 'envContent'],
]) {
    for (const failed of [false, true]) {
        test(`Clear keeps ${content} empty when a pending ${loader} ${failed ? 'fails' : 'finishes'}`, {timeout: 1000}, async () => {
            let complete;
            const ui = dashboard(url => url === '/api/proxies'
                ? Response.json({proxies: []})
                : new Promise((resolve, reject) => { complete = failed ? reject : resolve; }));
            ui.run("currentNode = {name: 'cache'}");
            const pending = ui.run(`${loader}()`);
            ui.run('clearEnvFilter()');
            complete(failed ? new Error('offline') : Response.json({is_healthy: true}));
            await pending;
            const element = ui.document.getElementById(content);
            assert.equal(content === 'overviewContent' ? element.innerHTML : element.textContent,
                'Please select a target node first');
            assert.equal(ui.requests.filter(request => request.url !== '/api/proxies').length, 1);
        });
    }
}

test('A proxy response that arrives after Clear does not re-enable the target selector', async () => {
    let complete;
    const ui = dashboard(url => url === '/api/proxies'
        ? Response.json({proxies: []}) : new Promise(resolve => { complete = resolve; }));
    ui.run("currentProxy = {name: 'proxy'}");
    const pending = ui.run("loadTargetNodes('proxy')");
    ui.run('clearEnvFilter()');
    complete(Response.json({nodes: [{name: 'old-target'}]}));
    await pending;
    assert.equal(ui.document.getElementById('targetSelector').disabled, true);
    assert.equal(ui.document.getElementById('targetSelector').children.length, 0);
});

test('An old configuration read cannot overwrite a newly selected node', async () => {
    let complete;
    const ui = dashboard(() => !complete
        ? new Promise(resolve => { complete = resolve; })
        : Response.json({chunk_size: 256}));
    ui.run("currentNode = {name: 'node-a'}");
    const pending = ui.run('loadConfig()');
    ui.run("currentNode = {name: 'node-b'}");
    ui.document.getElementById('configContent').textContent = 'Node B configuration';
    complete(new Response('Not Found', {status: 404}));
    await pending;
    assert.equal(ui.document.getElementById('configContent').textContent, 'Node B configuration');
    assert.ok(ui.requests.every(request => request.url.startsWith('/proxy2/node-a/')));
});

// These controls prove that Overview reaches each renderer before testing cancellation.
for (const mode of ['mp', 'version', 'version-error']) {
    test(`Overview renders the ${mode} path while its target remains selected`, async () => {
        const ui = dashboard(url => {
            if (url.endsWith('/api/status')) return mode === 'mp'
                ? Response.json({is_healthy: true, engine_type: '<MP & engine>'})
                : new Response('Not Found', {status: 404});
            if (mode === 'version-error') throw new Error('<offline & unavailable>');
            return new Response('<version & value>');
        });
        ui.run("currentNode = {name: 'cache', host: 'localhost', port: 8080}");
        await ui.run('loadOverview()');
        const html = ui.document.getElementById('overviewContent').innerHTML;
        assert.match(html, mode === 'mp' ? /&lt;MP &amp; engine&gt;/
            : mode === 'version' ? /&lt;version &amp; value&gt;/
            : /Failed to load overview: &lt;offline &amp; unavailable&gt;/);
        assert.equal(ui.requests.length, mode === 'mp' ? 1 : 2);
    });
}

for (const mode of ['mp', 'version', 'version-error']) {
    for (const selection of ['clear', 'switch']) {
        test(`Overview ${mode} cannot overwrite the UI after ${selection}`, async () => {
            let complete;
            let started;
            const waiting = new Promise(resolve => { started = resolve; });
            const ui = dashboard(url => {
                if (url === '/api/proxies') return Response.json({proxies: []});
                if (mode !== 'mp' && url.endsWith('/api/status')) return new Response('', {status: 404});
                started();
                return new Promise((resolve, reject) => { complete = mode === 'version-error' ? reject : resolve; });
            });
            ui.run("currentNode = {name: 'node-a', host: 'localhost', port: 8080}");
            const pending = ui.run('loadOverview()');
            await waiting;
            if (selection === 'clear') ui.run('clearEnvFilter()');
            else {
                ui.run("currentNode = {name: 'node-b'}");
                ui.document.getElementById('overviewContent').innerHTML = 'Node B overview';
            }
            complete(mode === 'version-error' ? new Error('offline') : mode === 'mp'
                ? Response.json({is_healthy: true, engine_type: 'MP'}) : new Response('old version'));
            await pending;
            assert.equal(ui.document.getElementById('overviewContent').innerHTML,
                selection === 'clear' ? 'Please select a target node first' : 'Node B overview');
            assert.ok(ui.requests.filter(request => request.url !== '/api/proxies')
                .every(request => request.url.startsWith('/proxy2/node-a/')));
        });
    }
}

for (const [loader, path, content, label] of [
    ['loadMetrics', 'metrics', 'metricsContent', 'metrics'],
    ['loadThreads', 'threads', 'threadsContent', 'threads'],
    ['loadMeta', 'meta', 'metaContent', 'meta information'],
    ['loadInference', 'inference_info', 'inferenceContent', 'inference information'],
]) {
    test(`${loader} preserves its endpoint, text and error label`, async () => {
        let failed = false;
        const ui = dashboard(() => {
            if (failed) throw new Error('offline');
            return new Response('<diagnostic text>');
        });
        ui.run("currentNode = {name: 'cache'}");
        await ui.run(`${loader}()`);
        assert.equal(ui.requests.at(-1).url, `/proxy2/cache/${path}`);
        assert.equal(ui.document.getElementById(content).textContent, '<diagnostic text>');
        failed = true;
        await ui.run(`${loader}()`);
        assert.equal(ui.document.getElementById(content).textContent, `Failed to load ${label}: offline`);
    });
}
