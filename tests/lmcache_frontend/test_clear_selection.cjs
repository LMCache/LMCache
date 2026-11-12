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
