// SPDX-License-Identifier: Apache-2.0
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const {dashboard, staticDir} = require('./dashboard_test_utils.cjs');

test('Set Config does not dispatch network requests and reports unsupported for MP target', async () => {
    const ui = dashboard(() => new Response('Not Found', {status: 404}));
    ui.run("currentNode = {name: 'worker', host: '127.0.0.1', port: 8085}");
    ui.field('configKeyInput', 'chunk_size');
    ui.field('configValueInput', '256');

    await ui.run('setConfig()');

    assert.equal(ui.requests.length, 0, 'No network requests must be sent for unsupported MP writes');
    assert.equal(
        ui.alerts.at(-1),
        'Configuration updates are not supported for multiprocess servers.',
        'Unsupported configuration write must report clear explanation'
    );
});

test('UI state: Set Config button and value input are disabled, with visible explanation', () => {
    const html = fs.readFileSync(path.join(staticDir, 'index.html'), 'utf8');

    // Assert setConfigBtn is disabled in HTML markup
    assert.match(
        html,
        /<button\s+[^>]*id="setConfigBtn"[^>]*\bdisabled\b[^>]*>/,
        'setConfigBtn button must have disabled attribute in index.html'
    );

    // Assert configValueInput is disabled in HTML markup
    assert.match(
        html,
        /<input\s+[^>]*id="configValueInput"[^>]*\bdisabled\b[^>]*>/,
        'configValueInput must have disabled attribute in index.html'
    );

    // Assert visible explanatory message exists in index.html
    assert.match(
        html,
        /Configuration updates are not supported for multiprocess servers\./,
        'index.html must include visible explanatory notice for unsupported MP configuration updates'
    );

    // Verify DOM element state via dashboard harness
    const ui = dashboard();
    assert.equal(
        ui.document.getElementById('setConfigBtn').disabled,
        true,
        'setConfigBtn must be disabled on initial dashboard load'
    );
    assert.equal(
        ui.document.getElementById('configValueInput').disabled,
        true,
        'configValueInput must be disabled on initial dashboard load'
    );
});

test('Direct setConfig() invocation with nested proxy target sends no network requests', async () => {
    const ui = dashboard(() => new Response('Not Found', {status: 404}));
    ui.run("proxyNodes = {gateway: {name: 'gateway'}}; currentNode = {name: 'worker', proxy_id: 'gateway'}");
    ui.field('configKeyInput', 'mp');
    ui.field('configValueInput', 'value');

    await ui.run('setConfig()');

    assert.equal(ui.requests.length, 0, 'Direct setConfig() on nested proxy target must not dispatch requests');
    assert.equal(
        ui.alerts.at(-1),
        'Configuration updates are not supported for multiprocess servers.'
    );
});

test('Target switching and clearing selection preserves disabled write affordance', async () => {
    const ui = dashboard();
    ui.run("currentNode = {name: 'node1', host: '127.0.0.1', port: 8085}");
    ui.run('clearAllTabs()');

    assert.equal(ui.document.getElementById('setConfigBtn').disabled, true);
    assert.equal(ui.document.getElementById('configValueInput').disabled, true);

    ui.run("currentNode = {name: 'node2', host: '127.0.0.1', port: 8086}");
    await ui.run('loadConfig()');

    assert.equal(ui.document.getElementById('setConfigBtn').disabled, true);
    assert.equal(ui.document.getElementById('configValueInput').disabled, true);

    await ui.run('setConfig()');
    assert.equal(ui.requests.length, 1); // Only the loadConfig() read request was made
    assert.equal(
        ui.alerts.at(-1),
        'Configuration updates are not supported for multiprocess servers.'
    );
});

test('Configuration reads remain functional with nested proxy routing', async () => {
    const configData = '{"mp": {"chunk_size": 256}}';
    const ui = dashboard(() => new Response(configData, {status: 200}));
    ui.run("proxyNodes = {gateway: {name: 'gateway'}}; currentNode = {name: 'worker', proxy_id: 'gateway'}");

    await ui.run('loadConfig()');

    assert.equal(ui.document.getElementById('configContent').textContent, configData);
    assert.equal(ui.requests.length, 1);
    assert.match(ui.requests[0].url, /\/proxy2\/gateway\/proxy2\/worker\//);

    ui.field('configKeyInput', 'mp');
    await ui.run('getConfig()');
    assert.equal(ui.requests.length, 2);
    assert.match(ui.requests[1].url, /\/proxy2\/gateway\/proxy2\/worker\//);
});
