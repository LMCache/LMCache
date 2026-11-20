// SPDX-License-Identifier: Apache-2.0
const assert = require('node:assert/strict');
const test = require('node:test');
const {dashboard} = require('./dashboard_test_utils.cjs');

test('Config loads MP configuration and Get selects a top-level section through a nested proxy', async () => {
    const config = {mp: {chunk_size: 256}, storage_manager: {}};
    const ui = dashboard(() => Response.json(config));
    ui.run("proxyNodes = {gateway: {name: 'gateway'}}; currentNode = {name: 'worker', proxy_id: 'gateway'}");
    await ui.run('loadConfig()');
    assert.deepEqual(JSON.parse(ui.document.getElementById('configContent').textContent), config);
    ui.field('configKeyInput', 'mp');
    await ui.run('getConfig()');
    assert.deepEqual(JSON.parse(ui.alerts.at(-1)), {mp: config.mp});
    assert.deepEqual(ui.requests.map(request => request.url), [
        '/proxy2/gateway/proxy2/worker/config', '/proxy2/gateway/proxy2/worker/config',
    ]);
});

for (const key of ['', 'missing', 'toString', '__proto__']) {
    test(`Get Config handles ${key || 'an empty key'} without treating inherited properties as sections`, async () => {
        const config = {mp: {chunk_size: 256}};
        const ui = dashboard(() => Response.json(config));
        ui.run("currentNode = {name: 'cache'}");
        ui.field('configKeyInput', key);
        await ui.run('getConfig()');
        assert.deepEqual(JSON.parse(ui.alerts.at(-1)), key ? {} : config);
        assert.equal(ui.requests[0].url, '/proxy2/cache/config');
    });
}

for (const status of [401, 404, 500, 503]) {
    test(`Config reports HTTP ${status} without a legacy request`, async () => {
        const ui = dashboard(() => new Response('config unavailable', {status}));
        ui.run("currentNode = {name: 'cache'}");
        await ui.run('loadConfig()');
        assert.equal(ui.document.getElementById('configContent').textContent,
            `Failed to load configuration: HTTP ${status}: config unavailable`);
        await ui.run('getConfig()');
        assert.equal(ui.alerts.at(-1), `Failed to get configuration: HTTP ${status}: config unavailable`);
        assert.deepEqual(ui.requests.map(request => request.url), ['/proxy2/cache/config', '/proxy2/cache/config']);
    });
}
