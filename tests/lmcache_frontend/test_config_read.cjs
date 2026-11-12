// SPDX-License-Identifier: Apache-2.0
const assert = require('node:assert/strict');
const test = require('node:test');
const {dashboard} = require('./dashboard_test_utils.cjs');

for (const legacy of [false, true]) {
    test(`Config loads ${legacy ? 'embedded vLLM /conf' : 'MP /config'} without breaking the other API`, async () => {
        const config = legacy ? {chunk_size: 256} : {mp: {chunk_size: 256}, storage_manager: {}};
        const ui = dashboard(url => legacy && url.endsWith('/config')
            ? new Response('Not Found', {status: 404}) : Response.json(config));
        ui.run("currentNode = {name: 'cache'}");
        await ui.run('loadConfig()');
        assert.deepEqual(JSON.parse(ui.document.getElementById('configContent').textContent), config);
        ui.field('configKeyInput', legacy ? 'chunk_size' : 'mp');
        await ui.run('getConfig()');
        if (legacy) {
            assert.equal(ui.requests.at(-1).url, '/proxy2/cache/conf?names=chunk_size');
        } else {
            assert.deepEqual(JSON.parse(ui.alerts.at(-1)), {mp: config.mp});
        }
    });
}

for (const status of [401, 500, 503]) {
    test(`Config reports HTTP ${status} without falling back to a different API`, async () => {
        const ui = dashboard(() => new Response('config unavailable', {status}));
        ui.run("currentNode = {name: 'cache'}");
        await ui.run('loadConfig()');
        assert.match(ui.document.getElementById('configContent').textContent, new RegExp(`Failed to load configuration: HTTP ${status}`));
        assert.equal(ui.requests.length, 1);
    });
}

