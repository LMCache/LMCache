// SPDX-License-Identifier: Apache-2.0
const assert = require('node:assert/strict');
const test = require('node:test');
const {dashboard, staticDir} = require('./dashboard_test_utils.cjs');
const fs = require('node:fs');
const path = require('node:path');

function nodeForm(ui, name = 'new-node') {
    ui.field('nodeName', name);
    ui.field('nodeHost', '127.0.0.1');
    ui.field('nodePort', '8085');
}

for (const isProxy of [true, false]) {
    test(`Add Node registers a ${isProxy ? 'proxy' : 'child'} through its existing route`, async () => {
        const ui = dashboard(url => Response.json(url === '/api/proxies' ? {proxies: []} : {nodes: []}));
        nodeForm(ui);
        ui.document.getElementById('isProxyCheck').checked = isProxy;
        ui.field('proxyIdInput', 'parent proxy');
        await ui.run('addNode()');
        const request = ui.requests.find(request => request.method === 'POST');
        assert.equal(request.url, isProxy ? '/api/proxies' : '/api/proxies/parent%20proxy/nodes');
        assert.equal(JSON.parse(request.body).name, 'new-node');
        assert.equal(JSON.parse(request.body).is_proxy, isProxy);
        assert.ok(ui.alerts.includes('Node added successfully'));
    });
}

test('Add Node requires a parent for children and preserves the form on server rejection', async () => {
    const ui = dashboard(() => Response.json({detail: 'Proxy name already exists'}, {status: 409}));
    nodeForm(ui);
    await ui.run('addNode()');
    assert.equal(ui.requests.length, 0);
    ui.document.getElementById('isProxyCheck').checked = true;
    await ui.run('addNode()');
    assert.match(ui.alerts.at(-1), /Proxy name already exists/);
    assert.equal(ui.document.getElementById('nodeName').value, 'new-node');
});

test('Current Nodes headings match the rendered row columns', async () => {
    const html = fs.readFileSync(path.join(staticDir, 'index.html'), 'utf8');
    const headings = [...html.matchAll(/<th>(.*?)<\/th>/g)].map(match => match[1]);
    const ui = dashboard(() => Response.json({nodes: [{name: 'proxy', host: 'localhost', port: '8085', is_proxy: true}]}));
    await ui.run('loadNodeListForManagement()');
    const row = ui.document.getElementById('nodeListBody').children[0];
    assert.deepEqual(headings, ['Name', 'Host', 'Port/Socket', 'Is Proxy', 'Proxy ID', 'Actions']);
    assert.equal([...row.innerHTML.matchAll(/<td>/g)].length, headings.length);
});

test('Edit retains the original identifier across repeated renames', async () => {
    const ui = dashboard(url => Response.json(url === '/api/proxies' ? {proxies: []} : {
        nodes: [{name: 'old name', host: 'localhost', port: '8085', is_proxy: true}],
    }));
    await ui.run('loadNodeListForManagement()');
    const edit = ui.buttons.find(button => button.kind === 'edit-node');
    edit.listeners.click({target: edit});
    ui.field('nodeName', 'renamed');
    await ui.run('updateNode()');
    ui.field('nodeName', 'renamed-again');
    await ui.run('updateNode()');
    const updates = ui.requests.filter(request => request.method === 'PUT');
    assert.deepEqual(updates.map(request => request.url), ['/api/nodes/old%20name', '/api/nodes/renamed']);
    assert.deepEqual(updates.map(request => JSON.parse(request.body).name), ['renamed', 'renamed-again']);
});

test('A failed rename keeps its original identifier for retry', async () => {
    const ui = dashboard(() => Response.json({detail: 'Node not found'}, {status: 404}));
    nodeForm(ui, 'renamed');
    ui.document.getElementById('nodeName').dataset.originalName = 'original';
    await ui.run('updateNode()');
    await ui.run('updateNode()');
    assert.deepEqual(ui.requests.map(request => request.url), ['/api/nodes/original', '/api/nodes/original']);
    assert.match(ui.alerts.at(-1), /Node not found/);
});

test('Deleting the edited node clears its identity even if the name field changed', async () => {
    const ui = dashboard(url => Response.json(url === '/api/proxies' ? {proxies: []} : {nodes: []}));
    nodeForm(ui, 'unsaved-name');
    ui.document.getElementById('nodeName').dataset.originalName = 'original';
    await ui.run("deleteNode('original')");
    assert.equal(ui.document.getElementById('nodeName').value, '');
    assert.equal(ui.document.getElementById('nodeName').dataset.originalName, undefined);
});

