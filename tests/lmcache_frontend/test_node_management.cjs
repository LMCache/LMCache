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
    assert.equal(row.children.length, headings.length);
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

test('Current Nodes includes children and retains existing proxy rows in order', async () => {
    const ui = dashboard(() => Response.json({nodes: [
        {name: 'proxy', host: 'localhost', port: '8000', is_proxy: true, children: [
            {name: 'child', host: 'cache.local', port: '8085', is_proxy: false, proxy_id: 'proxy'},
        ]},
        {name: 'empty-proxy', host: 'localhost', port: '8001', is_proxy: true, children: []},
    ]}));
    await ui.run('loadNodeListForManagement()');
    const rows = ui.document.getElementById('nodeListBody').children;
    assert.deepEqual(rows.map(row => row.children[0].textContent),
        ['proxy', 'empty-proxy', 'child']);
    assert.deepEqual(rows[2].children.slice(0, 5).map(cell => cell.textContent),
        ['child', 'cache.local', '8085', 'No', 'proxy']);
});

test('Edit preserves proxy precedence when an earlier child shares its name', async () => {
    const ui = dashboard(() => Response.json({nodes: [
        {name: 'first-proxy', host: 'localhost', port: '8000', is_proxy: true, children: [
            {name: 'shared', host: 'child.local', port: '8085', is_proxy: false, proxy_id: 'first-proxy'},
        ]},
        {name: 'shared', host: 'proxy.local', port: '8001', is_proxy: true, children: []},
    ]}));
    await ui.run('loadNodeListForManagement()');
    const edit = ui.buttons.find(button => button.kind === 'edit-node' && button.dataset.name === 'shared');
    edit.listeners.click({target: edit});
    assert.equal(ui.document.getElementById('nodeHost').value, 'proxy.local');
    assert.equal(ui.document.getElementById('nodePort').value, '8001');
    assert.equal(ui.document.getElementById('isProxyCheck').checked, true);
    assert.equal(ui.document.getElementById('proxyIdInput').value, '');
});

test('Edit loads a child and renames it using its original identity and parent', async () => {
    const ui = dashboard(() => Response.json({nodes: [
        {name: 'proxy', host: 'localhost', port: '8000', is_proxy: true, children: [
            {name: 'child', host: 'cache.local', port: '8085', is_proxy: false, proxy_id: 'proxy'},
        ]},
    ]}));
    await ui.run('loadNodeListForManagement()');
    const edit = ui.buttons.find(button => button.kind === 'edit-node' && button.dataset.name === 'child');
    assert.ok(edit, 'The child must have an Edit button');
    edit.listeners.click({target: edit});
    assert.equal(ui.document.getElementById('nodeName').value, 'child');
    assert.equal(ui.document.getElementById('nodeHost').value, 'cache.local');
    assert.equal(ui.document.getElementById('nodePort').value, '8085');
    assert.equal(ui.document.getElementById('isProxyCheck').checked, false);
    assert.equal(ui.document.getElementById('proxyIdInput').value, 'proxy');
    ui.field('nodeName', 'renamed-child');
    await ui.run('updateNode()');
    const request = ui.requests.find(request => request.method === 'PUT');
    assert.equal(request.url, '/api/nodes/child');
    assert.deepEqual(JSON.parse(request.body), {
        name: 'renamed-child', host: 'cache.local', port: '8085', is_proxy: false, proxy_id: 'proxy',
    });
});

test('Discovery markup remains literal in child cells and action identifiers', async () => {
    const name = '<img src=x onerror="alert(1)">';
    const host = '<svg onload="alert(1)">';
    const port = '<b>8085</b>';
    const parent = 'parent" & <tag>';
    const ui = dashboard(() => Response.json({nodes: [{
        name: 'proxy', children: [{name, host, port, is_proxy: false, proxy_id: parent}],
    }]}));
    await ui.run('loadNodeListForManagement()');
    const row = ui.document.getElementById('nodeListBody').children[1];
    assert.deepEqual(row.children.slice(0, 5).map(cell => cell.textContent),
        [name, host, port, 'No', parent]);
    const buttons = row.children[5].children;
    assert.equal(buttons.length, 2);
    assert.deepEqual(buttons.map(button => button.dataset.name), [name, name]);
    buttons[0].listeners.click({target: buttons[0]});
    assert.equal(ui.document.getElementById('nodeName').value, name);
    assert.equal(ui.document.getElementById('nodeName').dataset.originalName, name);
    ui.run('deleteNode = name => { window.deletedName = name; }');
    buttons[1].listeners.click({target: buttons[1]});
    assert.equal(ui.window.deletedName, name);
});
