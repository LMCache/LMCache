// SPDX-License-Identifier: Apache-2.0
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const source = fs.readFileSync(path.join(__dirname,
    '../../lmcache/lmcache_frontend/static/js/app.js'), 'utf8');

for (const [name, resource, key, expected] of [
    ['same-origin string', '/api/proxies', 'secret', 'Bearer secret'],
    ['cross-origin string', 'https://other.example/api', 'secret', null],
    ['cross-origin Request', new Request('https://other.example/api'), 'secret', null],
    ['no configured key', '/api/proxies', null, null],
    ['Request headers', new Request('https://frontend.example/api', {
        headers: {'X-Custom': 'preserved'},
    }), 'secret', 'Bearer secret'],
]) {
    test(name, async () => {
        let received;
        const window = {
            location: new URL('https://frontend.example/'),
            addEventListener() {},
            fetch: async (input, options) => {
                received = new Headers(options.headers ||
                    (input instanceof Request ? input.headers : {}));
            },
        };
        vm.runInNewContext(source, {
            window, URL, Request, Headers,
            sessionStorage: {getItem: () => key},
        });
        await window.fetch(resource);
        assert.equal(received.get('Authorization'), expected);
        if (name === 'Request headers') {
            assert.equal(received.get('X-Custom'), 'preserved');
        }
    });
}
