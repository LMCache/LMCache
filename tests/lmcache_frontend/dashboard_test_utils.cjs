// SPDX-License-Identifier: Apache-2.0
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const staticDir = path.join(__dirname, '../../lmcache/lmcache_frontend/static');
const source = fs.readFileSync(path.join(staticDir, 'js/app.js'), 'utf8');

function dashboard(respond = () => new Response('{}')) {
    const elements = new Map();
    const requests = [];
    const alerts = [];
    const buttons = [];
    function element() {
        return {
            value: '', dataset: {}, checked: false, disabled: false,
            textContent: '', innerHTML: '', children: [], listeners: {},
            classList: {add() {}, remove() {}},
            addEventListener(event, callback) { this.listeners[event] = callback; },
            appendChild(child) {
                this.children.push(child);
                for (const match of child.innerHTML.matchAll(/class="[^"]*(edit-node|delete-node)[^"]*" data-name="([^"]+)"/g)) {
                    const button = element();
                    button.kind = match[1];
                    button.dataset.name = match[2];
                    buttons.push(button);
                }
            },
        };
    }
    const document = {
        getElementById(id) {
            if (!elements.has(id)) elements.set(id, element());
            return elements.get(id);
        },
        createElement: element,
        addEventListener() {},
        querySelector() { return {id: 'metrics'}; },
        querySelectorAll(selector) {
            return buttons.filter(button => selector === `.${button.kind}`);
        },
    };
    const window = {
        location: new URL('http://frontend.test/'),
        addEventListener(event, callback) { this[event] = callback; },
        fetch: async (resource, options = {}) => {
            requests.push({url: resource, ...options});
            return respond(resource, options);
        },
    };
    const context = vm.createContext({
        document, window, URL, Request, Headers, console,
        sessionStorage: {getItem: () => null},
        fetch: (...args) => window.fetch(...args),
        alert: message => alerts.push(message), confirm: () => true,
    });
    vm.runInContext(source, context);
    return {
        document, requests, alerts, buttons, context, window,
        run: code => vm.runInContext(code, context),
        field: (id, value) => { document.getElementById(id).value = value; },
    };
}


module.exports = {dashboard, staticDir};
