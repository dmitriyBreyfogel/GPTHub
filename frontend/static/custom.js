(function () {
    var BACKEND = '';
    var INDICATOR_PREFIX = 'model:';
    var ttsActive = null;
    var workspaceId = null;

    function getUserId() {
        try {
            var raw = localStorage.getItem('user') || sessionStorage.getItem('user');
            if (raw) return JSON.parse(raw).id || 'anonymous';
        } catch (_) {}
        return 'anonymous';
    }

    function apiFetch(path, opts) {
        return fetch(BACKEND + path, Object.assign({ headers: { 'x-user-id': getUserId() } }, opts || {}));
    }

    function patchFetch() {
        var orig = window.fetch;
        window.fetch = function (url, opts) {
            var urlStr = typeof url === 'string' ? url : (url && url.url) || '';
            if (workspaceId && urlStr.includes('/api/chat/completions')) {
                try {
                    var body = JSON.parse(opts.body);
                    body.metadata = body.metadata || {};
                    body.metadata.workspace_id = workspaceId;
                    opts = Object.assign({}, opts, { body: JSON.stringify(body) });
                } catch (_) {}
            }
            return orig.apply(this, arguments);
        };
    }

    function buildPanel() {
        if (document.getElementById('gpthub-panel')) return;

        var panel = document.createElement('div');
        panel.id = 'gpthub-panel';

        var ws = document.createElement('select');
        ws.id = 'gpthub-ws-select';
        var def = document.createElement('option');
        def.value = '';
        def.textContent = 'Workspace';
        ws.appendChild(def);

        apiFetch('/v1/workspaces')
            .then(function (r) { return r.json(); })
            .then(function (data) {
                (data.workspaces || []).forEach(function (w) {
                    var o = document.createElement('option');
                    o.value = w.id;
                    o.textContent = w.name;
                    ws.appendChild(o);
                });
            })
            .catch(function () {});

        ws.addEventListener('change', function () {
            workspaceId = ws.value || null;
        });

        var memBtn = document.createElement('button');
        memBtn.id = 'gpthub-mem-btn';
        memBtn.textContent = 'Memory';
        memBtn.addEventListener('click', openMemoryModal);

        panel.appendChild(ws);
        panel.appendChild(memBtn);
        document.body.appendChild(panel);
    }

    function buildMemoryModal() {
        if (document.getElementById('gpthub-memory-modal')) return;
        var modal = document.createElement('div');
        modal.id = 'gpthub-memory-modal';
        modal.innerHTML = [
            '<div id="gpthub-memory-panel">',
            '<header><span>Memory</span><button id="gpthub-mem-close">&times;</button></header>',
            '<div id="gpthub-memory-list"><div id="gpthub-memory-empty">Loading...</div></div>',
            '</div>'
        ].join('');
        document.body.appendChild(modal);
        modal.addEventListener('click', function (e) { if (e.target === modal) closeMemoryModal(); });
        document.getElementById('gpthub-mem-close').addEventListener('click', closeMemoryModal);
    }

    function openMemoryModal() {
        buildMemoryModal();
        document.getElementById('gpthub-memory-modal').classList.add('open');
        loadMemories();
    }

    function closeMemoryModal() {
        var m = document.getElementById('gpthub-memory-modal');
        if (m) m.classList.remove('open');
    }

    function loadMemories() {
        var list = document.getElementById('gpthub-memory-list');
        if (!list) return;
        list.innerHTML = '<div id="gpthub-memory-empty">Loading...</div>';
        apiFetch('/v1/memory')
            .then(function (r) { return r.json(); })
            .then(function (data) {
                var items = data.memories || [];
                if (!items.length) {
                    list.innerHTML = '<div id="gpthub-memory-empty">No memories yet</div>';
                    return;
                }
                list.innerHTML = '';
                items.forEach(function (m) {
                    var item = document.createElement('div');
                    item.className = 'gpthub-memory-item';
                    var txt = document.createElement('span');
                    txt.textContent = m.text;
                    var del = document.createElement('button');
                    del.className = 'gpthub-memory-del';
                    del.textContent = '\u00d7';
                    del.addEventListener('click', function () {
                        apiFetch('/v1/memory/' + m.id, { method: 'DELETE' })
                            .then(function () { item.remove(); })
                            .catch(function () {});
                    });
                    item.appendChild(txt);
                    item.appendChild(del);
                    list.appendChild(item);
                });
            })
            .catch(function () {
                list.innerHTML = '<div id="gpthub-memory-empty">Error loading</div>';
            });
    }

    function transformIndicator(el) {
        if (el.dataset.gpthubDone) return;
        el.dataset.gpthubDone = '1';
        var text = (el.textContent || '').trim();
        if (!text.startsWith(INDICATOR_PREFIX)) return;
        var badge = document.createElement('div');
        badge.className = 'gpthub-indicator';
        badge.textContent = text;
        el.parentNode.replaceChild(badge, el);
    }

    function addTtsButton(msgEl) {
        if (msgEl.dataset.gpthubTts) return;
        if (!('speechSynthesis' in window)) return;
        msgEl.dataset.gpthubTts = '1';

        var btn = document.createElement('button');
        btn.className = 'gpthub-tts';
        btn.title = 'Read aloud';
        btn.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/><path d="M15.54 8.46a5 5 0 0 1 0 7.07"/><path d="M19.07 4.93a10 10 0 0 1 0 14.14"/></svg>';

        btn.addEventListener('click', function () {
            if (ttsActive === btn) {
                window.speechSynthesis.cancel();
                btn.classList.remove('active');
                ttsActive = null;
                return;
            }
            if (ttsActive) { window.speechSynthesis.cancel(); ttsActive.classList.remove('active'); }
            var clone = msgEl.cloneNode(true);
            clone.querySelectorAll('.gpthub-indicator, .gpthub-tts').forEach(function (n) { n.remove(); });
            var text = (clone.textContent || '').trim();
            if (!text) return;
            var utter = new SpeechSynthesisUtterance(text);
            utter.lang = 'ru-RU';
            utter.rate = 1.05;
            utter.onend = function () { btn.classList.remove('active'); ttsActive = null; };
            utter.onerror = function () { btn.classList.remove('active'); ttsActive = null; };
            btn.classList.add('active');
            ttsActive = btn;
            window.speechSynthesis.speak(utter);
        });

        var actions = msgEl.querySelector('[class*="action"], [class*="button"], [class*="toolbar"]');
        if (actions) {
            actions.appendChild(btn);
        } else {
            var wrap = document.createElement('div');
            wrap.style.cssText = 'display:flex;justify-content:flex-end;margin-top:6px;';
            wrap.appendChild(btn);
            msgEl.appendChild(wrap);
        }
    }

    function processNodes(root) {
        root.querySelectorAll('code').forEach(transformIndicator);
        ['[data-role="assistant"]', '[class*="assistant"]', '[class*="response"]'].forEach(function (sel) {
            root.querySelectorAll(sel).forEach(addTtsButton);
        });
    }

    function init() {
        patchFetch();
        buildPanel();
        processNodes(document.body);

        var observer = new MutationObserver(function (mutations) {
            mutations.forEach(function (m) {
                m.addedNodes.forEach(function (node) {
                    if (node.nodeType === 1) processNodes(node);
                });
            });
        });
        observer.observe(document.body, { childList: true, subtree: true });
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
