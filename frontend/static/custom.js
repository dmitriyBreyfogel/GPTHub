(function () {
    var INDICATOR_PREFIX = 'model:';
    var ttsActive = null;

    function cleanBranding(root) {
        (root || document).querySelectorAll('*').forEach(function (el) {
            if (el.childElementCount > 0) return;
            el.childNodes.forEach(function (node) {
                if (node.nodeType === 3 && node.textContent.includes(' (Open WebUI)')) {
                    node.textContent = node.textContent.replace(/ \(Open WebUI\)/g, '');
                }
            });
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
        btn.title = 'Озвучить';
        btn.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/><path d="M15.54 8.46a5 5 0 0 1 0 7.07"/><path d="M19.07 4.93a10 10 0 0 1 0 14.14"/></svg>';

        btn.addEventListener('click', function () {
            if (ttsActive === btn) {
                window.speechSynthesis.cancel();
                btn.classList.remove('active');
                ttsActive = null;
                return;
            }
            if (ttsActive) {
                window.speechSynthesis.cancel();
                ttsActive.classList.remove('active');
            }
            var clone = msgEl.cloneNode(true);
            clone.querySelectorAll('.gpthub-indicator, .gpthub-tts').forEach(function (n) { n.remove(); });
            var text = (clone.textContent || '').trim();
            if (!text) return;
            var utter = new SpeechSynthesisUtterance(text);
            utter.lang = 'ru-RU';
            utter.rate = 1.05;
            utter.onend = function () {
                btn.classList.remove('active');
                ttsActive = null;
            };
            utter.onerror = function () {
                btn.classList.remove('active');
                ttsActive = null;
            };
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
        cleanBranding(root);
        root.querySelectorAll('code').forEach(transformIndicator);
        ['[data-role="assistant"]', '[class*="assistant"]', '[class*="response"]'].forEach(function (sel) {
            root.querySelectorAll(sel).forEach(addTtsButton);
        });
    }

    function init() {
        cleanBranding(document);
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
