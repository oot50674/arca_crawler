(() => {
    const activeTargets = new WeakMap();

    const setBusy = (el) => {
        if (!el) return;
        el.dataset.busy = 'true';
        el.classList.add('opacity-80');
    };

    const clearBusy = (el) => {
        if (!el) return;
        el.dataset.busy = 'false';
        el.classList.remove('opacity-80');
    };

    document.addEventListener('htmx:beforeRequest', (event) => {
        const initiator = event.target;
        const targetSelector = initiator?.getAttribute('hx-target');
        const target = targetSelector ? document.querySelector(targetSelector) : initiator;
        if (!target) return;
        activeTargets.set(event.detail.xhr, target);
        setBusy(target);
    });

    document.addEventListener('htmx:afterRequest', (event) => {
        const target = activeTargets.get(event.detail.xhr);
        clearBusy(target);
        activeTargets.delete(event.detail.xhr);
    });

    document.addEventListener('htmx:responseError', (event) => {
        const target = activeTargets.get(event.detail.xhr);
        clearBusy(target);
        if (target) {
            target.classList.add('ring-2', 'ring-red-200');
            setTimeout(() => target.classList.remove('ring-2', 'ring-red-200'), 700);
        }
    });

    document.addEventListener('alpine:init', () => {
        window.HATHelpers = {
            focusById(id) {
                const el = document.getElementById(id);
                if (el && typeof el.focus === 'function') {
                    el.focus();
                }
            },
        };
    });

    window.addEventListener('DOMContentLoaded', () => {
        const form = document.getElementById('backup-form');
        const target = document.getElementById('run-result');
        const indicator = document.getElementById('run-indicator');

        // --- localStorage: save/load backup form settings ---
        const storageKey = 'arca_backup_settings';
        const fields = ['channel', 'category', 'start_page', 'end_page', 'sleep', 'out_name'];

        function applySettings(settings) {
            if (!settings || typeof settings !== 'object') return;
            fields.forEach((name) => {
                if (!(name in settings)) return;
                const el = document.querySelector(`[name="${name}"]`);
                if (!el) return;
                const val = settings[name];
                el.value = val === undefined || val === null ? '' : val;
                el.dispatchEvent(new Event('input', { bubbles: true }));
            });
            window.scrollTo({ top: 0, behavior: 'smooth' });
        }

        function loadSettings() {
            const raw = localStorage.getItem(storageKey);
            if (!raw) return;
            try {
                const obj = JSON.parse(raw);
                fields.forEach((name) => {
                    const val = obj[name];
                    if (val !== undefined && val !== null) {
                        const el = document.querySelector(`[name="${name}"]`);
                        if (el) {
                            el.value = val;
                            // trigger input event so Alpine updates bound models
                            el.dispatchEvent(new Event('input', { bubbles: true }));
                        }
                    }
                });
            } catch (e) {
                console.warn('Failed to parse saved settings', e);
            }
        }

        function saveSettings() {
            const obj = {};
            fields.forEach((name) => {
                const el = document.querySelector(`[name="${name}"]`);
                if (el) obj[name] = el.value;
            });
            try {
                localStorage.setItem(storageKey, JSON.stringify(obj));
            } catch (e) {
                console.warn('Failed to save settings', e);
            }
        }

        fields.forEach((name) => {
            const el = document.querySelector(`[name="${name}"]`);
            if (!el) return;
            el.addEventListener('input', saveSettings);
        });

        // load saved values once on start
        loadSettings();

        document.addEventListener('click', (event) => {
            const btn = event.target.closest('[data-action="apply-settings"]');
            if (!btn) return;
            const payload = btn.getAttribute('data-settings');
            if (!payload) return;
            try {
                const parsed = JSON.parse(payload);
                applySettings(parsed);
            } catch (err) {
                console.warn('설정 불러오기 실패', err);
            }
        });

        // Fallback: if htmx is unavailable, handle backup form submit with fetch
        if (window.htmx) {
            return;
        }
        if (!form || !target) return;

        const setIndicator = (show) => {
            if (!indicator) return;
            indicator.classList.toggle('hidden', !show);
        };

        form.addEventListener('submit', async (event) => {
            event.preventDefault();
            setIndicator(true);
            try {
                const data = new FormData(form);
                const response = await fetch('/backup/run', {
                    method: 'POST',
                    body: data,
                });
                const html = await response.text();
                target.innerHTML = html;
            } catch (err) {
                target.innerHTML = `<div class="text-red-700 text-sm">요청 실패: ${err}</div>`;
            } finally {
                setIndicator(false);
            }
        });
    });
})();
