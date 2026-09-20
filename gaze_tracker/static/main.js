const el = (id) => document.getElementById(id);

// Set while applying a server snapshot, so echoed changes don't post back.
let syncing = false;

async function post(config) {
    if (syncing) return;
    try {
        apply(await (await fetch("/config", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(config),
        })).json());
    } catch (e) {
        el("error").textContent = "Could not reach the app.";
    }
}

function apply(state) {
    syncing = true;

    el("label").value = state.label;
    el("conf").value = state.conf;
    el("conf-value").textContent = Number(state.conf).toFixed(2);
    el("enabled").checked = state.enabled;
    el("scan").checked = state.scan;
    if (document.activeElement !== el("server-url")) {
        el("server-url").value = state.server_url;
    }

    badge("badge-detector", state.detector_ok, state.detector_ok ? "detector up" : "detector down");
    badge("badge-lock", state.locked, state.locked ? `locked: ${state.label}` : "searching");
    el("badge-fps").textContent = state.detector_ok ? `${state.fps} fps` : "– fps";

    const marker = el("marker");
    if (state.center && state.locked) {
        marker.style.display = "block";
        marker.style.left = `${(state.center[0] + 1) / 2 * 100}%`;
        marker.style.top = `${(state.center[1] + 1) / 2 * 100}%`;
    } else {
        marker.style.display = "none";
    }

    el("error").textContent = state.error || "";
    syncing = false;
}

function badge(id, ok, text) {
    const node = el(id);
    node.textContent = text;
    node.classList.toggle("ok", ok);
    node.classList.toggle("bad", !ok);
}

async function poll() {
    try {
        apply(await (await fetch("/state")).json());
    } catch (e) {
        badge("badge-detector", false, "app unreachable");
    }
}

async function init() {
    const { classes } = await (await fetch("/classes")).json();
    el("label").append(...classes.map((name) => new Option(name, name)));

    el("label").addEventListener("change", (e) => post({ label: e.target.value }));
    el("conf").addEventListener("input", (e) => {
        el("conf-value").textContent = Number(e.target.value).toFixed(2);
        post({ conf: Number(e.target.value) });
    });
    el("enabled").addEventListener("change", (e) => post({ enabled: e.target.checked }));
    el("scan").addEventListener("change", (e) => post({ scan: e.target.checked }));
    el("apply-url").addEventListener("click", () => post({ server_url: el("server-url").value }));

    await poll();
    setInterval(poll, 300);
}

init();
