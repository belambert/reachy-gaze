const el = (id) => document.getElementById(id);

// Responses can arrive out of order — a poll issued before a write can resolve
// after it — so every request takes a sequence number and an older response is
// never allowed to paint over a newer one.
let seq = 0;
let applied = 0;

// Fields edited locally but not yet acknowledged by the app. Focus is the wrong
// test here: clicking Apply blurs the input first, which would expose it to a
// poll overwriting what was typed a moment before it gets read back.
const dirty = new Set();

async function request(path, options, keys = []) {
    const id = ++seq;
    let state;
    try {
        state = await (await fetch(path, options)).json();
    } catch (e) {
        badge("badge-detector", false, "app unreachable");
        return;
    }
    keys.forEach((key) => dirty.delete(key));
    if (id > applied) {
        applied = id;
        apply(state);
    }
}

const poll = () => request("/state");

const write = (config) =>
    request(
        "/config",
        {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(config),
        },
        Object.keys(config),
    );

function apply(state) {
    if (!dirty.has("label")) el("label").value = state.label;
    if (!dirty.has("conf")) el("conf").value = state.conf;
    if (!dirty.has("enabled")) el("enabled").checked = state.enabled;
    if (!dirty.has("scan")) el("scan").checked = state.scan;
    if (!dirty.has("server_url")) el("server-url").value = state.server_url;

    el("conf-value").textContent = Number(el("conf").value).toFixed(2);
    el("server-url").classList.toggle("unsaved", dirty.has("server_url"));

    badge("badge-detector", state.detector_ok, state.detector_ok ? "detector up" : "detector down");
    badge("badge-lock", state.locked, state.locked ? `locked: ${state.label}` : "searching");
    el("badge-fps").textContent = state.detector_ok ? `${state.fps} fps` : "– fps";

    const marker = el("marker");
    if (state.center && state.locked) {
        marker.style.display = "block";
        marker.style.left = `${((state.center[0] + 1) / 2) * 100}%`;
        marker.style.top = `${((state.center[1] + 1) / 2) * 100}%`;
    } else {
        marker.style.display = "none";
    }

    el("error").textContent = state.error || "";
}

function badge(id, ok, text) {
    const node = el(id);
    node.textContent = text;
    node.classList.toggle("ok", ok);
    node.classList.toggle("bad", !ok);
}

function applyUrl() {
    const url = el("server-url").value.trim();
    if (url) write({ server_url: url });
}

async function init() {
    let classes = [];
    try {
        ({ classes } = await (await fetch("/classes")).json());
    } catch (e) {
        // Losing the picker must not also cost us the status readout.
        badge("badge-detector", false, "app unreachable");
    }
    el("label").append(...classes.map((name) => new Option(name, name)));

    el("label").addEventListener("change", (e) => {
        dirty.add("label");
        write({ label: e.target.value });
    });

    // Writing on every drag event would flood the app; the label tracks live.
    let confTimer;
    el("conf").addEventListener("input", (e) => {
        dirty.add("conf");
        el("conf-value").textContent = Number(e.target.value).toFixed(2);
        clearTimeout(confTimer);
        confTimer = setTimeout(() => write({ conf: Number(e.target.value) }), 150);
    });

    for (const id of ["enabled", "scan"]) {
        el(id).addEventListener("change", (e) => {
            dirty.add(id);
            write({ [id]: e.target.checked });
        });
    }

    el("server-url").addEventListener("input", () => {
        dirty.add("server_url");
        el("server-url").classList.add("unsaved");
    });
    el("server-url").addEventListener("keydown", (e) => {
        if (e.key === "Enter") applyUrl();
    });
    el("apply-url").addEventListener("click", applyUrl);

    await poll();
    setInterval(poll, 300);
}

init();
