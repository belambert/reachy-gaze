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

// Set if the class list could not be loaded. Kept apart from the app's own
// error so a later successful poll cannot quietly paint over it.
let loadError = "";

async function request(path, options, keys = []) {
    const id = ++seq;
    let state;
    try {
        const response = await fetch(path, options);
        // A 404 body is still valid JSON, so status has to be checked directly.
        if (!response.ok) throw new Error(`${path} returned ${response.status}`);
        state = await response.json();
    } catch (e) {
        badge("badge-detector", false, "app unreachable");
        el("error").textContent = e.message;
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
    if (!dirty.has("pull")) el("pull").value = state.pull;
    if (!dirty.has("enabled")) el("enabled").checked = state.enabled;
    if (!dirty.has("scan")) el("scan").checked = state.scan;
    if (!dirty.has("server_url")) el("server-url").value = state.server_url;

    el("conf-value").textContent = Number(el("conf").value).toFixed(2);
    el("pull-value").textContent = Number(el("pull").value).toFixed(0);
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

    el("error").textContent = state.error || loadError || "";
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
        const response = await fetch("/classes");
        if (!response.ok) throw new Error(`/classes returned ${response.status}`);
        const body = await response.json();
        if (!Array.isArray(body.classes)) throw new Error("/classes sent no list");
        classes = body.classes;
    } catch (e) {
        // Losing the picker must not cost us the controls or the status readout,
        // and the reason has to be visible rather than only in the console.
        badge("badge-detector", false, "app unreachable");
        loadError = `Could not load classes: ${e.message}`;
        el("error").textContent = loadError;
    }
    el("label").append(...classes.map((name) => new Option(name, name)));

    el("label").addEventListener("change", (e) => {
        dirty.add("label");
        write({ label: e.target.value });
    });

    // Writing on every drag event would flood the app; the label tracks live.
    for (const [id, digits] of [["conf", 2], ["pull", 0]]) {
        let timer;
        el(id).addEventListener("input", (e) => {
            dirty.add(id);
            el(`${id}-value`).textContent = Number(e.target.value).toFixed(digits);
            clearTimeout(timer);
            timer = setTimeout(() => write({ [id]: Number(e.target.value) }), 150);
        });
    }

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
