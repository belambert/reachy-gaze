const el = (id) => document.getElementById(id);

// Each tracked class as a glyph for the view; anything else falls back to a dot.
const EMOJI = { cat: "🐱", dog: "🐶", bird: "🐦", person: "🧍" };
const emoji = (label) => EMOJI[label] ?? "🎯";

// Mirrors STALE_AFTER in the app. Only affects how the lock badge reads, so a
// drift between the two costs nothing but wording.
const STALE_AFTER = 1.0;

// Responses can arrive out of order — a poll issued before a write can resolve
// after it — so every request takes a sequence number and an older response is
// never allowed to paint over a newer one.
let seq = 0;
let applied = 0;

// Fields edited locally but not yet acknowledged by the app. Focus is the wrong
// test here: clicking Apply blurs the input first, which would expose it to a
// poll overwriting what was typed a moment before it gets read back.
const dirty = new Set();

// The backends the app offers, and the option keys currently rendered. Kept so
// switching backend can seed its default URL and the list is rebuilt only when
// it actually changes.
let backends = [];
let backendKeys = "";

async function request(path, options, keys = []) {
    const id = ++seq;
    let state;
    try {
        const response = await fetch(path, options);
        // A 404 body is still valid JSON, so status has to be checked directly.
        if (!response.ok) throw new Error(`${path} returned ${response.status}`);
        state = await response.json();
    } catch (e) {
        badge("badge-detector", "bad", "app unreachable");
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
    el("labels").textContent = list(state.labels);

    if (!dirty.has("conf")) el("conf").value = state.conf;
    if (!dirty.has("pull")) el("pull").value = state.pull;
    if (!dirty.has("enabled")) el("enabled").checked = state.enabled;
    if (!dirty.has("scan")) el("scan").checked = state.scan;
    if (!dirty.has("server_url")) el("server-url").value = state.server_url;

    backends = state.backends ?? [];
    const keys = backends.map((b) => b.key).join(",");
    if (keys !== backendKeys) {
        backendKeys = keys;
        el("backend").replaceChildren(...backends.map((b) => new Option(b.label, b.key)));
    }
    if (!dirty.has("backend")) el("backend").value = state.backend;

    el("conf-value").textContent = Number(el("conf").value).toFixed(2);
    el("pull-value").textContent = Number(el("pull").value).toFixed(0);
    el("server-url").classList.toggle("unsaved", dirty.has("server_url"));

    badge(
        "badge-detector",
        state.detector_ok ? "ok" : "bad",
        state.detector_ok ? "detector up" : "detector down",
    );

    // The head holds its aim long after the last sighting, so saying "locked"
    // for all of it would misreport a target that left seconds ago.
    if (!state.locked) {
        badge("badge-lock", "bad", "searching");
    } else if (state.seen_ago < STALE_AFTER) {
        badge("badge-lock", "ok", `locked: ${state.label}`);
    } else {
        badge("badge-lock", "warn", `holding: ${state.label} (${Math.round(state.seen_ago)}s)`);
    }

    el("badge-fps").textContent = state.detector_ok ? `${state.fps} fps` : "– fps";

    const marker = el("marker");
    if (state.center && state.locked) {
        marker.style.display = "flex";
        marker.textContent = emoji(state.label);
        marker.title = state.label;
        place(marker, state.center);
    } else {
        marker.style.display = "none";
    }
    renderTargets(state.targets ?? []);
    el("aim").textContent = aimText(state.aim);
    renderWorld(state.world ?? []);

    el("error").textContent = state.error || "";
}

// Head aim in words: +yaw is left, +pitch is up (see look_yaw_pitch in main.py).
function aimText(aim) {
    return aim ? `Aimed ${bearing(aim)}.` : "";
}

// A world direction in words, e.g. "20° left and 5° up".
function bearing({ yaw, pitch }) {
    yaw = Math.round(yaw);
    pitch = Math.round(pitch);
    if (!yaw && !pitch) return "straight ahead";
    const words = [];
    if (yaw) words.push(`${Math.abs(yaw)}° ${yaw > 0 ? "left" : "right"}`);
    if (pitch) words.push(`${Math.abs(pitch)}° ${pitch > 0 ? "up" : "down"}`);
    return words.join(" and ");
}

// The world model: everything seen lately, newest first, with its age.
function renderWorld(objects) {
    el("world").replaceChildren(
        ...objects.map((obj) => {
            const node = document.createElement("li");
            node.textContent = `${emoji(obj.label)} ${obj.label}, ${bearing(obj)}, ${Math.round(obj.age)}s ago`;
            return node;
        }),
    );
}

// A marker's position from a center normalized to [-1, 1] on both axes.
function place(node, [x, y]) {
    node.style.left = `${((x + 1) / 2) * 100}%`;
    node.style.top = `${((y + 1) / 2) * 100}%`;
}

// The other boxes in view: rebuilt each poll, which is plenty at this rate.
function renderTargets(targets) {
    el("targets").replaceChildren(
        ...targets.map(({ label, center }) => {
            const node = document.createElement("div");
            node.className = "target";
            node.textContent = emoji(label);
            node.title = label;
            place(node, center);
            return node;
        }),
    );
}

// "people, cats and dogs" reads better on the card than a bare CSV.
function list(labels = []) {
    const plural = labels.map((n) => (n === "person" ? "people" : `${n}s`));
    return plural.length < 2
        ? plural.join("")
        : `${plural.slice(0, -1).join(", ")} and ${plural.at(-1)}`;
}

function badge(id, tone, text) {
    const node = el(id);
    node.textContent = text;
    for (const name of ["ok", "warn", "bad"]) {
        node.classList.toggle(name, name === tone);
    }
}

function applyUrl() {
    const url = el("server-url").value.trim();
    if (url) write({ server_url: url });
}

async function init() {
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

    // Switching backend seeds its default address, so the change takes effect
    // at once rather than pointing a new client at the old backend's URL.
    el("backend").addEventListener("change", (e) => {
        const spec = backends.find((b) => b.key === e.target.value);
        dirty.add("backend");
        if (spec) {
            dirty.add("server_url");
            el("server-url").value = spec.default_url;
            el("server-url").classList.add("unsaved");
            write({ backend: e.target.value, server_url: spec.default_url });
        } else {
            write({ backend: e.target.value });
        }
    });

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
