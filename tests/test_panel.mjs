// Regression tests for the control panel's state sync.
//
// The panel polls /state every 300 ms while the user is editing, so responses
// routinely arrive out of order. These tests drive main.js against a stubbed
// DOM and a stubbed app whose latency we control, reproducing the interleavings
// that a browser produces only intermittently.
//
//     node tests/test_panel.mjs [path/to/main.js]

import assert from "node:assert/strict";
import fs from "node:fs";
import process from "node:process";
import vm from "node:vm";

const SOURCE = process.argv[2] ?? "reachy_gaze/static/main.js";
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function makeEl(id) {
    const handlers = {};
    return {
        id,
        value: "",
        checked: false,
        textContent: "",
        style: {},
        classList: {
            set: new Set(),
            add(c) { this.set.add(c); },
            remove(c) { this.set.delete(c); },
            toggle(c, on) { on ? this.set.add(c) : this.set.delete(c); },
            contains(c) { return this.set.has(c); },
        },
        addEventListener(ev, fn) { (handlers[ev] ??= []).push(fn); },
        options: [],
        append(...opts) { this.options.push(...opts); },
        replaceChildren(...opts) { this.options = opts; },
        fire(ev, extra = {}) {
            for (const fn of handlers[ev] ?? []) fn({ target: this, ...extra });
        },
    };
}

function harness(source, { stateStatus = 200 } = {}) {
    const ids = [
        "labels", "conf", "conf-value", "pull", "pull-value", "enabled", "scan",
        "server-url", "apply-url", "backend", "badge-detector", "badge-lock",
        "badge-fps", "marker", "error", "aim", "targets",
        "view", "rings",
    ];
    const els = Object.fromEntries(ids.map((i) => [i, makeEl(i)]));
    const app = {
        enabled: true, labels: ["person", "cat", "dog", "bird"], label: "",
        conf: 0.4, pull: 20, server_url: "http://old:8100", scan: true,
        locked: false, detector_ok: true, error: "", fps: 0, center: null,
        seen_ago: null, backend: "triton",
        backends: [
            { key: "triton", label: "Triton (vision-server)", default_url: "localhost:8101" },
            { key: "builtin", label: "Built-in server", default_url: "http://localhost:8100" },
        ],
    };
    const posts = [];
    const delays = { state: 0, config: 0 };
    const status = { state: stateStatus };

    const ok = (body) => ({ ok: true, status: 200, json: async () => body });

    async function fetchStub(path, options) {
        if (path === "/state" && status.state !== 200) {
            // FastAPI's 404 body is still valid JSON, which is what made this
            // failure mode so quiet in the first place.
            return {
                ok: false,
                status: status.state,
                json: async () => ({ detail: "Not Found" }),
            };
        }
        if (path === "/state") {
            // Snapshot before sleeping: that is what makes a slow reply stale.
            const snap = { ...app };
            await sleep(delays.state);
            return ok(snap);
        }
        if (path === "/config") {
            const body = JSON.parse(options.body);
            posts.push(body);
            Object.assign(app, body);
            const snap = { ...app };
            await sleep(delays.config);
            return ok(snap);
        }
        throw new Error(`unexpected path ${path}`);
    }

    const ctx = {
        document: {
            getElementById: (id) => els[id] ?? null,
            createElement: (tag) => makeEl(tag),
            activeElement: null,
        },
        fetch: fetchStub,
        Option: function (text, value) { return { text, value }; },
        setTimeout, clearTimeout, setInterval, clearInterval, console,
    };
    vm.createContext(ctx);
    vm.runInContext(fs.readFileSync(source, "utf8"), ctx);
    return { els, app, posts, delays, status };
}

const tests = {
    async "the panel names what the app is looking for"() {
        const h = harness(SOURCE);
        await sleep(30);
        assert.equal(h.els.labels.textContent, "people, cats, dogs and birds");
    },

    async "a single label still reads properly"() {
        const h = harness(SOURCE);
        h.app.labels = ["cat"];
        await sleep(400);
        assert.equal(h.els.labels.textContent, "cats");
    },

    async "the backend selector lists the options and marks the current one"() {
        const h = harness(SOURCE);
        await sleep(30);
        assert.deepEqual(h.els.backend.options.map((o) => o.value), ["triton", "builtin"]);
        assert.equal(h.els.backend.value, "triton");
    },

    async "switching backend seeds its default address and posts both"() {
        const h = harness(SOURCE);
        await sleep(30);

        h.els.backend.value = "builtin";
        h.els.backend.fire("change");
        await sleep(30);

        const post = h.posts.at(-1);
        assert.equal(post.backend, "builtin");
        assert.equal(post.server_url, "http://localhost:8100");
        assert.equal(h.els["server-url"].value, "http://localhost:8100");
    },

    async "an unreachable app is reported, not swallowed"() {
        const h = harness(SOURCE, { stateStatus: 404 });
        await sleep(60);

        assert.match(h.els.error.textContent, /404/);
        assert.ok(h.els["badge-detector"].classList.contains("bad"));
    },

    async "the lock badge distinguishes a live target from a held aim"() {
        const h = harness(SOURCE);
        await sleep(60);
        const lock = h.els["badge-lock"];

        assert.equal(lock.textContent, "searching");
        assert.ok(lock.classList.contains("bad"));

        Object.assign(h.app, { locked: true, seen_ago: 0.2, label: "cat" });
        await sleep(400);
        assert.equal(lock.textContent, "locked: cat");
        assert.ok(lock.classList.contains("ok"));

        // The head still aims there, but the cat was last seen 6s ago: saying
        // "locked" here is what would mislead.
        Object.assign(h.app, { locked: true, seen_ago: 6.0 });
        await sleep(400);
        assert.equal(lock.textContent, "holding: cat (6s)");
        assert.ok(lock.classList.contains("warn"));
        assert.ok(!lock.classList.contains("ok"), "tones must be exclusive");

        Object.assign(h.app, { locked: false, seen_ago: null });
        await sleep(400);
        assert.equal(lock.textContent, "searching");
    },

    async "the aim readout uses the same arrows"() {
        const h = harness(SOURCE);
        h.app.aim = { yaw: -20.4, pitch: 5 };
        await sleep(400);
        assert.equal(h.els.aim.textContent, "Aimed →20° ↑5°.");

        h.app.aim = { yaw: 0.3, pitch: -0.2 };
        await sleep(400);
        assert.equal(h.els.aim.textContent, "Aimed straight ahead.");
    },

    async "rings are drawn every 10 degrees out to the frame's corners"() {
        const h = harness(SOURCE);
        // 90° horizontal FOV on a 640x480 frame: corners sit ~51° off axis.
        h.app.lens = { fx: 320, fy: 320, cx: 320, cy: 240, width: 640, height: 480 };
        await sleep(400);

        const labels = h.els.rings.options.filter((n) => n.className === "ring-label");
        assert.deepEqual(labels.map((n) => n.textContent), ["10°", "20°", "30°", "40°", "50°"]);
        assert.equal(h.els.view.style.aspectRatio, "640 / 480");

        // At 320 px focal length the 40° ring is 537 px wide, inside the frame.
        const ring40 = h.els.rings.options.filter((n) => n.className === "ring")[3];
        assert.ok(parseFloat(ring40.style.width) < 100);
    },

    async "no lens, no rings"() {
        const h = harness(SOURCE);
        await sleep(60);
        assert.deepEqual(h.els.rings.options, []);
    },

    async "a stale poll must not steal what was typed"() {
        const h = harness(SOURCE);
        await sleep(30);
        assert.equal(h.els["server-url"].value, "http://old:8100");

        h.delays.state = 400;
        await sleep(320); // the interval fires a poll that is now in flight

        h.els["server-url"].value = "http://new:8100";
        h.els["server-url"].fire("input");
        h.els["apply-url"].fire("click");

        await sleep(600); // config lands, then the stale poll lands
        assert.equal(h.posts.at(-1)?.server_url, "http://new:8100",
            "the typed URL must be what gets posted");
        assert.equal(h.els["server-url"].value, "http://new:8100",
            "the applied URL must survive the late poll");
        assert.equal(h.app.server_url, "http://new:8100");
    },

    async "a stale poll must not revert the slider"() {
        const h = harness(SOURCE);
        await sleep(30);

        h.delays.state = 500;
        await sleep(320);

        h.els.conf.value = 0.8;
        h.els.conf.fire("input");

        await sleep(700);
        assert.equal(Number(h.els.conf.value), 0.8, "slider must not jump back");
        assert.equal(h.app.conf, 0.8);
    },

    async "a stale poll must not revert a toggle"() {
        const h = harness(SOURCE);
        await sleep(30);

        h.delays.state = 500;
        await sleep(320);

        h.els.enabled.checked = true;
        h.els.enabled.fire("change");

        await sleep(700);
        assert.equal(h.els.enabled.checked, true, "toggle must not flip back");
        assert.equal(h.app.enabled, true);
    },

    async "a stale poll must not revert the responsiveness slider"() {
        const h = harness(SOURCE);
        await sleep(30);

        h.delays.state = 500;
        await sleep(320);

        h.els.pull.value = 8;
        h.els.pull.fire("input");

        await sleep(700);
        assert.equal(Number(h.els.pull.value), 8, "slider must not jump back");
        assert.equal(h.app.pull, 8);
    },

    async "each slider writes only its own field"() {
        const h = harness(SOURCE);
        await sleep(30);

        h.els.pull.value = 12;
        h.els.pull.fire("input");
        await sleep(300);

        assert.deepEqual(h.posts.at(-1), { pull: 12 }, "must not resend conf");
        assert.equal(h.app.conf, 0.4, "conf must be untouched");
    },

    async "Enter applies the URL"() {
        const h = harness(SOURCE);
        await sleep(30);
        h.els["server-url"].value = "http://typed:9000";
        h.els["server-url"].fire("input");
        h.els["server-url"].fire("keydown", { key: "Enter" });
        await sleep(60);
        assert.equal(h.app.server_url, "http://typed:9000");
    },

    async "a blank URL is not submitted"() {
        const h = harness(SOURCE);
        await sleep(30);
        h.els["server-url"].value = "   ";
        h.els["server-url"].fire("input");
        h.els["apply-url"].fire("click");
        await sleep(60);
        assert.equal(h.app.server_url, "http://old:8100");
    },
};

let failed = 0;
for (const [name, fn] of Object.entries(tests)) {
    try {
        await fn();
        console.log(`  ok   ${name}`);
    } catch (e) {
        failed++;
        console.log(`  FAIL ${name}\n       ${e.message.split("\n")[0]}`);
    }
}
console.log(failed ? `\n${failed} failed` : `\n${Object.keys(tests).length} passed`);
process.exit(failed ? 1 : 0);
