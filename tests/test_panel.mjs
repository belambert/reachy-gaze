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

const SOURCE = process.argv[2] ?? "tracker/static/main.js";
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

function harness(source, { classesStatus = 200 } = {}) {
    const ids = [
        "label", "conf", "conf-value", "pull", "pull-value", "enabled", "scan",
        "server-url", "apply-url", "badge-detector", "badge-lock", "badge-fps",
        "marker", "error",
    ];
    const els = Object.fromEntries(ids.map((i) => [i, makeEl(i)]));
    const app = {
        enabled: false, label: "person", conf: 0.4, pull: 20,
        server_url: "http://old:8100", scan: true, locked: false,
        detector_ok: true, error: "", fps: 0, center: null, classes_version: 0,
        seen_ago: null,
    };
    const posts = [];
    const delays = { state: 0, config: 0 };
    const status = { classes: classesStatus };
    const vocabulary = ["person", "cat", "dog"];

    const ok = (body) => ({ ok: true, status: 200, json: async () => body });

    async function fetchStub(path, options) {
        if (path === "/classes") {
            if (status.classes !== 200) {
                // FastAPI's 404 body is valid JSON, which is what made this
                // failure mode so quiet in the first place.
                return {
                    ok: false,
                    status: status.classes,
                    json: async () => ({ detail: "Not Found" }),
                };
            }
            return ok({ classes: vocabulary.slice() });
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
        document: { getElementById: (id) => els[id] ?? null, activeElement: null },
        fetch: fetchStub,
        Option: function (text, value) { return { text, value }; },
        setTimeout, clearTimeout, setInterval, clearInterval, console,
    };
    vm.createContext(ctx);
    vm.runInContext(fs.readFileSync(source, "utf8"), ctx);
    return { els, app, posts, delays, status, vocabulary };
}

const tests = {
    // Bug: mousedown blurs the input, so a poll landing between blur and click
    // reset .value, and the click handler then posted the stale URL back.
    async "the class picker is populated from the app"() {
        const h = harness(SOURCE);
        await sleep(30);
        assert.deepEqual(
            h.els.label.options.map((o) => o.value),
            ["person", "cat", "dog"],
            "the picker must list what /classes returned",
        );
    },

    async "a missing /classes is reported, not swallowed"() {
        const h = harness(SOURCE, { classesStatus: 404 });
        await sleep(60);

        assert.match(h.els.error.textContent, /Could not load classes/);
        assert.match(h.els.error.textContent, /404/);
    },

    async "a picker that failed to load recovers on a later poll"() {
        // The real failure: one fetch at startup lost a race and the dropdown
        // stayed empty for as long as the page was open.
        const h = harness(SOURCE, { classesStatus: 404 });
        await sleep(60);
        assert.equal(h.els.label.options.length, 0);

        h.status.classes = 200;
        await sleep(700);
        assert.deepEqual(
            h.els.label.options.map((o) => o.value),
            ["person", "cat", "dog"],
            "polling must retry the class list",
        );
        assert.equal(h.els.error.textContent, "", "the error must clear");
    },

    async "the picker follows a change of vocabulary"() {
        const h = harness(SOURCE);
        await sleep(60);
        assert.equal(h.els.label.options.length, 3);

        h.vocabulary.length = 0;
        h.vocabulary.push("robot", "mug");
        h.app.classes_version = 1;

        await sleep(700);
        assert.deepEqual(h.els.label.options.map((o) => o.value), ["robot", "mug"]);
    },

    async "an unchanged vocabulary is not refetched"() {
        const h = harness(SOURCE);
        await sleep(60);
        const first = h.els.label.options;
        await sleep(700);
        assert.equal(h.els.label.options, first, "must not rebuild every poll");
    },

    async "the rest of the panel still works when /classes is missing"() {
        const h = harness(SOURCE, { classesStatus: 404 });
        await sleep(30);

        // Listeners and polling must still have been wired up.
        h.els.enabled.checked = true;
        h.els.enabled.fire("change");
        await sleep(60);
        assert.equal(h.app.enabled, true, "controls must survive an empty picker");
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
