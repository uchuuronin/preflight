// Audit finding 12: Boarding shows a Meta model's tag only where the record's provenance says that
// model was called, for this job. Mocks jobs (events and record) at desk and phone size, compares
// each leg's tags with what the record allows, and exits 1 on any mismatch. One case opens a job
// that ran triage and then, in the same page, a job that failed before any call: the second must
// show no tags. Read-only; run against the dev server:
//   NODE_PATH=web/node_modules node docs/audit/scripts/model-tags.mjs http://localhost:5185 out
import { createRequire } from "node:module";
const { chromium } = createRequire(import.meta.url)("playwright");
const [base, out] = process.argv.slice(2);
if (!base || !out) {
	console.error("usage: model-tags.mjs <dev server url> <out dir>");
	process.exit(2);
}
const sse = (ev) => ev.map((e, i) => `data: ${JSON.stringify({ job_id: "j", seq: i, at: "2026-11-01T00:00:00Z", ...e })}\n\n`).join("");
const MAP = { provider: "meta", model: "muse-spark-1.3", purpose: "script_map", ok: true };
const TRIAGE = { provider: "baseten", model: "meta-models/muse-spark-open-1026", purpose: "triage", ok: true };
const LEGS = ["Waiting for a GPU", "Reading your script", "Running probe steps", "Checking the logs", "Fitting ranges for every knob"];
const NONE = [[], [], [], [], []];
const SPARK = [[], ["Muse Spark"], [], [], []];
const BOTH = [[], ["Muse Spark"], [], ["Muse Spark Open"], []];
const FAILED_EARLY = [{ stage: "queued" }, { stage: "admitted" }, { stage: "mapping" }, { stage: "probing" }, { stage: "failed", error_code: "oom", message: "error.oom" }];
const TRIAGED = [{ stage: "queued" }, { stage: "mapping" }, { stage: "skeleton", share_token: "tok2", record_rev: 1 }, { stage: "probing", record_rev: 2 }, { stage: "triage", record_rev: 3 }, { stage: "done", record_rev: 4 }];
// Each case: the jobs opened one after another in the same page, and the tags the last must show.
const CASES = {
	"failed before any call (no record)": { jobs: [{ events: FAILED_EARLY, prov: null }], want: NONE },
	"done, script map only": { jobs: [{ events: [{ stage: "queued" }, { stage: "mapping" }, { stage: "skeleton", share_token: "tok1", record_rev: 1 }, { stage: "probing", record_rev: 2 }, { stage: "predicting", record_rev: 3 }, { stage: "done", record_rev: 4 }], prov: [MAP] }], want: SPARK },
	"done, script map and Spark Open triage": { jobs: [{ events: TRIAGED, prov: [MAP, TRIAGE] }], want: BOTH },
	"failed after triage ran": { jobs: [{ events: [...TRIAGED.slice(0, 5), { stage: "failed", error_code: "oom", message: "error.oom" }], prov: [MAP, TRIAGE] }], want: BOTH },
	"done, the script-map call failed": { jobs: [{ events: [{ stage: "queued" }, { stage: "skeleton", share_token: "tok4", record_rev: 1 }, { stage: "done", record_rev: 2 }], prov: [{ ...MAP, ok: false }] }], want: NONE },
	"a triaged job, then in the same page one that failed before any call": { jobs: [{ events: TRIAGED, prov: [MAP, TRIAGE] }, { events: FAILED_EARLY, prov: null }], want: NONE },
};
const b = await chromium.launch();
let failures = 0;
let shot = 0;
for (const [name, c] of Object.entries(CASES)) {
	for (const [vn, vp] of [["desk", { width: 1440, height: 900 }], ["phone", { width: 390, height: 844 }]]) {
		const page = await b.newPage({ viewport: vp });
		// Job ids are job_<case>_<n>; each request is answered with that job's events and record.
		const jobOf = (url) => c.jobs[Number(/job_\d+_(\d+)/.exec(url)?.[1] ?? 0)];
		const tokenJob = new Map(c.jobs.flatMap((j) => j.events.filter((e) => e.share_token).map((e) => [e.share_token, j])));
		await page.route("**/v1/jobs/*/events", (x) => x.fulfill({ status: 200, headers: { "content-type": "text/event-stream" }, body: sse(jobOf(x.request().url()).events) }));
		await page.route("**/v1/records/*", (x) => {
			const job = tokenJob.get(decodeURIComponent(x.request().url().split("/v1/records/")[1].split("?")[0]));
			return job?.prov ? x.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ provenance: job.prov }) }) : x.fulfill({ status: 404, body: "" });
		});
		const caseId = shot++;
		for (let n = 0; n < c.jobs.length; n++) {
			if (n === 0) await page.goto(`${base}/#/boarding/job_${caseId}_${n}`);
			else {
				await page.evaluate(() => (location.hash = "#/new"));
				await page.waitForTimeout(600);
				await page.evaluate((h) => (location.hash = h), `#/boarding/job_${caseId}_${n}`);
			}
			await page.waitForTimeout(2500);
		}
		const got = await page.locator(".pf-stage").evaluateAll((els) => els.map((e) => [...e.querySelectorAll(".pf-model")].map((m) => m.innerText)));
		const ok = JSON.stringify(got) === JSON.stringify(c.want);
		if (!ok) failures++;
		console.log(`${ok ? "ok  " : "FAIL"} ${name} · ${vn}: ${LEGS.map((l, i) => `${l} [${(got[i] ?? []).join(", ")}]`).join("; ")}`);
		if (vn === "phone") await page.screenshot({ path: `${out}/model-tags-${caseId}.png`, fullPage: true });
		await page.close();
	}
}
await b.close();
console.log(`${Object.keys(CASES).length * 2 - failures} of ${Object.keys(CASES).length * 2} pass`);
process.exit(failures ? 1 : 0);
