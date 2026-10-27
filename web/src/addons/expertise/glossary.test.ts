import { beforeAll, describe, expect, it } from "vitest";
import en from "../../i18n/en.json";
import { t } from "../../lib/i18n";
import { GLOSSARY, KNOB_TERMS, glossaryEntry, termForKnob } from "./glossary";
import { TIERS } from "./store";
import { registerExpertiseStrings } from "./strings";

type RecordKnobs = { knobs?: Array<{ id: string }> };

// A's golden records and C's dev record: every knob a visitor can meet today.
const RECORDS = {
	...import.meta.glob<RecordKnobs>("../../../../schema/examples/*.json", { eager: true, import: "default" }),
	...import.meta.glob<RecordKnobs>("../../workspace/fixtures/*.record.json", { eager: true, import: "default" }),
};

// Every knob C has an English label for: A's full declared set, not only the samples.
const NOT_KNOBS = new Set(["proposed", "pinned", "pinned_hint", "unpin_so_it_moves", "pin", "needs_code", "not_trusted_here"]);
const LABELLED_KNOBS = Object.keys(en)
	.map((k) => /^knob\.([a-z0-9_]+)$/.exec(k)?.[1])
	.filter((id): id is string => !!id && !NOT_KNOBS.has(id));

function recordKnobIds(): string[] {
	const ids = new Set<string>();
	for (const rec of Object.values(RECORDS)) for (const k of rec.knobs ?? []) ids.add(k.id);
	return [...ids].sort();
}

// British spellings the copy must use, and the US forms that give a slip away.
const US_FORMS = /\b(optimiz\w*|quantiz\w*|normaliz\w*|utiliz\w*|colors?|behaviors?|initializ\w*|materializ\w*|analyz\w*|summariz\w*|neighbors?|labeled|modeled|center)\b/i;

describe("glossary", () => {
	beforeAll(registerExpertiseStrings);

	it("reads the records it checks against", () => {
		expect(Object.keys(RECORDS).length).toBeGreaterThanOrEqual(4);
		expect(recordKnobIds()).toContain("eval_batch_size");
	});

	it("explains every knob the records declare", () => {
		const missing = recordKnobIds().filter((id) => !termForKnob(id));
		expect(missing).toEqual([]);
	});

	it("explains every knob C has a label for", () => {
		expect(LABELLED_KNOBS.length).toBeGreaterThan(50);
		expect(LABELLED_KNOBS.filter((id) => !termForKnob(id))).toEqual([]);
	});

	it("points every knob at a term that exists", () => {
		const dangling = Object.entries(KNOB_TERMS).filter(([, term]) => !GLOSSARY[term]);
		expect(dangling).toEqual([]);
	});

	it("gives every term a label and a line for each level", () => {
		for (const [id, term] of Object.entries(GLOSSARY)) {
			expect(term.label, id).toBeTruthy();
			for (const tier of TIERS) expect(term[tier], `${id}.${tier}`).toBeTruthy();
		}
	});

	it("keeps copy free of em dashes and US spellings", () => {
		for (const [id, term] of Object.entries(GLOSSARY)) {
			const text = [term.label, term.new, term.practitioner, term.expert, term.act ?? ""].join(" ");
			expect(text.includes("\u2014"), `${id} has an em dash`).toBe(false);
			expect(US_FORMS.test(text), `${id}: ${text.match(US_FORMS)?.[0]}`).toBe(false);
		}
	});

	it("keeps each line short enough for a card", () => {
		for (const [id, term] of Object.entries(GLOSSARY)) {
			for (const tier of TIERS) expect(term[tier].length, `${id}.${tier}`).toBeLessThanOrEqual(180);
		}
	});

	it("resolves through t() at each level", () => {
		const entry = glossaryEntry("eval_batch_size", "new");
		expect(entry).not.toBeNull();
		expect(t(entry!.titleKey)).toBe("Eval batch");
		expect(t(entry!.bodyKey)).toMatch(/crash a run that trained fine/);
		expect(entry!.actKey && t(entry!.actKey)).toMatch(/no_grad/);
		expect(t(glossaryEntry("eval_batch_size", "expert")!.bodyKey)).toMatch(/first-eval OOM/);
	});

	it("returns nothing for a knob it does not know", () => {
		expect(glossaryEntry("not_a_knob", "new")).toBeNull();
	});
});
