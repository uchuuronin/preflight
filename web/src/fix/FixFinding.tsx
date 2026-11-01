/** "Fix it with Muse Code" under a finding: opens the race for it. Only with the fix loop on. */
import { useState } from "react";
import { flag } from "../lib/flags";
import { t } from "../lib/i18n";
import type { Finding } from "../workspace/record/types";
import { FixRace } from "./FixRace";

export function FixFinding({ finding }: { finding: Finding }) {
	const [open, setOpen] = useState(false);
	if (!finding.fix_template || !flag("c.fix_loop")) return null;
	return open ? (
		<FixRace finding={finding} />
	) : (
		<button type="button" className="pf-btn pf-btn--small" onClick={() => setOpen(true)}>
			{t("guide.fix_it")}
		</button>
	);
}
