/**
 * What the visitor can do with a suggestion, besides applying it (both tours): "Not this" puts
 * it back and rules that value out for the session, so the guide moves to another lever; "Keep"
 * puts it back and pins the knob at the visitor's own value; "Pin" keeps the suggestion and pins
 * it there. Each button names the value it leaves the knob at, so a pin is never ambiguous.
 */
import { t } from "../lib/i18n";
import type { Knob, KnobValue } from "../workspace/record/types";
import { valueText } from "./GuidePanel";

export function MoveChoices({
	knob,
	from,
	to,
	disabled,
	onDecline,
	onKeep,
	onPin,
}: {
	knob: Knob | undefined;
	from: KnobValue | undefined;
	to: KnobValue;
	disabled?: boolean;
	onDecline: () => void;
	onKeep: () => void;
	onPin: () => void;
}) {
	const name = knob ? t(knob.label_key) : "";
	return (
		<>
			<button type="button" className="pf-btn pf-btn--small" disabled={disabled} onClick={onDecline} title={t("guide.not_this.help")}>
				{t("guide.not_this")}
			</button>
			{from !== undefined ? (
				<button type="button" className="pf-btn pf-btn--small" disabled={disabled} onClick={onKeep} title={t("guide.keep.help", { knob: name })}>
					{t("guide.keep", { value: valueText(knob, from) })}
				</button>
			) : null}
			<button type="button" className="pf-btn pf-btn--small" disabled={disabled} onClick={onPin} title={t("guide.pin_at.help", { knob: name })}>
				{t("guide.pin_at", { value: valueText(knob, to) })}
			</button>
		</>
	);
}
