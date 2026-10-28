/**
 * "Show costs in": the currency setting, docked in the Logbook's pilot card (registry slot
 * "logbook-settings"). A native select, so it works with every keyboard and screen reader.
 */
import { useId } from "react";
import { t } from "../../lib/i18n";
import { CURRENCIES, RATES } from "./rates";
import { autoCurrency, resolvedCurrency, setChoice, useCurrencyChoice } from "./store";

function nameOf(code: string): string {
	return t(`addon.currency.name.${code}`);
}

function asOf(iso: string): string {
	return new Intl.DateTimeFormat("en-GB", { day: "numeric", month: "short", year: "numeric", timeZone: "UTC" }).format(new Date(`${iso}T00:00:00Z`));
}

export function CurrencySetting() {
	const choice = useCurrencyChoice();
	const noteId = useId();
	const code = resolvedCurrency();
	const auto = autoCurrency() ?? "USD";
	return (
		<div className="pf-addon-currency">
			<select
				className="pf-addon-currency__select"
				aria-label={t("addon.currency.title")}
				aria-describedby={code === "USD" ? undefined : noteId}
				value={choice}
				onChange={(e) => setChoice(e.currentTarget.value)}
			>
				<option value="auto">{t("addon.currency.auto", { name: nameOf(auto) })}</option>
				{CURRENCIES.map((c) => (
					<option key={c} value={c}>
						{t("addon.currency.option", { name: nameOf(c), code: c })}
					</option>
				))}
			</select>
			{code === "USD" ? null : (
				<p id={noteId} className="pf-addon-currency__note">
					{t("addon.currency.note", { date: asOf(RATES.as_of) })}{" "}
					<a href={RATES.url} target="_blank" rel="noopener noreferrer">
						{t("addon.currency.source")}
					</a>
					.
				</p>
			)}
		</div>
	);
}
