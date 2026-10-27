import { useEffect, useMemo, useState } from "react";
import { t } from "./lib/i18n";
import { GridProvider } from "./workspace/provider/GridProvider";
import type { FlightRecord } from "./workspace/record/types";
import { WorkspaceProvider } from "./workspace/state";
import { Workspace } from "./workspace/Workspace";
import { loadRecord } from "./workspace/record/load";

type Load = { kind: "loading" } | { kind: "ready"; record: FlightRecord } | { kind: "error"; message: string };

export function App() {
	const [load, setLoad] = useState<Load>({ kind: "loading" });
	const provider = useMemo(() => new GridProvider(), []);

	useEffect(() => {
		let alive = true;
		loadRecord(new URLSearchParams(window.location.search))
			.then((record) => alive && setLoad({ kind: "ready", record }))
			.catch((e: unknown) => alive && setLoad({ kind: "error", message: e instanceof Error ? e.message : String(e) }));
		return () => {
			alive = false;
		};
	}, []);

	if (load.kind === "loading") {
		return (
			<div className="pf-center-screen" role="status">
				{t("app.loading")}
			</div>
		);
	}
	if (load.kind === "error") {
		return (
			<div className="pf-center-screen" role="alert">
				<h1>{t("app.load_failed")}</h1>
				<p>{load.message}</p>
				<a className="pf-btn" href="/">
					{t("app.open_sample")}
				</a>
			</div>
		);
	}
	return (
		<WorkspaceProvider record={load.record} provider={provider}>
			<Workspace />
		</WorkspaceProvider>
	);
}
