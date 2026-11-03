import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { Shell } from "./app/Shell";
import { installBusyFlag } from "./lib/busy";
import { installStaleChunkReload } from "./lib/staleChunk";
import { loadAddons } from "./loadAddons";
import "./styles.css";

function render(): void {
	const root = document.getElementById("root");
	if (root) {
		createRoot(root).render(
			<StrictMode>
				<Shell />
			</StrictMode>,
		);
	}
}

installBusyFlag();
installStaleChunkReload();

// Render once add-ons have settled (or after a short wait); a broken add-on never blocks it.
void loadAddons().finally(render);
