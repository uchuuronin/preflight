import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { Shell } from "./app/Shell";
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

// Render once add-ons have settled (or after a short wait); a broken add-on never blocks it.
void loadAddons().finally(render);
