import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import { loadAddons } from "./registry";
import { applyTokenSet } from "./theme/tokens";
import "./workspace/registerDefaults";
import "./styles.css";

applyTokenSet("light");
loadAddons();

const root = document.getElementById("root");
if (root) {
	createRoot(root).render(
		<StrictMode>
			<App />
		</StrictMode>,
	);
}
