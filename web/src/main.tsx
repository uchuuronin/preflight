import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { Shell } from "./app/Shell";
import { loadAddons } from "./registry";
import "./styles.css";

loadAddons();

const root = document.getElementById("root");
if (root) {
	createRoot(root).render(
		<StrictMode>
			<Shell />
		</StrictMode>,
	);
}
