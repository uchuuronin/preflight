import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

export default defineConfig({
	plugins: [react()],
	server: {
		proxy: {
			// Local API during development; production uses VITE_API_BASE.
			"/v1": "http://127.0.0.1:8000",
		},
	},
	test: {
		environment: "jsdom",
	},
});
