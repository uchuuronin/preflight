/**
 * Free and starter GPU tiers (D4). Each figure carries where it came from and when it was
 * read; "reported" marks numbers the platform itself doesn't publish. GPU names are B's
 * canonical names (engine/gpus.py), so capacities come from the same table as everything else.
 */

export interface Source {
	label: string;
	url: string;
	/** Date D read it, YYYY-MM-DD. */
	read: string;
}

export interface Preset {
	id: "colab" | "kaggle" | "lightning";
	/** GPUs the tier offers, one per session for this check. */
	gpus: readonly string[];
	/** Longest single session before the runtime stops or restarts. */
	sessionCapS: number;
	/** The GPU list is what users report, not what the platform publishes. */
	gpusReported: boolean;
	sources: readonly Source[];
}

const H = 3600;

const COLAB_FAQ: Source = {
	label: "Colab FAQ",
	url: "https://research.google.com/colaboratory/faq.html",
	read: "2026-10-27",
};

const GPUPERHOUR: Source = {
	label: "GPU Per Hour, Free cloud GPUs in 2026 (21 Sep 2026)",
	url: "https://gpuperhour.com/blog/free-cloud-gpus-and-credits",
	read: "2026-10-27",
};

const LIGHTNING: Source = {
	label: "Usage Pricing, Lightning AI plan page (captured 22 Oct 2026)",
	url: "https://usagepricing.com/blueprint/lightning-ai",
	read: "2026-10-27",
};

export const PRESETS: readonly Preset[] = [
	{
		// "Notebooks can run for at most 12 hours, depending on availability and your usage
		// patterns." Colab names no GPU and publishes no idle timeout or quota; users report a T4.
		id: "colab",
		gpus: ["T4"],
		sessionCapS: 12 * H,
		gpusReported: true,
		sources: [COLAB_FAQ, GPUPERHOUR],
	},
	{
		// Kaggle's docs: 12 hours of execution per GPU session. T4 and P100 class GPUs; the weekly
		// quota shows in account settings (users report about 30 hours).
		id: "kaggle",
		gpus: ["T4", "P100-16GB"],
		sessionCapS: 12 * H,
		gpusReported: false,
		sources: [GPUPERHOUR],
	},
	{
		// Free plan: one Studio that restarts every 4 hours; GPUs run on a one-time starter grant of
		// up to 30 credits (the monthly credits ended in September 2026).
		id: "lightning",
		gpus: ["L40S", "H100-SXM-80GB", "H200-SXM-141GB"],
		sessionCapS: 4 * H,
		gpusReported: false,
		sources: [LIGHTNING],
	},
];
