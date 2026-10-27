/**
 * Free and starter GPU tiers (D4). Each figure carries where it came from and when it was
 * read; "reported" marks numbers the platform itself doesn't publish. A Sep 2026 article listed
 * an L4 on Lightning's free credits; the plan page captured 22 Oct 2026 lists L40S, A100, H100
 * and H200 and ends the monthly credits, so the newer page wins. GPU names are B's
 * canonical names (engine/gpus.py), so capacities come from the same table as everything else.
 */

export interface Source {
	label: string;
	url: string;
	/** Date D read it, YYYY-MM-DD. */
	read: string;
}

export interface Preset {
	id: "colab" | "kaggle" | "modal" | "lightning";
	/** GPUs the tier offers, one per session for this check. */
	gpus: readonly string[];
	/** Longest single session before the runtime stops or restarts. */
	sessionCapS: number;
	/** The GPU list is what users report, not what the platform publishes. */
	gpusReported: boolean;
	/** Every GPU offered predates bf16 (T4, P100): train in fp16 with loss scaling. */
	fp16Only: boolean;
	/** Free compute credit per month, US dollars, when the tier is a credit rather than a session. */
	monthlyCreditUsd?: number;
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

const MODAL_PRICING: Source = {
	label: "Modal pricing",
	url: "https://modal.com/pricing",
	read: "2026-10-27",
};

const MODAL_TIMEOUTS: Source = {
	label: "Modal timeouts",
	url: "https://modal.com/docs/guide/timeouts",
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
		sessionCapS: 12 * H, // [cited: Colab FAQ]
		gpusReported: true,
		fp16Only: true,
		sources: [COLAB_FAQ, GPUPERHOUR],
	},
	{
		// Kaggle's docs: 12 hours of execution per GPU session. T4 and P100 class GPUs; the weekly
		// quota shows in account settings (users report about 30 hours).
		id: "kaggle",
		gpus: ["T4", "P100-16GB"],
		sessionCapS: 12 * H, // [cited: Kaggle docs, via GPU Per Hour]
		gpusReported: false,
		fp16Only: true,
		sources: [GPUPERHOUR],
	},
	{
		// Starter plan: $0 base with $30 of free compute credit a month (L4 at $0.000222 a second,
		// about 37 GPU-hours [derived: 30 / (0.000222 x 3,600)]); a Function runs at most 24 hours.
		id: "modal",
		gpus: ["L4", "T4"],
		sessionCapS: 24 * H, // [cited: Modal timeouts, "between 1 second and 24 hours"]
		gpusReported: false,
		fp16Only: false,
		monthlyCreditUsd: 30, // [cited: Modal pricing, Starter "$30 / month free credits"]
		sources: [MODAL_PRICING, MODAL_TIMEOUTS],
	},
	{
		// Free plan: one Studio that restarts every 4 hours; GPUs run on a one-time starter grant of
		// up to 30 credits (the monthly credits ended in September 2026).
		id: "lightning",
		gpus: ["L40S", "H100-SXM-80GB", "H200-SXM-141GB"],
		sessionCapS: 4 * H, // [cited: Lightning plan page, "4-hr restarts"]
		gpusReported: false,
		fp16Only: false,
		sources: [LIGHTNING],
	},
];
