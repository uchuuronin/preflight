/**
 * Wraps every add-on component C renders. An add-on that throws while rendering disappears
 * (and is logged once) instead of taking the page or the cockpit with it.
 */
import { Component, type ReactNode } from "react";

interface Props {
	id: string;
	children: ReactNode;
}

export class AddonBoundary extends Component<Props, { failed: boolean }> {
	override state = { failed: false };

	static getDerivedStateFromError(): { failed: boolean } {
		return { failed: true };
	}

	override componentDidCatch(error: unknown): void {
		console.error(`[preflight] add-on "${this.props.id}" failed to render and was hidden`, error);
	}

	override render(): ReactNode {
		return this.state.failed ? null : this.props.children;
	}
}
