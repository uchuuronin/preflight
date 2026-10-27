export function Plane({ size = 22 }: { size?: number }) {
	return (
		<svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
			<path d="M2.5 13.5l7-1.5L14 4.5c.6-1 2-1.2 2.6-.3.3.5.2 1.1-.1 1.6L13 12l6.8 1.6c1 .2 1.4 1.4.7 2.1L19 17l-6.5-1.5-3.2 4.3h-2l1.2-4.7-5-1.2z" />
		</svg>
	);
}
