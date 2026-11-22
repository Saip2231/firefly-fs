// Role colours are CSS variables (defined per theme in index.css), so every
// card, stroke and glow switches with the theme without a re-render.
export const COLORS = {
  flow: "var(--flow)",
  flowAlt: "var(--flow-alt)",
  search: "var(--search)",
  lock: "var(--lock)",
  accent: "var(--accent)",
} as const;

/** Text-safe variants (WCAG AA on the card surface) of the role colours. */
export const INK = {
  flow: "var(--flow)",
  flowAlt: "var(--flow-alt)",
  search: "var(--search-text)",
  lock: "var(--lock)",
  accent: "var(--accent-text)",
} as const;

/** `color` at `pct`% opacity; works with CSS variables. */
export const tint = (color: string, pct: number): string =>
  `color-mix(in srgb, ${color} ${pct}%, transparent)`;

export const pct = (x: number | null | undefined, digits = 1): string =>
  x === null || x === undefined || Number.isNaN(x) ? "—" : `${(x * 100).toFixed(digits)}%`;

export const pts = (x: number | null | undefined): string =>
  x === null || x === undefined || Number.isNaN(x) ? "—" : `±${(x * 100).toFixed(1)} pts`;

export function seconds(s: number | null | undefined): string {
  if (s === null || s === undefined || Number.isNaN(s)) return "—";
  if (s < 1) return `${(s * 1000).toFixed(s < 0.01 ? 1 : 0)} ms`;
  if (s < 100) return `${s.toFixed(1)} s`;
  return `${Math.round(s)} s`;
}

export const int = (x: number | null | undefined): string =>
  x === null || x === undefined || Number.isNaN(x) ? "—" : Math.round(x).toLocaleString("en-US");
