// Fixed-coordinate layouts.  Cards (HTML) and connectors (SVG) share one
// logical coordinate space that is scaled as a whole, so paths stay aligned
// with the cards at every container width.  "wide" flows left to right,
// "tall" (narrow screens) flows top to bottom.

export type NodeId =
  | "dataset" | "split" | "train" | "test" | "baseline"
  | "inner" | "search" | "mask" | "final" | "score";

export type EdgeId =
  | "dataset-split" | "split-train" | "split-test" | "train-baseline" | "train-inner"
  | "baseline-mask" | "inner-search" | "search-mask" | "mask-final" | "final-score"
  | "test-score";

export interface Rect { x: number; y: number; w: number; h: number }
export type Pt = [number, number];

export interface Group {
  id: "selection" | "locked" | "evaluation";
  rect: Rect;
  label: string;
}

export interface EdgeLabel {
  lines: string[];
  x: number;
  y: number;
  anchor: "start" | "middle";
  tone: "lock" | "flow";
}

export interface Layout {
  kind: "wide" | "tall";
  width: number;
  height: number;
  nodes: Record<NodeId, Rect>;
  results: Rect;
  edges: Record<EdgeId, Pt[]>;
  groups: Group[];
  labels: EdgeLabel[];
}

/** Edges that belong to only one feature-selection track. */
export const TRACK_EDGES: Record<"baseline" | "meta", EdgeId[]> = {
  baseline: ["train-baseline", "baseline-mask"],
  meta: ["train-inner", "inner-search", "search-mask"],
};
export const LOCKED_EDGES: EdgeId[] = ["split-test", "test-score"];

const GROUP_LABELS = {
  selection: "Feature selection · training rows only",
  locked: "Locked test split",
  evaluation: "Final evaluation",
};

export const WIDE: Layout = {
  kind: "wide",
  width: 1550,
  height: 700,
  nodes: {
    dataset: { x: 16, y: 322, w: 140, h: 136 },
    split: { x: 206, y: 322, w: 150, h: 136 },
    train: { x: 420, y: 322, w: 150, h: 136 },
    test: { x: 420, y: 36, w: 150, h: 116 },
    baseline: { x: 610, y: 195, w: 320, h: 150 },
    inner: { x: 610, y: 500, w: 150, h: 130 },
    search: { x: 785, y: 465, w: 170, h: 200 },
    mask: { x: 990, y: 322, w: 150, h: 136 },
    final: { x: 1180, y: 322, w: 150, h: 136 },
    score: { x: 1370, y: 322, w: 160, h: 136 },
  },
  results: { x: 1370, y: 478, w: 160, h: 202 },
  edges: {
    "dataset-split": [[156, 390], [206, 390]],
    "split-train": [[356, 390], [420, 390]],
    "split-test": [[281, 322], [281, 94], [420, 94]],
    "train-baseline": [[570, 390], [590, 390], [590, 270], [610, 270]],
    "train-inner": [[570, 390], [590, 390], [590, 565], [610, 565]],
    "baseline-mask": [[930, 270], [972, 270], [972, 390], [990, 390]],
    "inner-search": [[760, 565], [785, 565]],
    "search-mask": [[955, 565], [972, 565], [972, 390], [990, 390]],
    "mask-final": [[1140, 390], [1180, 390]],
    "final-score": [[1330, 390], [1370, 390]],
    "test-score": [[570, 94], [1450, 94], [1450, 322]],
  },
  groups: [
    { id: "locked", rect: { x: 404, y: 16, w: 182, h: 152 }, label: GROUP_LABELS.locked },
    { id: "selection", rect: { x: 398, y: 172, w: 758, h: 516 }, label: GROUP_LABELS.selection },
    { id: "evaluation", rect: { x: 1356, y: 172, w: 188, h: 516 }, label: GROUP_LABELS.evaluation },
  ],
  labels: [
    { lines: ["not used", "until the end"], x: 292, y: 196, anchor: "start", tone: "lock" },
    { lines: ["opened once, at the end"], x: 1010, y: 82, anchor: "middle", tone: "lock" },
  ],
};

export const TALL: Layout = {
  kind: "tall",
  width: 450,
  height: 1612,
  nodes: {
    dataset: { x: 16, y: 16, w: 190, h: 130 },
    split: { x: 16, y: 186, w: 190, h: 130 },
    test: { x: 234, y: 186, w: 190, h: 130 },
    train: { x: 16, y: 366, w: 190, h: 120 },
    baseline: { x: 16, y: 536, w: 190, h: 180 },
    inner: { x: 234, y: 536, w: 190, h: 130 },
    search: { x: 234, y: 706, w: 190, h: 220 },
    mask: { x: 125, y: 976, w: 190, h: 120 },
    final: { x: 125, y: 1136, w: 190, h: 120 },
    score: { x: 125, y: 1296, w: 190, h: 120 },
  },
  results: { x: 16, y: 1436, w: 408, h: 160 },
  edges: {
    "dataset-split": [[111, 146], [111, 186]],
    "split-train": [[111, 316], [111, 366]],
    "split-test": [[206, 251], [234, 251]],
    "train-baseline": [[111, 486], [111, 536]],
    "train-inner": [[111, 486], [111, 511], [329, 511], [329, 536]],
    "baseline-mask": [[111, 716], [111, 951], [220, 951], [220, 976]],
    "inner-search": [[329, 666], [329, 706]],
    "search-mask": [[329, 926], [329, 951], [220, 951], [220, 976]],
    "mask-final": [[220, 1096], [220, 1136]],
    "final-score": [[220, 1256], [220, 1296]],
    "test-score": [[424, 251], [440, 251], [440, 1356], [315, 1356]],
  },
  groups: [
    { id: "locked", rect: { x: 226, y: 174, w: 206, h: 154 }, label: GROUP_LABELS.locked },
    { id: "selection", rect: { x: 6, y: 340, w: 426, h: 768 }, label: GROUP_LABELS.selection },
    { id: "evaluation", rect: { x: 6, y: 1284, w: 428, h: 320 }, label: GROUP_LABELS.evaluation },
  ],
  labels: [
    { lines: ["test rows: not used", "until the end"], x: 329, y: 372, anchor: "middle", tone: "lock" },
  ],
};

/** Orthogonal polyline with rounded corners as an SVG path string. */
export function roundedPath(points: Pt[], radius = 14): string {
  if (points.length < 2) return "";
  const [x0, y0] = points[0];
  let d = `M ${x0} ${y0}`;
  for (let i = 1; i < points.length - 1; i++) {
    const [px, py] = points[i - 1];
    const [cx, cy] = points[i];
    const [nx, ny] = points[i + 1];
    const inLen = Math.hypot(cx - px, cy - py);
    const outLen = Math.hypot(nx - cx, ny - cy);
    const r = Math.min(radius, inLen / 2, outLen / 2);
    const ax = cx - ((cx - px) / inLen) * r;
    const ay = cy - ((cy - py) / inLen) * r;
    const bx = cx + ((nx - cx) / outLen) * r;
    const by = cy + ((ny - cy) / outLen) * r;
    d += ` L ${ax} ${ay} Q ${cx} ${cy} ${bx} ${by}`;
  }
  const [xl, yl] = points[points.length - 1];
  return `${d} L ${xl} ${yl}`;
}
