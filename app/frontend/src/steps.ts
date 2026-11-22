// The protocol, in order, as animation steps.  Each step lights one node after
// packets have travelled the listed edge chains (chains run in parallel; the
// edges inside one chain run one after another).

import type { EdgeId, NodeId } from "./layout";
import type { Track } from "./types";

export interface StepContext {
  datasetShort: string;
  methodName: string;
  isPca: boolean;
  k: number;
  budget: number | null;
  accText: string;
}

export interface Step {
  node: NodeId;
  chains: EdgeId[][];
  status: (c: StepContext) => string;
}

const head: Step[] = [
  { node: "dataset", chains: [], status: (c) => `Loading ${c.datasetShort}…` },
  { node: "split", chains: [["dataset-split"]],
    status: () => "Stratified split for this seed (shared by every method)…" },
  { node: "train", chains: [["split-train"]],
    status: () => "Fitting scaler on training rows only…" },
];

const tail: Step[] = [
  { node: "mask", chains: [], // filled per track below
    status: (c) => (c.isPca ? `Projecting to ${c.k} PCA components…`
                             : `Keeping exactly k = ${c.k} features…`) },
  { node: "final", chains: [["mask-final"]],
    status: () => "Fitting logistic regression on the full training split…" },
  { node: "test", chains: [["split-test"]], status: () => "Unlocking test split…" },
  { node: "score", chains: [["final-score"], ["test-score"]],
    status: (c) => `Scored once on the test split: ${c.accText} test accuracy` },
];

const withMaskChain = (chain: EdgeId[]): Step[] =>
  tail.map((s) => (s.node === "mask" ? { ...s, chains: [chain] } : s));

export function buildSteps(track: Track): Step[] {
  if (track === "baseline") {
    return [
      ...head,
      { node: "baseline", chains: [["train-baseline"]],
        status: (c) => (c.isPca ? "Fitting PCA projection on training rows…"
                                : `Scoring features with ${c.methodName} on training rows…`) },
      ...withMaskChain(["baseline-mask"]),
    ];
  }
  return [
    ...head,
    { node: "inner", chains: [["train-inner"]],
      status: () => "Carving inner-train / inner-validation from training rows…" },
    { node: "search", chains: [["inner-search"]],
      status: (c) => `Evaluating inner-val fitness… ${c.budget ?? ""} new fits`.trim() },
    ...withMaskChain(["search-mask"]),
  ];
}

/** Step index at which each edge is traversed (-1 = never on this track). */
export function edgeStepIndex(steps: Step[]): Partial<Record<EdgeId, number>> {
  const out: Partial<Record<EdgeId, number>> = {};
  steps.forEach((s, i) => s.chains.flat().forEach((e) => { out[e] = i; }));
  return out;
}
