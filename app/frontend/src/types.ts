// Data contract between app/lib/pipeline_view.py (producer) and the
// visualizer (consumer).  Every number shown in the diagram comes from this
// payload; nothing is typed into the frontend (DASHBOARD.md rule 2).

export type DatasetKey = "digits" | "cifar_r18";
export type Track = "baseline" | "meta";
export type BaselineMethod = "anova" | "mi" | "l1" | "rf" | "pca";
export type MetaMethod = "firefly" | "pso";
export type MethodKey = BaselineMethod | MetaMethod;

export const BASELINE_METHODS: BaselineMethod[] = ["anova", "mi", "l1", "rf", "pca"];
export const META_METHODS: MetaMethod[] = ["firefly", "pso"];

/** One (method, k, budget) cell: mean and std over seeds. Fractions, not %. */
export interface ResultCell {
  n: number;
  acc_mean: number;
  acc_std: number;
  /** Selection time in seconds (see payload.search_s_note for how it was measured). */
  search_s_mean: number | null;
  /** New (uncached) classifier fits used by the search; null for baselines. */
  evals_used_mean: number | null;
  /** Mean inner-validation error of the returned mask; null for baselines. */
  inner_val_error_mean: number | null;
}

export interface DatasetBlock {
  label: string;
  short: string;
  /** Number of input features (64 pixels or 512 embedding dimensions). */
  d: number;
  k_values: number[];
  budgets: number[];
  all_features: { n: number; acc_mean: number; acc_std: number } | null;
  /** Keyed by resultKey(method, k, budget). */
  results: Record<string, ResultCell>;
}

export interface PipelinePayload {
  /** Theme the server rendered with; the live host theme takes precedence. */
  theme?: "light" | "dark";
  sources: string[];
  methods: Record<MethodKey, string>;
  search_s_note: string;
  inner_val_note: string;
  datasets: Partial<Record<DatasetKey, DatasetBlock>>;
}

export const resultKey = (method: MethodKey, k: number, budget: number | null): string =>
  `${method}|${k}|${budget ?? ""}`;
