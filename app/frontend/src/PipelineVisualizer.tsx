import { motion, useReducedMotion, type Variants } from "framer-motion";
import {
  Database, GitBranch, ListChecks, Lock, LockOpen, Rows3, Sparkles, SlidersHorizontal,
  SquareSplitHorizontal, Target, TrendingUp,
} from "lucide-react";
import { useMemo, useRef, useState } from "react";
import { Connectors, type EdgeState, type Packet } from "./Connectors";
import { ControlBar } from "./ControlBar";
import { COLORS, INK, int, pct, tint } from "./format";
import { useCountUp, usePostFrameHeight, useThemeMode, useWidth } from "./hooks";
import { LOCKED_EDGES, TALL, TRACK_EDGES, WIDE, type EdgeId, type NodeId } from "./layout";
import { Chips, NodeCard, type NodeState } from "./NodeCard";
import { ResultsCard } from "./ResultsCard";
import { buildSteps, edgeStepIndex, type StepContext } from "./steps";
import {
  BASELINE_METHODS, META_METHODS, resultKey, type BaselineMethod, type DatasetKey,
  type MetaMethod, type MethodKey, type PipelinePayload, type Track,
} from "./types";
import { SEARCH_DWELL_S, usePlayback } from "./usePlayback";

const NARROW_PX = 720;
const SHORT: Record<MethodKey, string> = {
  anova: "ANOVA", mi: "MI", l1: "L1", rf: "RF", pca: "PCA", firefly: "Firefly", pso: "PSO",
};
const NODE_COLOR: Record<NodeId, string> = {
  dataset: COLORS.flow, split: COLORS.flow, train: COLORS.flow, test: COLORS.lock,
  baseline: COLORS.flowAlt, inner: COLORS.flow, search: COLORS.search, mask: COLORS.flow,
  final: COLORS.flowAlt, score: COLORS.lock,
};
const GROUP_EMOJI: Record<string, string> = { locked: "🔒", evaluation: "🎯", selection: "" };
const ALL_DATASETS: DatasetKey[] = ["digits", "cifar_r18"];

const nearest = (xs: number[], x: number) =>
  xs.reduce((best, v) => (Math.abs(v - x) < Math.abs(best - x) ? v : best), xs[0]);

function edgeColor(e: EdgeId): string {
  if (LOCKED_EDGES.includes(e)) return COLORS.lock;
  if (e === "search-mask") return COLORS.search;
  if (TRACK_EDGES.baseline.includes(e)) return COLORS.flowAlt;
  return COLORS.flow;
}

const section: Variants = {
  hidden: { opacity: 0, y: 8 },
  show: { opacity: 1, y: 0, transition: { duration: 0.35, ease: "easeOut" } },
};

export default function PipelineVisualizer({ payload }: { payload: PipelinePayload | null }) {
  useThemeMode(payload?.theme ?? null);
  // ----- inputs ------------------------------------------------------------
  const systemReduce = useReducedMotion() ?? false;
  const [manualReduce, setManualReduce] = useState(false);
  const still = systemReduce || manualReduce;

  const available = ALL_DATASETS.filter((d) => payload?.datasets[d]);
  const datasetKeys = available.length ? available : ALL_DATASETS;
  const [dataset, setDataset] = useState<DatasetKey>(datasetKeys[0]);
  const block = payload?.datasets[dataset] ?? null;

  const [track, setTrack] = useState<Track>("meta");
  const [baseMethod, setBaseMethod] = useState<BaselineMethod>("anova");
  const [metaMethod, setMetaMethod] = useState<MetaMethod>("firefly");
  const method: MethodKey = track === "baseline" ? baseMethod : metaMethod;
  const isPca = method === "pca";
  const methodName = payload?.methods[method] ?? SHORT[method];

  const kValues = block?.k_values.length ? block.k_values : [16];
  const [kWanted, setKWanted] = useState(16);
  const k = nearest(kValues, kWanted);
  const budgets = block?.budgets.length ? block.budgets : [800];
  const [budgetWanted, setBudget] = useState(800);
  const budget = budgets.includes(budgetWanted) ? budgetWanted : budgets[0];
  const cell = block?.results[resultKey(method, k, track === "meta" ? budget : null)] ?? null;

  // ----- layout ------------------------------------------------------------
  const wrapRef = useRef<HTMLDivElement>(null);
  const width = useWidth(wrapRef);
  const layout = width > 0 && width < NARROW_PX ? TALL : WIDE;
  const scale = width > 0 ? Math.min(1, width / layout.width) : 1;
  const offsetX = width > layout.width ? (width - layout.width) / 2 : 0;

  // ----- playback ----------------------------------------------------------
  const steps = useMemo(() => buildSteps(track), [track]);
  const edgeAt = useMemo(() => edgeStepIndex(steps), [steps]);
  const pb = usePlayback(steps, still, track);
  const { step, arrived } = pb;

  const edgeState = (e: EdgeId): EdgeState => {
    const at = edgeAt[e];
    if (at === undefined) return "inactive";
    if (step < 0 || at > step) return "pending";
    if (at < step || arrived) return "lit";
    return "current";
  };
  const nodeState = (id: NodeId): NodeState => {
    const at = steps.findIndex((s) => s.node === id);
    if (at === -1) return "dim";
    if (id === "test" && (step < at || (step === at && !arrived))) return "locked";
    if (step === at) return arrived ? "active" : "idle";
    return step > at ? "done" : "idle";
  };
  const packets: Packet[] = step >= 0 && !arrived && !still
    ? steps[step].chains.map((chain, i) => ({
        chain: i, start: layout.edges[chain[0]][0], color: edgeColor(chain[0]) }))
    : [];

  const ctx: StepContext = {
    datasetShort: block?.short ?? dataset, methodName, isPca, k,
    budget: track === "meta" ? budget : null, accText: pct(cell?.acc_mean),
  };
  const statusOf = (id: NodeId) => steps.find((s) => s.node === id)?.status(ctx);
  const globalStatus = step < 0
    ? "Press Play to run the protocol, or Step through it one stage at a time."
    : steps[step].status(ctx);

  const searchState = nodeState("search");
  const fits = useCountUp(searchState === "active" || searchState === "done" ? budget : 0,
                          still, SEARCH_DWELL_S);
  const innerErr = useCountUp(track === "meta" ? cell?.inner_val_error_mean ?? null : null, still);
  const testState = nodeState("test");

  usePostFrameHeight([layout.kind, scale]);

  const N = layout.nodes;
  const node = (id: NodeId) => ({
    rect: N[id], color: NODE_COLOR[id], state: nodeState(id), status: statusOf(id), still,
  });
  return (
    <motion.div
      initial={still ? false : "hidden"} animate="show"
      variants={{ show: { transition: { staggerChildren: 0.06 } } }}
      className="relative overflow-hidden rounded-2xl border border-line p-4 font-sans text-ink
                 shadow-card sm:p-[26px]"
      style={{ background: "radial-gradient(130% 120% at 12% 0%, var(--glow) 0%, transparent 60%), var(--bg)" }}>
      <div aria-hidden className="pointer-events-none absolute inset-0"
           style={{ backgroundImage: "linear-gradient(var(--grid) 1px, transparent 1px), linear-gradient(90deg, var(--grid) 1px, transparent 1px)",
                    backgroundSize: "32px 32px" }} />
      <div aria-hidden className="pointer-events-none absolute -bottom-32 right-0 h-80 w-80 rounded-full blur-3xl"
           style={{ background: tint(COLORS.accent, 10) }} />

      <div className="relative space-y-[26px]">
        <motion.header variants={section} className="flex flex-col gap-2.5">
          <span className="font-mono text-[12px] font-medium uppercase tracking-[0.12em] text-muted">
            ⚡ Pipeline
          </span>
          <h2 className="font-heading text-[clamp(1.5rem,2.2vw,2.05rem)] font-bold leading-[1.25]
                         tracking-[-0.015em]">
            The protocol, stage by stage
          </h2>
          <p className="max-w-[65ch] text-[15.5px] leading-[1.65] text-muted">
            Pick a dataset, a method and k, then play the protocol stage by stage.
          </p>
        </motion.header>

        {!payload && (
          <div className="rounded-xl border px-3 py-2 text-[13px]"
               style={{ borderColor: tint(COLORS.lock, 45), background: tint(COLORS.lock, 10) }}>
            No benchmark data was injected, so result values show as “—”. Open this from the
            Streamlit Protocol page (or provide dev-data.json in dev mode).
          </div>
        )}
        <motion.div variants={section}>
          <ControlBar
            dataset={dataset}
            datasets={datasetKeys.map((d) => ({
              value: d, label: payload?.datasets[d]?.short ?? (d === "digits" ? "Digits" : "CIFAR-10 (ResNet-18)") }))}
            onDataset={setDataset}
            track={track} onTrack={setTrack}
            kValues={kValues} k={k} onK={setKWanted} isPca={isPca}
            budgets={budgets} budget={budget} onBudget={setBudget}
            playing={pb.playing} onPlay={pb.onPlay} onStep={pb.onStep} onReplay={pb.onReplay}
            reduceMotion={still} systemReduce={systemReduce} onReduceMotion={setManualReduce}
          />
        </motion.div>

        <motion.div variants={section} aria-live="polite"
                    className="flex flex-wrap items-center gap-x-3 gap-y-1 text-[13px]">
          <span className="rounded-full border border-line bg-surface px-2.5 py-0.5 font-mono
                           text-[12px] tabular-nums text-muted">
            {step < 0 ? "Ready" : `Stage ${step + 1} / ${steps.length}`}
          </span>
          <span>{globalStatus}</span>
          <div className="ml-auto flex gap-1" aria-hidden>
            {steps.map((s, i) => (
              <span key={s.node} className="h-1.5 w-5 rounded-full transition-colors duration-300"
                    style={{ background: i < step || (i === step && arrived) ? NODE_COLOR[s.node]
                      : i === step ? tint(NODE_COLOR[s.node], 40) : "var(--edge-off)" }} />
            ))}
          </div>
        </motion.div>

        <motion.div variants={section} ref={wrapRef} className="w-full">
          <div className="relative" style={{ height: layout.height * scale }}>
            <div className="absolute top-0 origin-top-left"
                 role="img"
                 aria-label="Protocol diagram: dataset, stratified split, training split with feature selection on training rows only, locked test split used once for the final score."
                 style={{ left: offsetX, width: layout.width, height: layout.height,
                          transform: `scale(${scale})` }}>
              {layout.groups.map((g) => {
                const c = g.id === "selection" ? "var(--edge-idle)" : tint(COLORS.lock, 45);
                return (
                  <div key={g.id} className="absolute rounded-2xl border border-dashed"
                       style={{ left: g.rect.x, top: g.rect.y, width: g.rect.w, height: g.rect.h,
                                borderColor: c }}>
                    <span className="absolute -top-2.5 left-3 whitespace-nowrap rounded-full bg-page
                                     px-2 font-mono text-[11px] font-medium uppercase
                                     tracking-[0.12em]"
                          style={{ color: g.id === "selection" ? "var(--muted)" : COLORS.lock }}>
                      {GROUP_EMOJI[g.id] ? `${GROUP_EMOJI[g.id]} ` : ""}{g.label}
                    </span>
                  </div>
                );
              })}

              <Connectors layout={layout} edgeState={edgeState} edgeColor={edgeColor} still={still}
                          packets={packets} pathEls={pb.pathEls} trailEls={pb.trailEls}
                          packetEls={pb.packetEls} />

              <NodeCard {...node("dataset")} icon={Database} title="Dataset"
                        subtitle="Digits or CIFAR-10 subset (ResNet-18 embeddings)">
                <div className="mt-auto font-mono text-[11.5px]" style={{ color: INK.flow }}>
                  {block?.short ?? "—"} · d = {block?.d ?? "—"}
                </div>
              </NodeCard>
              <NodeCard {...node("split")} icon={GitBranch} title="Stratified train / test split"
                        subtitle="One per seed, same for every method" />
              <NodeCard {...node("train")} icon={Rows3} title="Training split"
                        subtitle="Scaler fitted here only" />
              <NodeCard {...node("test")} icon={testState === "locked" ? Lock : LockOpen}
                        title="Test split (locked)" dashed
                        subtitle={testState === "locked" ? "Not used until the end"
                                                         : "Opened once, for the final score"} />
              <NodeCard {...node("baseline")} icon={SlidersHorizontal} title="Baseline selectors"
                        subtitle="ANOVA, MI, L1, RF; PCA projection (fit on training rows)"
                        interactive>
                <Chips options={BASELINE_METHODS} value={track === "baseline" ? baseMethod : null}
                       label={(m) => SHORT[m]} color={COLORS.flowAlt} ink={INK.flowAlt}
                       onPick={(m) => { setBaseMethod(m); setTrack("baseline"); }} />
              </NodeCard>
              <NodeCard {...node("inner")} icon={SquareSplitHorizontal}
                        title="Inner split of training rows" subtitle="Inner-train / inner-validation" />
              <NodeCard {...node("search")} icon={Sparkles} title="Firefly / PSO search"
                        subtitle="Fitness = inner-validation error (budget of new fits)"
                        ink={INK.search} interactive>
                <Chips options={META_METHODS} value={track === "meta" ? metaMethod : null}
                       label={(m) => SHORT[m]} color={COLORS.search} ink={INK.search}
                       onPick={(m) => { setMetaMethod(m); setTrack("meta"); }} />
                {track === "meta" && (
                  <div className="mt-auto space-y-1">
                    <div className="h-1 overflow-hidden rounded-full"
                         style={{ background: "var(--edge-off)" }}>
                      <div className="h-full rounded-full"
                           style={{ width: `${((fits ?? 0) / budget) * 100}%`,
                                    background: COLORS.search }} />
                    </div>
                    <div className="font-mono text-[11px] tabular-nums text-muted">
                      {int(fits)} / {int(budget)} fits
                    </div>
                    <div className="rounded-full border px-2 py-0.5 font-mono text-[11px] tabular-nums"
                         title={payload?.inner_val_note}
                         style={{ borderColor: tint(COLORS.search, 45),
                                  background: tint(COLORS.search, 10), color: INK.search }}>
                      inner-val error {pct(innerErr)}
                    </div>
                  </div>
                )}
              </NodeCard>
              <NodeCard {...node("mask")} icon={ListChecks}
                        title={isPca ? `${k} PCA components` : `Feature subset (exactly k = ${k})`}
                        subtitle={isPca ? "Projection, not a selector"
                                        : `${k} of ${block?.d ?? "—"} input features`} />
              <NodeCard {...node("final")} icon={TrendingUp} title="Final logistic regression"
                        subtitle="Trained on the full training split" />
              <NodeCard {...node("score")} icon={Target} title="Scored ONCE on the test split"
                        subtitle="One final evaluation per seed" />

              <ResultsCard rect={layout.results} wide={layout.kind === "wide"} cell={cell}
                           block={block} methodName={methodName} isPca={isPca} k={k}
                           highlight={nodeState("score") === "active"} still={still}
                           searchNote={payload?.search_s_note ?? ""} />
            </div>
          </div>
        </motion.div>

        <motion.p variants={section}
                  className="max-w-[65ch] text-[13px] leading-[1.65] text-muted">
          The animation shows the order of the protocol; it is not a live run. The result card
          shows the benchmark mean ± std over seeds for the selected dataset, method, k
          {track === "meta" ? " and budget" : ""}.
          {payload ? ` Selection time: ${payload.search_s_note}.` : ""}
          {payload && track === "meta" ? ` ${payload.inner_val_note}` : ""}
          {payload?.sources.length ? ` Sources: ${payload.sources.join(", ")}.` : ""}
        </motion.p>
      </div>
    </motion.div>
  );
}
