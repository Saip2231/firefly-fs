import { motion } from "framer-motion";
import { Pause, Play, RotateCcw, StepForward } from "lucide-react";
import type { ReactNode } from "react";
import { COLORS, tint } from "./format";
import type { DatasetKey, Track } from "./types";

interface SegmentProps<T extends string> {
  id: string;
  label: string;
  value: T;
  options: { value: T; label: string }[];
  onChange: (v: T) => void;
  color?: string;
  still: boolean;
}

/** Segmented control with a pill indicator that slides to the chosen option. */
export function Segmented<T extends string>({ id, label, value, options, onChange,
                                              color = COLORS.accent, still }: SegmentProps<T>) {
  return (
    <div role="radiogroup" aria-label={label} className="flex flex-col gap-1">
      <span className="font-mono text-[12px] font-medium uppercase tracking-[0.12em] text-muted">{label}</span>
      <div className="flex flex-wrap gap-0.5 rounded-full border border-line bg-page p-1">
        {options.map((o) => {
          const on = o.value === value;
          return (
            <button key={o.value} type="button" role="radio" aria-checked={on}
                    onClick={() => onChange(o.value)}
                    className={`relative rounded-full px-3 py-1 text-[13px] font-medium
                                transition-[color,transform] duration-200 active:scale-[0.98]
                                focus:outline-none focus-visible:ring-2
                                focus-visible:ring-[color:var(--accent)]
                                ${on ? "text-ink" : "text-muted hover:text-ink"}`}>
              {on && (
                <motion.span
                  layoutId={`seg-${id}`}
                  transition={still ? { duration: 0 } : { type: "spring", stiffness: 500, damping: 38 }}
                  className="absolute inset-0 rounded-full"
                  style={{ background: tint(color, 16), boxShadow: `inset 0 0 0 1px ${color}` }}
                />
              )}
              <span className="relative">{o.label}</span>
            </button>
          );
        })}
      </div>
    </div>
  );
}

function IconButton({ label, onClick, children, primary }: {
  label: string; onClick: () => void; children: ReactNode; primary?: boolean;
}) {
  return (
    <button type="button" onClick={onClick} aria-label={label} title={label}
            className={`flex items-center gap-1.5 rounded-full border px-3.5 py-1.5 text-[13px]
                        font-semibold transition-[transform,background-color,border-color,color]
                        duration-200 active:scale-[0.98] focus:outline-none focus-visible:ring-2
                        focus-visible:ring-[color:var(--accent)]
                        ${primary
                          ? "border-transparent bg-[color:var(--accent)] text-[color:var(--on-accent)] hover:border-[color:var(--on-accent)]"
                          : "border-line text-ink hover:border-[color:var(--hover-border)]"}`}>
      {children}
    </button>
  );
}

interface Props {
  dataset: DatasetKey;
  datasets: { value: DatasetKey; label: string }[];
  onDataset: (d: DatasetKey) => void;
  track: Track;
  onTrack: (t: Track) => void;
  kValues: number[];
  k: number;
  onK: (k: number) => void;
  isPca: boolean;
  budgets: number[];
  budget: number;
  onBudget: (b: number) => void;
  playing: boolean;
  onPlay: () => void;
  onStep: () => void;
  onReplay: () => void;
  reduceMotion: boolean;
  systemReduce: boolean;
  onReduceMotion: (v: boolean) => void;
}

export function ControlBar(p: Props) {
  const kIdx = Math.max(0, p.kValues.indexOf(p.k));
  const still = p.reduceMotion;
  return (
    <div className="flex flex-wrap items-end gap-x-[26px] gap-y-4 rounded-2xl border border-line
                    bg-surface p-4 shadow-card sm:p-[26px]">
      <Segmented id="dataset" label="Dataset" value={p.dataset} options={p.datasets}
                 onChange={p.onDataset} still={still} color={COLORS.flow} />
      <Segmented<Track>
        id="track" label="Method" value={p.track} onChange={p.onTrack} still={still}
        color={p.track === "meta" ? COLORS.search : COLORS.flowAlt}
        options={[{ value: "baseline", label: "Baseline (ANOVA / MI / L1 / RF / PCA)" },
                  { value: "meta", label: "✨ Metaheuristic (Firefly / PSO)" }]} />
      {p.track === "meta" && p.budgets.length > 1 && (
        <Segmented<string>
          id="budget" label="Budget (new fits)" value={String(p.budget)} color={COLORS.search}
          onChange={(v) => p.onBudget(Number(v))} still={still}
          options={p.budgets.map((b) => ({ value: String(b), label: b.toLocaleString("en-US") }))} />
      )}
      <label className="flex min-w-[180px] flex-col gap-1">
        <span className="font-mono text-[12px] font-medium uppercase tracking-[0.12em] text-muted">
          {p.isPca ? "Components" : "Feature count"}{" "}
          <span className="font-mono text-[13px] normal-case tracking-normal text-ink">k = {p.k}</span>
        </span>
        <input type="range" min={0} max={Math.max(0, p.kValues.length - 1)} step={1} value={kIdx}
               onChange={(e) => p.onK(p.kValues[Number(e.target.value)])}
               aria-valuetext={`k = ${p.k}`} style={{ accentColor: "var(--accent)" }} />
        <div className="flex justify-between font-mono text-[11px] tabular-nums text-muted">
          {p.kValues.map((k) => <span key={k}>{k}</span>)}
        </div>
      </label>
      <div className="ml-auto flex flex-wrap items-center gap-2">
        <IconButton label={p.playing ? "Pause" : "Play"} onClick={p.onPlay} primary>
          {p.playing ? <Pause size={15} aria-hidden /> : <Play size={15} aria-hidden />}
          {p.playing ? "Pause" : "Play"}
        </IconButton>
        <IconButton label="Step" onClick={p.onStep}>
          <StepForward size={15} aria-hidden /> Step
        </IconButton>
        <IconButton label="Replay" onClick={p.onReplay}>
          <RotateCcw size={15} aria-hidden /> Replay
        </IconButton>
        <label className="flex cursor-pointer items-center gap-1.5 text-[12px] text-muted"
               title={p.systemReduce ? "Your system asks for reduced motion" : undefined}>
          <input type="checkbox" checked={p.reduceMotion} disabled={p.systemReduce}
                 onChange={(e) => p.onReduceMotion(e.target.checked)}
                 style={{ accentColor: "var(--accent)" }} />
          Reduce motion
        </label>
      </div>
    </div>
  );
}
