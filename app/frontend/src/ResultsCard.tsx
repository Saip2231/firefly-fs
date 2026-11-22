import { AnimatePresence, motion } from "framer-motion";
import { COLORS, int, pct, pts, seconds, tint } from "./format";
import { useCountUp } from "./hooks";
import type { Rect } from "./layout";
import type { DatasetBlock, ResultCell } from "./types";

interface Props {
  rect: Rect;
  wide: boolean;
  cell: ResultCell | null;
  block: DatasetBlock | null;
  methodName: string;
  isPca: boolean;
  k: number;
  highlight: boolean;
  still: boolean;
  searchNote: string;
}

function Row({ label, value, title }: { label: string; value: string; title?: string }) {
  return (
    <div className="flex items-baseline justify-between gap-2" title={title}>
      <span className="text-[11.5px] text-muted">{label}</span>
      <span className="font-mono text-[12.5px] font-semibold tabular-nums text-ink">{value}</span>
    </div>
  );
}

/** Benchmark result for the current selection (mean ± std over seeds). */
export function ResultsCard({ rect, wide, cell, block, methodName, isPca, k, highlight, still,
                              searchNote }: Props) {
  const acc = useCountUp(cell?.acc_mean ?? null, still);
  const std = useCountUp(cell?.acc_std ?? null, still);
  const time = useCountUp(cell?.search_s_mean ?? null, still);
  const fits = useCountUp(cell?.evals_used_mean ?? null, still);
  const kAnim = useCountUp(k, still, 0.4);
  const all = block?.all_features ?? null;

  return (
    <div
      aria-live="polite"
      className="absolute flex flex-col rounded-2xl border bg-surface p-3
                 transition-[box-shadow,border-color,background-color] duration-300"
      style={{
        left: rect.x, top: rect.y, width: rect.w, height: rect.h,
        borderColor: highlight ? COLORS.lock : tint(COLORS.lock, 35),
        boxShadow: highlight ? `0 0 34px ${tint(COLORS.lock, 35)}` : "var(--card-shadow)",
      }}
    >
      <div className="flex items-center gap-1.5 font-mono text-[11px] font-medium uppercase
                      tracking-[0.12em]"
           style={{ color: COLORS.lock }}>
        <span aria-hidden>🎯</span> Benchmark result
      </div>
      <AnimatePresence mode="wait" initial={false}>
        <motion.div key={methodName}
                    initial={still ? false : { opacity: 0, y: 4 }}
                    animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -4 }}
                    transition={{ duration: still ? 0 : 0.2 }}
                    className="mt-0.5 truncate text-[12px] text-muted" title={methodName}>
          {methodName}
        </motion.div>
      </AnimatePresence>
      {cell ? (
        <div className={wide ? "mt-1 flex flex-1 flex-col" : "mt-1 grid flex-1 grid-cols-2 gap-x-4"}>
          <div>
            <div className="font-mono text-[25px] font-bold leading-none tracking-tight
                            tabular-nums text-ink">
              {pct(acc)}
            </div>
            <div className="mt-1 font-mono text-[11.5px] tabular-nums text-muted">
              test accuracy {pts(std)}
            </div>
          </div>
          <div className={wide ? "mt-2 space-y-0.5" : "space-y-0.5"}>
            <Row label={isPca ? "Components" : "Selected k"}
                 value={`${int(kAnim)} of ${block?.d ?? "—"}`} />
            <Row label="Selection time" value={seconds(time)} title={searchNote} />
            {cell.evals_used_mean !== null && <Row label="New fits" value={int(fits)} />}
            <Row label="All features" value={pct(all?.acc_mean)} />
            <Row label="Seeds" value={`n = ${cell.n}`} />
          </div>
        </div>
      ) : (
        <div className="mt-2 text-[12px] text-muted">No benchmark row for this combination.</div>
      )}
    </div>
  );
}
