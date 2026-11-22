import { AnimatePresence, motion } from "framer-motion";
import { Check, type LucideIcon } from "lucide-react";
import type { ReactNode } from "react";
import { tint } from "./format";
import type { Rect } from "./layout";

export type NodeState = "idle" | "active" | "done" | "dim" | "locked";

interface Props {
  rect: Rect;
  icon: LucideIcon;
  title: string;
  subtitle: string;
  /** Role colour for strokes, fills and glows (a CSS variable). */
  color: string;
  /** Text-safe variant of `color`; defaults to it. */
  ink?: string;
  state: NodeState;
  status?: string;
  still: boolean;
  dashed?: boolean;
  interactive?: boolean;
  children?: ReactNode;
}

export function NodeCard({ rect, icon: Icon, title, subtitle, color, ink, state, status, still,
                           dashed, interactive, children }: Props) {
  const active = state === "active";
  const done = state === "done";
  const textColor = ink ?? color;
  const borderColor = active ? color : done ? tint(color, 60) : dashed ? tint(color, 45)
    : "var(--border)";
  const shadow = active ? `0 0 0 1px ${color}, 0 0 30px ${tint(color, 40)}`
    : done ? `0 0 16px ${tint(color, 14)}` : "var(--card-shadow)";
  return (
    <div
      role="group"
      aria-label={`${title}. ${subtitle}${state === "locked" ? ". Locked" : ""}`}
      className={`group absolute flex flex-col rounded-2xl bg-surface p-3 backdrop-blur-md
                  ${still ? "" : "transition-[transform,box-shadow,border-color,opacity,filter,background-color] duration-300 ease-out"}
                  ${active && !still ? "scale-[1.03]" : ""}
                  border-[color:var(--card-border)]
                  ${interactive && !still ? "hover:-translate-y-0.5" : ""}
                  ${interactive && !active ? "hover:border-[color:var(--hover-border)]" : ""}`}
      style={{
        left: rect.x, top: rect.y, width: rect.w, height: rect.h,
        borderWidth: 1.5, borderStyle: dashed ? "dashed" : "solid",
        ["--card-border" as string]: borderColor,
        boxShadow: shadow,
        opacity: state === "dim" ? 0.35 : state === "locked" ? 0.55 : 1,
        filter: state === "dim" ? "grayscale(0.85)" : "none",
      }}
    >
      <div className="mb-1.5 flex items-center gap-2">
        <span className="grid h-8 w-8 shrink-0 place-items-center rounded-lg"
              style={{ background: tint(color, 14), color }}>
          <Icon size={18} strokeWidth={2.2} aria-hidden />
        </span>
        {done && (
          <span className="ml-auto grid h-5 w-5 place-items-center rounded-full"
                style={{ background: tint(color, 20), color: textColor }} aria-label="done">
            <Check size={13} strokeWidth={3} aria-hidden />
          </span>
        )}
      </div>
      <div className="font-heading text-[15px] font-semibold leading-[1.25] tracking-[-0.015em]
                      text-ink">{title}</div>
      <div className="mt-1 text-[13px] leading-[1.5] text-muted">{subtitle}</div>
      {children}
      <AnimatePresence>
        {active && status && (
          <motion.div
            key="status"
            initial={still ? false : { opacity: 0, y: 4 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            transition={{ duration: still ? 0 : 0.25 }}
            className="pointer-events-none absolute -bottom-3 left-2 right-2 flex items-center
                       gap-1.5 truncate rounded-full border bg-page px-2 py-0.5 text-[11px]"
            style={{ borderColor: tint(color, 55), color: textColor }}
          >
            <span className={`h-1.5 w-1.5 shrink-0 rounded-full ${still ? "" : "animate-pulse"}`}
                  style={{ background: color }} />
            <span className="truncate">{status}</span>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

interface ChipsProps<T extends string> {
  options: T[];
  value: T | null;
  label: (v: T) => string;
  onPick: (v: T) => void;
  color: string;
  ink?: string;
}

/** Small clickable method chips (pill badges) shown inside a node card. */
export function Chips<T extends string>({ options, value, label, onPick, color,
                                          ink }: ChipsProps<T>) {
  return (
    <div className="mt-2 flex flex-wrap gap-1">
      {options.map((o) => {
        const on = o === value;
        return (
          <button
            key={o}
            type="button"
            aria-pressed={on}
            onClick={() => onPick(o)}
            className="rounded-full border px-2 py-0.5 font-mono text-[11px] font-medium
                       transition-[transform,background-color,border-color,color] duration-200
                       border-[color:var(--chip-border)]
                       hover:border-[color:var(--hover-border)] focus:outline-none
                       focus-visible:ring-2 focus-visible:ring-[color:var(--accent)]
                       active:scale-[0.98]"
            style={{
              ["--chip-border" as string]: on ? color : "var(--border)",
              background: on ? tint(color, 16) : "transparent",
              color: on ? (ink ?? "var(--text)") : "var(--muted)",
            }}
          >
            {label(o)}
          </button>
        );
      })}
    </div>
  );
}
