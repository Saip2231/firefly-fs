import type { MutableRefObject } from "react";
import { COLORS, tint } from "./format";
import { LOCKED_EDGES, roundedPath, type EdgeId, type Layout } from "./layout";

export type EdgeState = "inactive" | "pending" | "current" | "lit";

export interface Packet {
  chain: number;
  start: [number, number];
  color: string;
}

interface Props {
  layout: Layout;
  edgeState: (e: EdgeId) => EdgeState;
  edgeColor: (e: EdgeId) => string;
  still: boolean;
  packets: Packet[];
  pathEls: MutableRefObject<Partial<Record<EdgeId, SVGPathElement | null>>>;
  trailEls: MutableRefObject<Partial<Record<EdgeId, SVGPathElement | null>>>;
  packetEls: MutableRefObject<(SVGGElement | null)[]>;
}

// Strokes and fills are set through `style` (not presentation attributes) so
// CSS variables resolve and theme switches transition smoothly.
const ARROWS: Record<string, string> = {
  idle: "var(--edge-idle)",
  flow: COLORS.flow,
  flowAlt: COLORS.flowAlt,
  search: COLORS.search,
  lock: COLORS.lock,
};
const arrowFor = (color: string) =>
  `url(#arrow-${Object.keys(ARROWS).find((k) => ARROWS[k] === color) ?? "idle"})`;

export function Connectors({ layout, edgeState, edgeColor, still, packets, pathEls, trailEls,
                             packetEls }: Props) {
  const ids = Object.keys(layout.edges) as EdgeId[];
  return (
    <svg
      className="pointer-events-none absolute inset-0"
      width={layout.width}
      height={layout.height}
      viewBox={`0 0 ${layout.width} ${layout.height}`}
      aria-hidden
    >
      <defs>
        {Object.entries(ARROWS).map(([name, c]) => (
          <marker key={name} id={`arrow-${name}`} viewBox="0 0 10 10" refX="9" refY="5"
                  markerWidth="9" markerHeight="9" markerUnits="userSpaceOnUse"
                  orient="auto-start-reverse">
            <path d="M 0 0.5 L 9.5 5 L 0 9.5 z" style={{ fill: c }} />
          </marker>
        ))}
        <filter id="packet-glow" x="-200%" y="-200%" width="500%" height="500%">
          <feGaussianBlur stdDeviation="4" />
        </filter>
      </defs>

      {ids.map((id) => {
        const d = roundedPath(layout.edges[id]);
        const state = edgeState(id);
        const color = edgeColor(id);
        const locked = LOCKED_EDGES.includes(id);
        const dash = locked ? "7 7" : undefined;
        const base = locked ? tint(COLORS.lock, state === "inactive" ? 15 : 30)
          : state === "inactive" ? "var(--edge-off)" : "var(--edge-idle)";
        return (
          <g key={id}>
            <path
              ref={(el) => { pathEls.current[id] = el; }}
              d={d}
              fill="none"
              style={{ stroke: base }}
              strokeWidth={1.75}
              strokeDasharray={dash}
              markerEnd={state === "lit" ? undefined : arrowFor("var(--edge-idle)")}
            />
            {state === "current" && (
              <path
                key={`${id}-current`}
                ref={(el) => { trailEls.current[id] = el; }}
                d={d}
                fill="none"
                strokeWidth={2.5}
                strokeLinecap="round"
                style={{ stroke: color, strokeDasharray: 1e5, strokeDashoffset: 1e5 }}
              />
            )}
            {state === "lit" && (
              <>
                <path key={`${id}-lit`} d={d} fill="none" strokeWidth={2.25}
                      strokeDasharray={dash} style={{ stroke: color, strokeOpacity: 0.9 }}
                      markerEnd={arrowFor(color)} />
                {!still && (
                  <path d={d} fill="none" strokeWidth={2} strokeLinecap="round"
                        className="flow-dash"
                        style={{ stroke: "var(--packet-core)", strokeOpacity: 0.6 }} />
                )}
              </>
            )}
          </g>
        );
      })}

      {layout.labels.map((l, i) => (
        <text key={i} x={l.x} y={l.y} textAnchor={l.anchor} fontSize={12} fontStyle="italic"
              style={{ fill: l.tone === "lock" ? COLORS.lock : COLORS.flow }}>
          {l.lines.map((line, j) => (
            <tspan key={j} x={l.x} dy={j === 0 ? 0 : 15}>{line}</tspan>
          ))}
        </text>
      ))}

      {packets.map((p, i) => (
        <g key={`${p.chain}-${i}`} ref={(el) => { packetEls.current[p.chain] = el; }}
           transform={`translate(${p.start[0]} ${p.start[1]})`}>
          <circle r={11} opacity={0.55} filter="url(#packet-glow)" style={{ fill: p.color }} />
          <circle r={6} fill="none" strokeWidth={1.5} style={{ stroke: p.color }} />
          <circle r={3.5} style={{ fill: "var(--packet-core)" }} />
        </g>
      ))}
    </svg>
  );
}
