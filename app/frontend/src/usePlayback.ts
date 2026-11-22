// Playback state machine for the protocol animation: one step at a time,
// packets travel the step's edge chains (rAF, written straight to the DOM so
// React does not re-render 60 times a second), then the node "arrives" and
// dwells.  Play auto-advances, Step does one stage, Replay restarts.

import { useCallback, useEffect, useRef, useState } from "react";
import type { EdgeId } from "./layout";
import type { Step } from "./steps";

const TRAVEL_MS = 1100;
const DWELL_MS = 900;
const SEARCH_DWELL_MS = 1700;
const STILL_DWELL_MS = 1300;
export const SEARCH_DWELL_S = SEARCH_DWELL_MS / 1000;

const ease = (t: number) => (t < 0.5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2);

export function usePlayback(steps: Step[], still: boolean, resetKey: unknown) {
  const [step, setStep] = useState(-1);
  const [arrived, setArrived] = useState(false);
  const [running, setRunning] = useState(false);
  const [autoplay, setAutoplayState] = useState(false);
  const autoplayRef = useRef(false);
  const elapsed = useRef(0);
  const pathEls = useRef<Partial<Record<EdgeId, SVGPathElement | null>>>({});
  const trailEls = useRef<Partial<Record<EdgeId, SVGPathElement | null>>>({});
  const packetEls = useRef<(SVGGElement | null)[]>([]);

  const setAutoplay = useCallback((v: boolean) => {
    autoplayRef.current = v;
    setAutoplayState(v);
  }, []);

  const goTo = useCallback((s: number) => {
    elapsed.current = 0;
    setArrived(false);
    setStep(s);
  }, []);

  // A different path (track) starts again from the top.
  useEffect(() => {
    setAutoplay(false);
    setRunning(false);
    goTo(-1);
  }, [resetKey, goTo, setAutoplay]);

  const drawPackets = useCallback((chains: EdgeId[][], p: number) => {
    chains.forEach((chain, ci) => {
      const lens = chain.map((e) => pathEls.current[e]?.getTotalLength() ?? 0);
      let dist = p * lens.reduce((a, b) => a + b, 0);
      let point: DOMPoint | null = null;
      chain.forEach((e, i) => {
        const L = lens[i];
        const local = Math.max(0, Math.min(L, dist));
        const trail = trailEls.current[e];
        if (trail) {
          trail.style.strokeDasharray = `${L}`;
          trail.style.strokeDashoffset = `${L - local}`;
        }
        const path = pathEls.current[e];
        if (!point && path && (dist <= L || i === chain.length - 1)) {
          point = path.getPointAtLength(local);
        }
        dist -= L;
      });
      const g = packetEls.current[ci];
      const pt = point as DOMPoint | null;
      if (g && pt) g.setAttribute("transform", `translate(${pt.x} ${pt.y})`);
    });
  }, []);

  useEffect(() => {
    if (!running || step < 0) return;
    const cur = steps[step];
    const travel = still || cur.chains.length === 0 ? 0 : TRAVEL_MS;
    const dwell = still ? STILL_DWELL_MS : cur.node === "search" ? SEARCH_DWELL_MS : DWELL_MS;
    let raf = 0;
    let last = performance.now();
    const tick = (now: number) => {
      elapsed.current += Math.min(now - last, 64);
      last = now;
      const p = travel === 0 ? 1 : Math.min(1, elapsed.current / travel);
      if (travel > 0) drawPackets(cur.chains, ease(p));
      if (p >= 1) setArrived(true);
      if (elapsed.current >= travel + dwell) {
        if (autoplayRef.current && step < steps.length - 1) goTo(step + 1);
        else {
          setRunning(false);
          setAutoplay(false);
        }
        return;
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [running, step, still, steps, drawPackets, goTo, setAutoplay]);

  const finished = step === steps.length - 1 && arrived && !running;
  const playing = running && autoplay;

  const onPlay = () => {
    if (playing) {
      setAutoplay(false);
      setRunning(false);
      return;
    }
    setAutoplay(true);
    if (step < 0 || finished) goTo(0);
    setRunning(true);
  };
  const onStep = () => {
    setAutoplay(false);
    if (finished) goTo(0);
    else if (step >= 0 && !arrived) elapsed.current = still ? 0 : TRAVEL_MS; // jump to arrival
    else if (step < steps.length - 1) goTo(step + 1);
    setRunning(true);
  };
  const onReplay = () => {
    setAutoplay(true);
    goTo(0);
    setRunning(true);
  };

  return { step, arrived, playing, onPlay, onStep, onReplay, pathEls, trailEls, packetEls };
}
