import { animate } from "framer-motion";
import { useEffect, useLayoutEffect, useRef, useState, type RefObject } from "react";

/** Smoothly counts from the previous value to `target`; instant when `still`. */
export function useCountUp(target: number | null, still: boolean, duration = 0.7): number | null {
  const [value, setValue] = useState<number>(target ?? 0);
  const current = useRef<number>(target ?? 0);
  useEffect(() => {
    if (target === null) return;
    if (still) {
      current.current = target;
      setValue(target);
      return;
    }
    const controls = animate(current.current, target, {
      duration,
      ease: "easeOut",
      onUpdate: (v) => {
        current.current = v;
        setValue(v);
      },
    });
    return () => controls.stop();
  }, [target, still, duration]);
  return target === null ? null : value;
}

/** Width of an element, tracked with ResizeObserver. */
export function useWidth(ref: RefObject<HTMLElement>): number {
  const [width, setWidth] = useState(0);
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    setWidth(el.clientWidth);
    const ro = new ResizeObserver((entries) => setWidth(entries[0].contentRect.width));
    ro.observe(el);
    return () => ro.disconnect();
  }, [ref]);
  return width;
}

export type ThemeMode = "light" | "dark";

function hostTheme(): ThemeMode | null {
  try {
    const t = window.parent?.document?.documentElement?.dataset?.ffTheme;
    return t === "light" || t === "dark" ? t : null;
  } catch {
    return null;                      // cross-origin host: fall back below
  }
}

const systemTheme = (): ThemeMode =>
  window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light";

/**
 * Theme of the page that embeds us.  Inside the Streamlit app the parent's
 * <html data-ff-theme> (kept in sync by app/lib/theme.py) is watched, so a
 * switch in the app menu reaches the diagram at once; otherwise the mode the
 * server rendered with, then the system preference.  Applies `.dark` on our
 * own <html> so the CSS variables in index.css follow.
 */
export function useThemeMode(fallback: ThemeMode | null): ThemeMode {
  const [mode, setMode] = useState<ThemeMode>(() => hostTheme() ?? fallback ?? systemTheme());
  useEffect(() => {
    const update = () => setMode(hostTheme() ?? fallback ?? systemTheme());
    let observer: MutationObserver | null = null;
    try {
      const root = window.parent?.document?.documentElement;
      if (root && root !== document.documentElement) {
        observer = new MutationObserver(update);
        observer.observe(root, { attributes: true, attributeFilter: ["data-ff-theme"] });
      }
    } catch {
      observer = null;
    }
    const mq = window.matchMedia?.("(prefers-color-scheme: dark)");
    mq?.addEventListener?.("change", update);
    update();
    return () => {
      observer?.disconnect();
      mq?.removeEventListener?.("change", update);
    };
  }, [fallback]);
  useLayoutEffect(() => {
    document.documentElement.classList.toggle("dark", mode === "dark");
  }, [mode]);
  return mode;
}

/** Ask the Streamlit host iframe to fit our height (ignored outside Streamlit). */
export function usePostFrameHeight(deps: unknown[]): void {
  useEffect(() => {
    const height = Math.ceil(document.documentElement.scrollHeight);
    window.parent?.postMessage(
      { isStreamlitMessage: true, type: "streamlit:setFrameHeight", height }, "*");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
}
