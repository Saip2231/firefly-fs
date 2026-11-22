import { StrictMode, useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import "./index.css";
import PipelineVisualizer from "./PipelineVisualizer";
import type { PipelinePayload } from "./types";

/** Payload injected by app/lib/pipeline_view.py; null if the placeholder is still there. */
function readInjected(): PipelinePayload | null {
  const text = document.getElementById("pipeline-data")?.textContent?.trim() ?? "";
  if (!text || text.startsWith("__")) return null;
  try {
    return JSON.parse(text) as PipelinePayload;
  } catch {
    return null;
  }
}

function App() {
  const [payload, setPayload] = useState<PipelinePayload | null>(readInjected);
  const [ready, setReady] = useState(payload !== null || !import.meta.env.DEV);

  // `npm run dev` only: read app/frontend/dev-data.json (written by
  // `python -m app.lib.pipeline_view app/frontend/dev-data.json`).
  useEffect(() => {
    if (ready) return;
    fetch("/dev-data.json")
      .then((r) => (r.ok ? r.json() : null))
      .then((d: PipelinePayload | null) => setPayload(d))
      .catch(() => setPayload(null))
      .finally(() => setReady(true));
  }, [ready]);

  return ready ? <PipelineVisualizer payload={payload} /> : null;
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
