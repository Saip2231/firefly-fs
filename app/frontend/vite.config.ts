import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { viteSingleFile } from "vite-plugin-singlefile";

// Builds ONE self-contained HTML file (JS and CSS inlined, no external
// requests: DASHBOARD.md rule 8) into app/assets/pipeline/index.html.
// app/lib/pipeline_view.py injects the result data into it at runtime.
export default defineConfig({
  plugins: [react(), viteSingleFile()],
  publicDir: false,
  build: {
    outDir: "../assets/pipeline",
    emptyOutDir: true,
    target: "es2020",
    reportCompressedSize: false,
  },
});
