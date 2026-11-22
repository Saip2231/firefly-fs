// Copies the three self-hosted fonts (SIL Open Font License, from the
// @fontsource-variable packages) into app/static/fonts/, where Streamlit
// serves them for [[theme.fontFaces]] (.streamlit/config.toml).  Latin
// subset, variable weight axis only.  DASHBOARD.md rule 8: no font CDN.
import { copyFileSync, existsSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const frontend = join(here, "..");
const out = join(frontend, "..", "static", "fonts");

const FONTS = ["sora", "plus-jakarta-sans", "jetbrains-mono"];

mkdirSync(out, { recursive: true });
let missing = 0;
for (const id of FONTS) {
  const src = join(frontend, "node_modules", "@fontsource-variable", id, "files",
                   `${id}-latin-wght-normal.woff2`);
  if (!existsSync(src)) {
    console.error(`missing ${src} (run npm install in app/frontend)`);
    missing += 1;
    continue;
  }
  copyFileSync(src, join(out, `${id}.woff2`));
  console.log(`fonts: ${id}.woff2`);
}
if (missing) process.exit(1);
