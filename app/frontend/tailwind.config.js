/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  darkMode: "class",
  theme: {
    extend: {
      // Semantic tokens (CSS variables from src/index.css).
      colors: {
        page: "var(--bg)",
        surface: "var(--surface)",
        line: "var(--border)",
        ink: "var(--text)",
        muted: "var(--muted)",
        accent: "var(--accent)",
      },
      boxShadow: {
        card: "var(--card-shadow)",
      },
      // Same families as the Streamlit app (app/lib/theme.py FONTS).
      fontFamily: {
        sans: ["Plus Jakarta Sans", "system-ui", "-apple-system", "Segoe UI", "Roboto",
               "Helvetica", "Arial", "sans-serif"],
        heading: ["Sora", "system-ui", "-apple-system", "Segoe UI", "sans-serif"],
        mono: ["JetBrains Mono", "ui-monospace", "SFMono-Regular", "Cascadia Mono", "Menlo",
               "Consolas", "monospace"],
      },
    },
  },
  plugins: [],
};
