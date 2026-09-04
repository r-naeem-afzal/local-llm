import type { Metadata } from "next";

import "./globals.css";

export const metadata: Metadata = {
  title: "Local LLM Dashboard",
  description: "Local model activity and Claude plan usage, side by side.",
};

/**
 * The root layout, required by the Next.js app router — it supplies the <html> and <body>
 * that every page is rendered into.
 *
 * `lang="en"` is not decoration: screen readers choose a pronunciation model from it, and
 * without it they guess from the browser locale, which can be wrong.
 */
export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
