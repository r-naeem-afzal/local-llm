import type { NextConfig } from "next";

/**
 * The dashboard is a pure client of the Python API — it has no database access and no
 * server-side data of its own, which is why there are no route handlers here.
 *
 * `reactStrictMode` deliberately stays on. In development it mounts every component
 * twice to surface effects that are not safe to run repeatedly, and an SSE subscription
 * is exactly that kind of effect: a listener that is opened without being closed leaks a
 * connection per mount. Better to have that fail loudly here than to slowly accumulate
 * open streams against the API.
 */
const nextConfig: NextConfig = {
  reactStrictMode: true,
};

export default nextConfig;
