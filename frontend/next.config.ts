import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      {
        source: "/ai/:path*",
        destination: "http://localhost:8001/:path*",
      },
      {
        source: "/api/:path*",
        destination: "http://localhost:8000/:path*",
      },
      {
        // SeaweedFS blob fetch; prod uses nginx X-Accel-Redirect, dev proxies directly.
        source: "/blob/:path*",
        destination: "http://localhost:8333/:path*",
      },
    ];
  },
};

export default nextConfig;
