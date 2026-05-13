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
        // SeaweedFS blob fetch for plant photos. In prod nginx serves this via
        // X-Accel-Redirect; in dev we proxy directly so <Image /> can load.
        source: "/blob/:path*",
        destination: "http://localhost:8333/:path*",
      },
    ];
  },
};

export default nextConfig;
