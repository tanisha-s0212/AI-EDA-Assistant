import type { NextConfig } from "next";

const backendOrigin = process.env.BACKEND_ORIGIN || 'http://127.0.0.1:3004';
const useBackendRewrite = process.env.ENABLE_BACKEND_REWRITE === 'true';

const nextConfig: NextConfig = {
  output: "standalone",
  reactStrictMode: false,
  devIndicators: false,
  experimental: {
    proxyClientMaxBodySize: '600mb',
  },
  async rewrites() {
    if (!useBackendRewrite) {
      return [];
    }

    return [
      {
        source: '/api/:path*',
        destination: `${backendOrigin}/api/:path*`,
      },
    ];
  },
};

export default nextConfig;
