import type { NextConfig } from "next";

const backendOrigin = process.env.BACKEND_ORIGIN || 'http://127.0.0.1:3004';

const nextConfig: NextConfig = {
  output: "standalone",
  reactStrictMode: false,
  devIndicators: false,
  experimental: {
    proxyClientMaxBodySize: '250mb',
  },
  env: {
    NEXT_PUBLIC_API_BASE_URL: process.env.NEXT_PUBLIC_API_BASE_URL || '/api',
  },
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: `${backendOrigin}/api/:path*`,
      },
    ];
  },
};

export default nextConfig;

