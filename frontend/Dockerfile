# syntax=docker/dockerfile:1.7

ARG NODE_IMAGE=node:20-bookworm-slim

FROM ${NODE_IMAGE} AS deps
WORKDIR /app
COPY package.json package-lock.json ./
RUN set -eux; \
  npm ci --include=dev --no-audit --no-fund || true; \
  if [ ! -x ./node_modules/.bin/next ]; then \
    npm cache clean --force || true; \
    rm -rf ./node_modules; \
    npm install --include=dev --no-audit --no-fund; \
  fi; \
  test -x ./node_modules/.bin/next

FROM ${NODE_IMAGE} AS builder
WORKDIR /app
ARG NEXT_PUBLIC_API_BASE_URL=/api
ENV NEXT_PUBLIC_API_BASE_URL=$NEXT_PUBLIC_API_BASE_URL
COPY --from=deps /app/node_modules ./node_modules
COPY . .
RUN npm run build

FROM ${NODE_IMAGE} AS runner
WORKDIR /app
ENV NODE_ENV=production
ENV PORT=3000
ENV HOSTNAME=0.0.0.0
ENV NEXT_PUBLIC_API_BASE_URL=/api
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static
COPY --from=builder /app/public ./public
EXPOSE 3000
CMD ["node", "server.js"]
