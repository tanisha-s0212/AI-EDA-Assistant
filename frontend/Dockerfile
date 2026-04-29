# syntax=docker/dockerfile:1.7

ARG NODE_IMAGE=node:20-bookworm-slim

FROM ${NODE_IMAGE} AS deps
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm --version && node --version
RUN npm install -g npm@latest --no-audit --no-fund
RUN npm --version
RUN set -eux; \
  npm config delete proxy || true; \
  npm config delete https-proxy || true; \
  npm config set registry https://registry.npmjs.org/; \
  for attempt in 1 2 3; do \
    npm ci --include=dev --no-audit --no-fund; \
    if [ -x ./node_modules/.bin/next ]; then \
      break; \
    fi; \
    echo "npm install attempt ${attempt} did not produce next binary; retrying..."; \
    npm cache clean --force || true; \
    rm -rf ./node_modules; \
    if [ "$attempt" -eq 3 ]; then exit 1; fi; \
    sleep 5; \
  done
RUN test -x ./node_modules/.bin/next || (echo "ERROR: next binary missing after npm install retries" && exit 1)

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
