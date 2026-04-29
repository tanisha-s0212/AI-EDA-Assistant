# syntax=docker/dockerfile:1.7

ARG NODE_IMAGE=node:20-bookworm-slim
ARG NPM_REGISTRY=https://registry.npmjs.org/
ARG NPM_STRICT_SSL=false
ARG CORPORATE_CA_CERT_B64=

FROM ${NODE_IMAGE} AS deps
ARG CORPORATE_CA_CERT_B64=
ARG NPM_REGISTRY=https://registry.npmjs.org/
ARG NPM_STRICT_SSL=false
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm --version && node --version
# Approach 1: If corporate CA is provided (base64-encoded PEM/CRT), trust it in the container.
# This is the preferred fix for SSL-intercepting proxy environments.
RUN set -eux; \
  if [ -n "${CORPORATE_CA_CERT_B64}" ]; then \
    echo "${CORPORATE_CA_CERT_B64}" | base64 -d > /usr/local/share/ca-certificates/corporate-proxy.crt; \
    update-ca-certificates; \
  fi
RUN set -eux; \
  npm config delete proxy || true; \
  npm config delete https-proxy || true; \
  yarn config delete proxy || true; \
  yarn config delete https-proxy || true; \
  if [ -f /usr/local/share/ca-certificates/corporate-proxy.crt ]; then \
    npm config set cafile /usr/local/share/ca-certificates/corporate-proxy.crt; \
    yarn config set cafile /usr/local/share/ca-certificates/corporate-proxy.crt; \
  fi; \
  # Approach 2: Allow internal npm registry/mirror via build-arg override.
  npm config set registry "${NPM_REGISTRY}"; \
  yarn config set registry "${NPM_REGISTRY}"; \
  # Approach 3 (last resort for SSL MITM environments): disable strict SSL for npm only.
  npm config set strict-ssl "${NPM_STRICT_SSL}"; \
  yarn config set strict-ssl "${NPM_STRICT_SSL}"
RUN npm install -g yarn --prefer-offline || npm install -g yarn
# Yarn is used instead of npm install/npm ci because npm repeatedly crashes in this proxy environment
# with "Exit handler never called!" after long-lived connection interruptions.
RUN set -eux; \
  if [ ! -f yarn.lock ] && [ -f package-lock.json ]; then \
    yarn import; \
  fi
RUN yarn install --frozen-lockfile --non-interactive --network-timeout 300000
RUN test -x ./node_modules/.bin/next || { echo "ERROR: next binary missing after yarn install"; exit 1; }

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
