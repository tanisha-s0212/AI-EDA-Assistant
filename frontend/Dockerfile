# Stage 1: Build
FROM node:22-bullseye-slim AS builder

WORKDIR /app

# Install build dependencies if needed
# RUN apt-get update && apt-get install -y python3 make g++ && rm -rf /var/lib/apt/lists/*

COPY package*.json ./
RUN npm install --no-audit --no-fund

COPY . .

# Set environment for build if needed
ENV NEXT_TELEMETRY_DISABLED=1
ENV NODE_ENV=production

RUN npm run build

# Stage 2: Runner
FROM node:22-bullseye-slim AS runner

WORKDIR /app

ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1

# Copy standalone build
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static
COPY --from=builder /app/public ./public
COPY --from=builder /app/start.sh ./start.sh

RUN chmod +x ./start.sh

EXPOSE 3000

ENV PORT=3000
ENV HOSTNAME="0.0.0.0"

CMD ["./start.sh"]
