# Dockerfile for Next.js Frontend
FROM node:22-alpine

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy project files
COPY . .

# Build arguments
ARG NEXT_PUBLIC_API_BASE_URL
ENV NEXT_PUBLIC_API_BASE_URL=${NEXT_PUBLIC_API_BASE_URL}

# Limit Node.js memory to prevent OOM kills on low-memory servers
ENV NODE_OPTIONS="--max-old-space-size=512"

# Bind to all interfaces so other containers (nginx) can reach this server
ENV HOSTNAME="0.0.0.0"

# Build the application
RUN npm run build

# Expose port
EXPOSE 3000

# Start the application — bind explicitly to 0.0.0.0
CMD ["npm", "start"]
