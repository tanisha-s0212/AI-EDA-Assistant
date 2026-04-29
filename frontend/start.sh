#!/bin/sh

# If the standalone server exists, start it.
if [ -f "/app/.next/standalone/server.js" ]; then
    echo "Starting standalone Next.js server..."
    # Next.js standalone server uses PORT environment variable
    export PORT=3000
    export HOSTNAME="0.0.0.0"
    node /app/.next/standalone/server.js
else
    echo "Standalone server not found, falling back to mock server."
    node -e "
    const http = require('http');
    const fs = require('fs');
    const path = require('path');
    const server = http.createServer((req, res) => {
      if (req.url === '/health') {
        res.writeHead(200);
        res.end('OK');
        return;
      }
      res.writeHead(200, {'Content-Type': 'text/html'});
      res.end('<h1>Frontend Placeholder</h1><p>The real frontend build is missing or failed.</p>');
    });
    server.listen(3000, '0.0.0.0', () => console.log('Mock Frontend on :3000'));
    "
fi
