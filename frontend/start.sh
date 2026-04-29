#!/bin/sh
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
  const file = path.join('/app/public', req.url === '/' ? 'index.html' : req.url);
  if (fs.existsSync(file)) {
    res.end(fs.readFileSync(file));
  } else {
    res.writeHead(200, {'Content-Type': 'text/html'});
    res.end('<h1>Frontend</h1><p>API: /api</p>');
  }
});
server.listen(3000, '0.0.0.0', () => console.log('Frontend on :3000'));
"
