import { spawn } from 'node:child_process';
import path from 'node:path';

const serverPath = path.join(process.cwd(), '.next', 'standalone', 'server.js');

const child = spawn(process.execPath, [serverPath], {
  stdio: 'inherit',
  env: {
    ...process.env,
    NODE_ENV: 'production',
  },
});

child.on('exit', (code, signal) => {
  if (signal) {
    process.kill(process.pid, signal);
    return;
  }

  process.exit(code ?? 0);
});
