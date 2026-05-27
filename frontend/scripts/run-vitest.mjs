import { spawnSync } from 'node:child_process';
import path from 'node:path';

const args = process.argv.slice(2).filter((arg) => !arg.startsWith('--watchAll'));
const command = process.platform === 'win32'
  ? path.join('node_modules', '.bin', 'vitest.cmd')
  : path.join('node_modules', '.bin', 'vitest');
const result = spawnSync(command, ['run', ...args], {
  stdio: 'inherit',
  shell: process.platform === 'win32',
});

if (result.error) {
  console.error(result.error);
}

process.exit(result.status ?? 1);
