import { cp, mkdir, stat } from 'node:fs/promises';
import path from 'node:path';

const projectRoot = process.cwd();
const standaloneRoot = path.join(projectRoot, '.next', 'standalone');
const standaloneNextRoot = path.join(standaloneRoot, '.next');

async function pathExists(targetPath) {
  try {
    await stat(targetPath);
    return true;
  } catch {
    return false;
  }
}

async function copyIfPresent(sourcePath, destinationPath) {
  if (!(await pathExists(sourcePath))) {
    return;
  }

  await cp(sourcePath, destinationPath, { recursive: true, force: true });
}

await mkdir(standaloneNextRoot, { recursive: true });
await copyIfPresent(path.join(projectRoot, '.next', 'static'), path.join(standaloneNextRoot, 'static'));
await copyIfPresent(path.join(projectRoot, 'public'), path.join(standaloneRoot, 'public'));
