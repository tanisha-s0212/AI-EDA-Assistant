from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .config import AGENTIC_ROOT, WORKSPACE_ROOT


IGNORED_DIRS = {
    ".git",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".next",
    "dist",
    "build",
    ".cache",
    "logs",
}

IGNORED_FILES = {
    "package-lock.json",
    "dev-server.err.log",
    "dev-server.out.log",
    "run-frontend.log",
    "backend.log",
    "backend-stdout.log",
    "backend-stderr.log",
}

PRIORITY_FILES = [
    "frontend/src/app/page.tsx",
    "frontend/src/lib/store.ts",
    "frontend/src/lib/api.ts",
    "frontend/src/types/forecast.ts",
    "frontend/src/components/login-page.tsx",
    "frontend/src/components/tabs/upload-tab.tsx",
    "frontend/src/components/tabs/understanding-tab.tsx",
    "frontend/src/components/tabs/eda-tab.tsx",
    "frontend/src/components/tabs/cleaning-tab.tsx",
    "frontend/src/components/tabs/time-series-forecast-tab.tsx",
    "frontend/src/components/tabs/ml-forecast-tab.tsx",
    "frontend/src/components/tabs/loss-forecast-tab.tsx",
    "frontend/src/components/tabs/profit-forecast-tab.tsx",
    "frontend/src/components/tabs/ml-tab.tsx",
    "frontend/src/components/tabs/prediction-tab.tsx",
    "frontend/src/components/tabs/report-tab.tsx",
    "backend/main.py",
]

TEXT_EXTENSIONS = {
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".json",
    ".md",
    ".txt",
    ".yml",
    ".yaml",
    ".html",
    ".css",
    ".scss",
    ".sql",
    ".sh",
    ".ps1",
    ".toml",
    ".ini",
}


@dataclass
class SearchHit:
    path: str
    line: int
    text: str


def _is_ignored(path: Path) -> bool:
    return any(part in IGNORED_DIRS for part in path.parts)


def _safe_relative(path: Path) -> str:
    return path.relative_to(WORKSPACE_ROOT).as_posix()


def list_workspace_tree(max_depth: int = 3, max_entries: int = 160) -> str:
    lines: list[str] = []
    count = 0

    for root, dirs, files in os.walk(WORKSPACE_ROOT):
        root_path = Path(root)
        if _is_ignored(root_path.relative_to(WORKSPACE_ROOT)):
            dirs[:] = []
            continue

        depth = len(root_path.relative_to(WORKSPACE_ROOT).parts)
        if depth > max_depth:
            dirs[:] = []
            continue

        dirs[:] = sorted(d for d in dirs if d not in IGNORED_DIRS)
        files = sorted(files)

        if root_path == WORKSPACE_ROOT:
            lines.append("./")
        else:
            lines.append(f"{'  ' * depth}{root_path.name}/")
            count += 1

        if depth < max_depth:
            for file_name in files[:24]:
                if count >= max_entries:
                    lines.append("... output truncated ...")
                    return "\n".join(lines)
                lines.append(f"{'  ' * (depth + 1)}{file_name}")
                count += 1

    return "\n".join(lines)


def search_workspace(query: str, max_hits: int = 24) -> list[SearchHit]:
    query = query.strip()
    if not query:
        return []

    hits: list[SearchHit] = []
    tokens = [token.lower() for token in query.split() if len(token) > 2]
    lowered = query.lower()

    for path in WORKSPACE_ROOT.rglob("*"):
        if len(hits) >= max_hits:
            break
        if not path.is_file() or path.suffix.lower() not in TEXT_EXTENSIONS:
            continue
        if path.name in IGNORED_FILES:
            continue
        if _is_ignored(path.relative_to(WORKSPACE_ROOT)):
            continue
        if path == AGENTIC_ROOT / ".env":
            continue

        try:
            for index, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
                line_lowered = line.lower()
                if lowered in line_lowered or any(token in line_lowered for token in tokens):
                    hits.append(SearchHit(_safe_relative(path), index, line.strip()[:240]))
                    if len(hits) >= max_hits:
                        break
        except OSError:
            continue

    return hits


def search_priority_files(query: str, max_hits: int = 36) -> list[SearchHit]:
    tokens = [token.lower() for token in query.split() if len(token) > 2]
    if not tokens:
        return []

    hits: list[SearchHit] = []
    for relative_path in PRIORITY_FILES:
        if len(hits) >= max_hits:
            break
        path = WORKSPACE_ROOT / relative_path
        if not path.exists():
            continue
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            continue

        for index, line in enumerate(lines, 1):
            line_lowered = line.lower()
            score = sum(1 for token in tokens if token in line_lowered)
            if score:
                hits.append(SearchHit(relative_path, index, line.strip()[:240]))
                if len(hits) >= max_hits:
                    break

    return hits


def workflow_knowledge() -> str:
    path = AGENTIC_ROOT / "knowledge" / "application-workflow.md"
    if not path.exists():
        return "No workflow knowledge file is available."
    return path.read_text(encoding="utf-8", errors="ignore")


def read_file_excerpt(relative_path: str, max_chars: int = 6000) -> str:
    candidate = (WORKSPACE_ROOT / relative_path).resolve()
    if not str(candidate).startswith(str(WORKSPACE_ROOT.resolve())):
        raise ValueError("Path is outside the workspace.")
    if candidate == AGENTIC_ROOT / ".env":
        raise ValueError("The local .env file is intentionally not readable through the agent.")
    if not candidate.exists() or not candidate.is_file():
        raise FileNotFoundError(relative_path)

    text = candidate.read_text(encoding="utf-8", errors="ignore")
    if len(text) > max_chars:
        return text[:max_chars] + "\n\n... excerpt truncated ..."
    return text


def format_hits(hits: list[SearchHit]) -> str:
    if not hits:
        return "No direct text matches found."
    return "\n".join(f"- {hit.path}:{hit.line}: {hit.text}" for hit in hits)
