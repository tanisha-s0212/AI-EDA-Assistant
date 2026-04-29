#!/usr/bin/env bash
set -euo pipefail

echo "Detecting Windows proxy..."

# Try WinHTTP proxy first
WINHTTP_PROXY=$(powershell.exe -NoProfile -Command \
  "(netsh winhttp show proxy | Select-String 'Proxy Server').ToString()" \
  2>/dev/null | tr -d '\r' || true)

PROXY_HOSTPORT=""
if [[ "$WINHTTP_PROXY" == *":"* ]]; then
  # Extract text after colon.
  PROXY_HOSTPORT=$(echo "$WINHTTP_PROXY" | sed -E 's/.*:\s*//')
fi

# Fallback: user-level env vars in Windows.
if [[ -z "${PROXY_HOSTPORT}" ]]; then
  HTTP_PROXY_WIN=$(powershell.exe -NoProfile -Command \
    "[Environment]::GetEnvironmentVariable('HTTP_PROXY','User')" \
    2>/dev/null | tr -d '\r' || true)
  HTTPS_PROXY_WIN=$(powershell.exe -NoProfile -Command \
    "[Environment]::GetEnvironmentVariable('HTTPS_PROXY','User')" \
    2>/dev/null | tr -d '\r' || true)

  if [[ -n "${HTTPS_PROXY_WIN}" ]]; then
    export HTTPS_PROXY="${HTTPS_PROXY_WIN}"
    export HTTP_PROXY="${HTTP_PROXY_WIN:-$HTTPS_PROXY_WIN}"
  elif [[ -n "${HTTP_PROXY_WIN}" ]]; then
    export HTTP_PROXY="${HTTP_PROXY_WIN}"
    export HTTPS_PROXY="${HTTP_PROXY_WIN}"
  fi
fi

# Build proxy URL from host:port if needed.
if [[ -z "${HTTP_PROXY:-}" && -n "${PROXY_HOSTPORT}" ]]; then
  export HTTP_PROXY="http://${PROXY_HOSTPORT}"
  export HTTPS_PROXY="http://${PROXY_HOSTPORT}"
fi

export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1,host.docker.internal}"

echo "HTTP_PROXY=${HTTP_PROXY:-<not set>}"
echo "HTTPS_PROXY=${HTTPS_PROXY:-<not set>}"
echo "NO_PROXY=${NO_PROXY}"

if [[ -z "${HTTP_PROXY:-}" || -z "${HTTPS_PROXY:-}" ]]; then
  echo "Proxy not detected automatically."
  echo "Set manually, e.g.:"
  echo 'export HTTP_PROXY="http://host:port"'
  echo 'export HTTPS_PROXY="http://host:port"'
  exit 1
fi

echo "Running frontend Docker build..."
docker compose build \
  --build-arg HTTP_PROXY="${HTTP_PROXY:-}" \
  --build-arg HTTPS_PROXY="${HTTPS_PROXY:-}" \
  --build-arg NO_PROXY="${NO_PROXY:-}" \
  --build-arg NPM_REGISTRY="http://registry.npmjs.org/" \
  --build-arg NPM_STRICT_SSL="false" \
  --no-cache frontend
