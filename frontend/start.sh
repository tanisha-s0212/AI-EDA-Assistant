#!/bin/sh

set -eu

if [ ! -f "/app/server.js" ]; then
    echo "Next.js standalone server is missing; the image build is incomplete." >&2
    exit 1
fi

exec node /app/server.js
