#!/bin/sh
# Build (if needed) and run the splash menubar app.
cd "$(dirname "$0")"
if [ ! -x .build/release/splash-studio ] || [ Package.swift -nt .build/release/splash-studio ]; then
    swift build -c release || exit 1
fi
exec .build/release/splash-studio "$@"
