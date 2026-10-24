#!/bin/sh
set -eu

cd "$(dirname "$0")"
# SwiftPM checks source dependencies and reuses its incremental build cache.
# Checking only Package.swift leaves an old executable after source edits.
swift build -c release
exec .build/release/splash-studio "$@"
