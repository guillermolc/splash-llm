#!/bin/sh
set -eu

cd "$(dirname "$0")"
swift build -c release --product splash-studio
studio_bin_dir=$(swift build -c release --show-bin-path)
studio_bundle="$(pwd)/.build/Splash Studio.app"
mkdir -p "$studio_bundle/Contents/MacOS" "$studio_bundle/Contents/Resources"
cp "$studio_bin_dir/splash-studio" "$studio_bundle/Contents/MacOS/splash-studio"
cp Resources/Info.plist "$studio_bundle/Contents/Info.plist"
codesign --force --sign - "$studio_bundle"
printf '%s\n' "$studio_bundle"
