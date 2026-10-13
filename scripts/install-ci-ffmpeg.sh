#!/bin/sh
# CI/developer tooling only. Ubuntu 22.04's FFmpeg 4.4 lacks -fps_mode; build the
# verified 6.1-series source while retaining the glibc 2.35 distribution baseline.
# This installs no runtime dependency into the product archive or the user's OS.
set -eu
prefix=$1
mkdir -p "$prefix"
prefix=$(cd "$prefix" && pwd)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT HUP INT TERM
version=6.1.6
sha256=d4fcb164028dd3beee5d92c0ac72e46aac6973c75ea12dc14de07bf8f407370a
curl --fail --silent --show-error --location --retry 3 \
  "https://ffmpeg.org/releases/ffmpeg-$version.tar.xz" --output "$work/source.tar.xz"
printf '%s  %s\n' "$sha256" "$work/source.tar.xz" | sha256sum -c -
tar -xJf "$work/source.tar.xz" -C "$work"
cd "$work/ffmpeg-$version"
./configure --prefix="$prefix" --disable-shared --enable-static --disable-autodetect \
  --enable-zlib --disable-x86asm --disable-doc --disable-debug --disable-ffplay --disable-network
make -j "$(getconf _NPROCESSORS_ONLN)"
make install
"$prefix/bin/ffmpeg" -version
"$prefix/bin/ffprobe" -version
