#!/bin/sh
# CI/developer tooling only. Build the verified 6.1-series source on both target
# platforms; Ubuntu 22.04's FFmpeg 4.4 lacks -fps_mode and Homebrew changes versions.
# This installs no runtime dependency into the product archive or the user's OS.
set -eu
prefix=$1
case "$(uname -s)" in
  Darwin)
    jobs=$(sysctl -n hw.ncpu)
    export MACOSX_DEPLOYMENT_TARGET=14.0
    ;;
  Linux) jobs=$(getconf _NPROCESSORS_ONLN) ;;
  *) printf 'Unsupported CI FFmpeg build platform\n' >&2; exit 1 ;;
esac
mkdir -p "$prefix"
prefix=$(cd "$prefix" && pwd)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT HUP INT TERM
version=6.1.6
sha256=d4fcb164028dd3beee5d92c0ac72e46aac6973c75ea12dc14de07bf8f407370a
curl --fail --silent --show-error --location --retry 3 \
  "https://ffmpeg.org/releases/ffmpeg-$version.tar.xz" --output "$work/source.tar.xz"
printf '%s  %s\n' "$sha256" "$work/source.tar.xz" | shasum -a 256 -c -
tar -xJf "$work/source.tar.xz" -C "$work"
cd "$work/ffmpeg-$version"
./configure --prefix="$prefix" --disable-shared --enable-static --disable-autodetect \
  --enable-zlib --disable-x86asm --disable-doc --disable-debug --disable-ffplay --disable-network
make -j "$jobs"
make install
"$prefix/bin/ffmpeg" -version
"$prefix/bin/ffprobe" -version
