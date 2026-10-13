#!/bin/sh
# Fail on accidental developer-library links or a higher OS/glibc requirement.
set -eu
binary=$1
target=$2
report=$3
case "$target" in
  aarch64-apple-darwin)
    test "$(uname -s)" = Darwin
    file "$binary" | grep -q 'Mach-O 64-bit executable arm64'
    otool -L "$binary" > "$report"
    otool -l "$binary" >> "$report"
    sed 's|^.*\/bin\/game-screen-pick:$|bin/game-screen-pick:|' "$report" > "$report.normalized"
    mv "$report.normalized" "$report"
    # System dylibs/frameworks are supplied by macOS, never Homebrew or an rpath.
    otool -L "$binary" | awk 'NR > 1 {print $1}' | while IFS= read -r dependency; do
      case "$dependency" in /usr/lib/*|/System/Library/Frameworks/*) ;;
        *) printf 'Unsupported dynamic library: %s\n' "$dependency" >&2; exit 1 ;;
      esac
    done
    minimum=$(otool -l "$binary" | awk '/cmd LC_BUILD_VERSION/ {build=1} build && $1 == "minos" {print $2; build=0}')
    test "$minimum" = 14.0
    ;;
  x86_64-unknown-linux-gnu)
    test "$(uname -s)" = Linux
    file "$binary" | grep -q 'ELF 64-bit LSB.*x86-64'
    readelf -d "$binary" > "$report"
    readelf --version-info "$binary" >> "$report"
    # Loader addresses vary with ASLR; retain every resolved library and path.
    ldd "$binary" | sed -E 's/ \(0x[0-9a-f]+\)//g' >> "$report"
    if grep -q 'not found' "$report"; then
      printf 'Unresolved dynamic library\n' >&2; exit 1
    fi
    readelf -d "$binary" | awk '/\(NEEDED\)/ {gsub(/.*\[/, ""); gsub(/\].*/, ""); print}' \
      | while IFS= read -r dependency; do
        case "$dependency" in libc.so.6|libm.so.6|libgcc_s.so.1|libpthread.so.0|libdl.so.2|librt.so.1|ld-linux-x86-64.so.2) ;;
          *) printf 'Unsupported dynamic library: %s\n' "$dependency" >&2; exit 1 ;;
        esac
      done
    if readelf -d "$binary" | grep -Eq '\((RPATH|RUNPATH)\)'; then
      printf 'Runtime library search path must not be embedded\n' >&2; exit 1
    fi
    readelf --version-info "$binary" | grep -oE 'GLIBC_[0-9]+\.[0-9]+(\.[0-9]+)?' | sort -u \
      | awk -F_ '{split($2,v,"."); if (v[1] > 2 || (v[1] == 2 && (v[2] > 35 || (v[2] == 35 && v[3] > 0)))) bad=1} END {exit bad}'
    ;;
  *) printf 'Unsupported target: %s\n' "$target" >&2; exit 1 ;;
esac
