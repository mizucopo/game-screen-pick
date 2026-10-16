#!/usr/bin/env python3
"""Product evidence checks and a local adapter to the Copier Rust builder."""

import argparse
import importlib.util
import json
import os
import re
import shutil
import stat
import sys
import tarfile
import tempfile
import tomllib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Loading the shared module must not dirty the source checkout before staging.
sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location(
    "release_controller", ROOT / ".github/scripts/release.py"
)
controller = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controller)
TARGETS = json.loads((ROOT / "scripts/distribution-targets.json").read_text(encoding="utf-8"))
REQUIRED = (
    "BUILD-INFO.json DEPENDENCIES.txt LICENSE config.example.toml README.md "
    "CONTRIBUTING.md assets/OFL.txt licenses/NotoSansJP-OFL.txt "
    "licenses/THIRD-PARTY-LICENSES.txt licenses/dependencies.json "
    "licenses/rust/COPYRIGHT.html licenses/rust/SOURCE-LICENSES.txt "
    "docs/release.md docs/technical-reference.md docs/ai-client.md "
    "docs/brave-search.md docs/acceptance.md docs/verification.md"
).split()


def verify(directory, version, commit):
    controller.version_key(version, "semver")
    controller.require(re.fullmatch(r"[0-9a-f]{40}", commit), "Invalid source commit")
    names = [controller.rust_name("game-screen-pick", version, t) for t in TARGETS]
    expected = {name + suffix for name in names for suffix in ("", ".sha256")}
    controller.require(
        {p.name for p in directory.iterdir()} == expected
        and all(p.is_file() and not p.is_symlink() for p in directory.iterdir()),
        "Release requires exactly four archives and their checksums",
    )
    for target, name in zip(TARGETS, names):
        archive = directory / name
        digest = controller.file_sha256(archive)
        controller.require(
            (directory / (name + ".sha256")).read_bytes()
            == f"{digest}  {name}\n".encode(),
            f"Archive checksum differs: {name}",
        )
        # The common extractor rejects unsafe paths and non-regular tar entries.
        # Product ZIPs also forbid links/devices, directories and encryption.
        if name.endswith(".zip"):
            with zipfile.ZipFile(archive) as bundle:
                controller.require(
                    all(
                        not entry.is_dir()
                        and not entry.flag_bits & 1
                        and stat.S_IFMT(entry.external_attr >> 16) in (0, stat.S_IFREG)
                        for entry in bundle.infolist()
                    ),
                    "Product ZIP must contain regular unencrypted files",
                )
        else:
            # Validate archived POSIX permissions, including when this guard
            # runs on Windows where host stat cannot represent executable bits.
            with tarfile.open(archive) as bundle:
                controller.require(
                    any(entry.name == "game-screen-pick" and entry.mode & 0o111 for entry in bundle.getmembers()),
                    "Archived binary is not executable",
                )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            members = controller.rust_extract(archive, root)
            controller.require(
                all(not re.search(r"\s", member) for member in members),
                "Unexpected whitespace in archive path",
            )
            binary_name = "game-screen-pick" + (".exe" if "windows" in target else "")
            for required in [binary_name, *REQUIRED]:
                path = root / required
                controller.require(
                    path.is_file() and path.stat().st_size > 0,
                    f"Missing product evidence: {target}/{required}",
                )
            binary = root / binary_name
            info = json.loads((root / "BUILD-INFO.json").read_text(encoding="utf-8"))
            required_info = {
                "schema": 1,
                "version": version,
                "target": target,
                "commit": commit,
                "minimum_os": TARGETS[target]["minimum_os"],
                "binary_sha256": controller.file_sha256(binary),
                "external_commands": ["ffmpeg", "ffprobe"],
            }
            controller.require(
                all(info.get(key) == value for key, value in required_info.items())
                and info.get("dirty") is False
                and all(isinstance(info.get(key), str) and info[key] for key in ("rustc", "cargo")),
                f"Product build provenance differs: {target}",
            )
            licenses = json.loads((root / "licenses/dependencies.json").read_text(encoding="utf-8"))
            controller.require(
                isinstance(licenses, list)
                and licenses
                and all(
                    isinstance(item, dict)
                    and all(isinstance(item.get(key), str) and item[key] for key in ("name", "version", "license"))
                    for item in licenses
                ),
                f"Missing dependency license inventory: {target}",
            )
    print(f"Verified four target archives at {version} ({commit})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    build = commands.add_parser("build")
    build.add_argument("target", choices=TARGETS)
    build.add_argument("directory", type=Path)
    check = commands.add_parser("verify")
    check.add_argument("directory", type=Path)
    check.add_argument("version")
    check.add_argument("commit")
    commands.add_parser("upload")
    flatten = commands.add_parser("flatten")
    flatten.add_argument("source", type=Path)
    flatten.add_argument("directory", type=Path)
    flatten.add_argument("version")
    flatten.add_argument("commit")
    args = parser.parse_args()
    if args.action == "verify":
        verify(args.directory, args.version, args.commit)
        return
    if args.action == "flatten":
        policy = json.loads((ROOT / ".github/release.json").read_text(encoding="utf-8"))
        plan = {"release_tag": args.version, "rust": policy["publication"]["rust"]}
        archives = controller.rust_deliverables(controller.Git(ROOT), plan, args.commit, args.source)
        args.directory.mkdir(parents=True, exist_ok=True)
        controller.require(not any(args.directory.iterdir()), "Flatten destination must be empty")
        for archive in archives:
            for source in (archive, archive.with_name(archive.name + ".sha256")):
                shutil.copy2(source, args.directory / source.name)
        verify(args.directory, args.version, args.commit)
        return
    if args.action == "upload":
        # Product preflight has checked every retained byte and permissible
        # starter. Revalidate the numbering commit before common asset upload.
        controller.require(os.environ.get("GITHUB_REF") == "refs/heads/main", "Release requires main")
        git, gh = controller.Git(ROOT), controller.GitHub()
        plan = controller.checkout_plan(git)["publication"]
        releases = [release for release in gh.pages("/releases") if release["tag_name"] == plan["release_tag"]]
        controller.require(len(releases) == 1, "Expected one GitHub Release")
        release = releases[0]
        release_id = controller.checked_release_id(release, plan)
        controller.require(
            release.get("target_commitish") == os.environ["RELEASE_SHA"]
            and release.get("prerelease") == plan["is_prerelease"]
            and (not os.environ.get("RELEASE_ID") or str(release_id) == os.environ["RELEASE_ID"]),
            "Release provenance differs from numbering plan",
        )
        controller.rust_asset(git, gh, release, plan, gh.pages(f"/releases/{release_id}/assets"))
        return
    # PR/local builds have no numbering commit. Reuse the exact common builder,
    # Cargo artifact selection, names, archive, checksum and native startup smoke.
    policy = json.loads((ROOT / ".github/release.json").read_text(encoding="utf-8"))
    version = tomllib.loads((ROOT / "Cargo.toml").read_text(encoding="utf-8"))["package"]["version"]
    plan = {"release_tag": version, "rust": policy["publication"]["rust"]}
    git = controller.Git(ROOT)
    destination = args.directory.resolve()
    controller.build_rust_distribution(
        git, plan, git.text("rev-parse", "HEAD"), args.target, destination
    )
    sys.stdout.write(str(destination / args.target / controller.rust_name("game-screen-pick", version, args.target)))


if __name__ == "__main__":
    try:
        main()
    except controller.PreparationError as error:
        raise SystemExit(str(error)) from error
