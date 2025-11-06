#!/usr/bin/env python3
import argparse
import os
import shutil
import sys
import glob
import hashlib
import zipfile
from pathlib import Path

REQUIRED = [
    'app',
    'src',
    'requirements.txt',
    'README.md'
]
# Optional-but-desired items; warn if missing
OPTIONAL = [
    'models',
    'data/processed/train_multilabel.csv',
    'data/processed/val_multilabel.csv',
    'data/processed/test_multilabel.csv',
    'data/processed/merged_multilabel.csv',
    'scripts',
    'REVIEWER_GUIDE.md',
    'RUN_INSTRUCTIONS.md',
    'LICENSE'
]

INCLUDE_GLOBS = [
    'models/*.pth',
    'models/*config.json'
]

MAX_ZIP_MB = 2500


def fail(msg):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def copy_path(src, dst):
    src_p = Path(src)
    dst_p = Path(dst)
    if src_p.is_dir():
        shutil.copytree(src_p, dst_p, dirs_exist_ok=True)
    elif src_p.is_file():
        dst_p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_p, dst_p)
    else:
        fail(f"Path not found: {src}")


def prepare_staging_area(repo, dist, stage):
    """Prepare and clean staging directory."""
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True, exist_ok=True)
    dist.mkdir(parents=True, exist_ok=True)


def verify_required_paths(repo):
    """Verify all required paths exist."""
    missing = [p for p in REQUIRED if not (repo / p).exists()]
    if missing:
        fail(f"Missing required path(s): {missing}")


def copy_required_files(repo, stage):
    """Copy all required files to staging area."""
    for p in REQUIRED:
        copy_path(repo / p, stage / p)


def copy_optional_files(repo, stage):
    """Copy optional files if they exist."""
    for p in OPTIONAL:
        src = repo / p
        if src.exists():
            copy_path(src, stage / p)


def copy_glob_patterns(repo, stage):
    """Copy files matching glob patterns."""
    for pattern in INCLUDE_GLOBS:
        for match in glob.glob(str(repo / pattern)):
            rel = Path(match).relative_to(repo)
            copy_path(repo / rel, stage / rel)


def create_verify_file(stage, version):
    """Create VERIFY.txt with version and commit hash."""
    verify = stage / 'VERIFY.txt'
    sha = os.popen('git rev-parse HEAD').read().strip()
    verify.write_text(f"version={version}\ncommit={sha}\n", encoding='utf-8')


def create_zip_file(stage, out_zip):
    """Create ZIP file from staging directory."""
    if out_zip.exists():
        out_zip.unlink()
    with zipfile.ZipFile(out_zip, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(stage):
            for f in files:
                fp = Path(root) / f
                z.write(fp, fp.relative_to(stage))


def validate_zip_size(out_zip):
    """Validate ZIP file size and return size in MB."""
    size_mb = out_zip.stat().st_size / (1024 * 1024)
    if size_mb > MAX_ZIP_MB:
        fail(f"ZIP too large: {size_mb:.1f} MB > {MAX_ZIP_MB} MB")
    return size_mb


def calculate_sha256(out_zip, dist):
    """Calculate and save SHA256 checksum of ZIP file."""
    h = hashlib.sha256()
    with open(out_zip, 'rb') as fh:
        for chunk in iter(lambda: fh.read(8192), b''):
            h.update(chunk)
    (dist / 'ZIP_SHA256.txt').write_text(h.hexdigest())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--version', required=True, help='Release version without leading v, e.g., 1.0.1')
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    dist = repo / 'dist'
    stage = dist / 'release'
    out_zip = dist / f"toxicity-detector-v{args.version}.zip"

    prepare_staging_area(repo, dist, stage)
    verify_required_paths(repo)
    copy_required_files(repo, stage)
    copy_optional_files(repo, stage)
    copy_glob_patterns(repo, stage)
    create_verify_file(stage, args.version)
    create_zip_file(stage, out_zip)
    size_mb = validate_zip_size(out_zip)
    calculate_sha256(out_zip, dist)

    print(f"Created {out_zip} ({size_mb:.1f} MB)")


if __name__ == '__main__':
    main()
