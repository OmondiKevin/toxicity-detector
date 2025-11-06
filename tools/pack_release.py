#!/usr/bin/env python3
import argparse, os, shutil, sys, glob, hashlib, zipfile
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--version', required=True, help='Release version without leading v, e.g., 1.0.1')
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    dist = repo / 'dist'
    stage = dist / 'release'
    out_zip = dist / f"toxicity-detector-v{args.version}.zip"

    # Clean staging area
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True, exist_ok=True)
    dist.mkdir(parents=True, exist_ok=True)

    # Verify required paths
    missing = [p for p in REQUIRED if not (repo / p).exists()]
    if missing:
        fail(f"Missing required path(s): {missing}")

    # Copy required
    for p in REQUIRED:
        copy_path(repo / p, stage / p)

    # Copy optional if present
    for p in OPTIONAL:
        src = repo / p
        if src.exists():
            copy_path(src, stage / p)

    # Include model/config globs if present
    for pattern in INCLUDE_GLOBS:
        for match in glob.glob(str(repo / pattern)):
            rel = Path(match).relative_to(repo)
            copy_path(repo / rel, stage / rel)

    # Write VERIFY.txt
    verify = stage / 'VERIFY.txt'
    sha = os.popen('git rev-parse HEAD').read().strip()
    verify.write_text(f"version={args.version}\ncommit={sha}\n", encoding='utf-8')

    # Create ZIP
    if out_zip.exists():
        out_zip.unlink()
    with zipfile.ZipFile(out_zip, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(stage):
            for f in files:
                fp = Path(root) / f
                z.write(fp, fp.relative_to(stage))

    # Size guard
    size_mb = out_zip.stat().st_size / (1024 * 1024)
    if size_mb > MAX_ZIP_MB:
        fail(f"ZIP too large: {size_mb:.1f} MB > {MAX_ZIP_MB} MB")

    # SHA256 of ZIP
    h = hashlib.sha256()
    with open(out_zip, 'rb') as fh:
        for chunk in iter(lambda: fh.read(8192), b''):
            h.update(chunk)
    (dist / 'ZIP_SHA256.txt').write_text(h.hexdigest())

    print(f"Created {out_zip} ({size_mb:.1f} MB)")

if __name__ == '__main__':
    main()

