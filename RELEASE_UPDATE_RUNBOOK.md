# Release Update Runbook

Quick reference for managing GitHub releases with the single-ZIP packaging system.

## Automated Release (Recommended)

### Creating a New Release

When you push a tag, the release workflow automatically:
1. Packages everything into `toxicity-detector-v<version>.zip`
2. Uploads the ZIP to the release
3. Cleans up any old assets (keeps only the ZIP + GitHub's source archives)

**Steps:**
```bash
# 1. Ensure all changes are committed
git add .
git commit -m "Your commit message"

# 2. Create and push a tag
git tag v1.0.1
git push origin v1.0.1
```

The `.github/workflows/release-single-zip.yml` workflow will automatically:
- Extract version from tag (e.g., `v1.0.1` → `1.0.1`)
- Run `tools/pack_release.py` to create the ZIP
- Upload `toxicity-detector-v1.0.1.zip` to the release
- Delete any old assets (except GitHub's source archives)

### What Gets Packaged

The `tools/pack_release.py` script includes:
- **Required**: `app/`, `src/`, `requirements.txt`, `README.md`
- **Optional** (if present): `models/`, processed data CSVs, `scripts/`, documentation files, `LICENSE`
- **Verification**: Creates `VERIFY.txt` with version and commit hash

## Manual Release Update

### Option A: Update Existing Release with New ZIP

If you need to update an existing release:

**1. Create new ZIP locally:**
```bash
python3 tools/pack_release.py --version 1.0.0
# Creates: dist/toxicity-detector-v1.0.0.zip
```

**2. Upload to release:**
```bash
gh release upload v1.0.0 dist/toxicity-detector-v1.0.0.zip --clobber
```

**3. Clean up old assets:**
```bash
# List current assets
gh release view v1.0.0 --json assets -q '.assets[] | .name'

# Delete old assets (keep only the ZIP and GitHub source archives)
gh api repos/:owner/:repo/releases/tags/v1.0.0 --jq '.assets[] | select(.name != "toxicity-detector-v1.0.0.zip" and .name != "Source code (zip)" and .name != "Source code (tar.gz)") | .id' | while read id; do gh api -X DELETE "repos/:owner/:repo/releases/assets/$id"; done
```

### Option B: GitHub Actions Workflow (Legacy)

For updating with individual files (not recommended, use single ZIP instead):

1. Go to: **Actions** → **Update Existing Release** → **Run workflow**
2. Fill in:
   - **release_tag**: `v1.0.0`
   - **files_glob**: `models/*.pth data/processed/*.csv`
   - **release_body**: `Updated models and datasets`
   - **overwrite**: `true`

## Release Checklist

- [ ] All code changes committed and pushed
- [ ] Tests pass: `make test`
- [ ] Linting passes: `make lint`
- [ ] Version number updated (if needed)
- [ ] Create and push tag: `git tag v<version> && git push origin v<version>`
- [ ] Verify release on GitHub
- [ ] Verify ZIP contains all necessary files
- [ ] Test downloading and extracting the ZIP

## Package Contents Verification

After creating a release, verify the ZIP contains:
- ✅ `app/` directory with Streamlit app
- ✅ `src/` directory with all source code
- ✅ `models/` directory (if models are included)
- ✅ `data/processed/` directory with CSV files
- ✅ `requirements.txt`
- ✅ `README.md`, `RUN_INSTRUCTIONS.md`, `REVIEWER_GUIDE.md`
- ✅ `VERIFY.txt` with version and commit hash

## Troubleshooting

**ZIP too large?**
- The script enforces a 2.5GB limit
- Check `models/` directory size
- Consider excluding large evaluation results

**Missing files in ZIP?**
- Check `tools/pack_release.py` REQUIRED and OPTIONAL lists
- Verify files exist in repository before packaging

**Workflow fails?**
- Check GitHub Actions logs
- Verify tag format: `v*` (e.g., `v1.0.1`)
- Ensure `tools/pack_release.py` is executable and in repository

