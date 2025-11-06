# Contributing to Toxicity Detector

Thank you for your interest in contributing! This document provides guidelines for maintaining code quality and consistency.

## Code Style

### Comments

**Comments should explain WHY, not WHAT.**

#### ✅ Good Comments

- Explain non-obvious intent, business rules, or domain constraints
- Document tricky algorithms, edge cases, or performance trade-offs
- Security, privacy, and compliance notes (e.g., PII handling)
- Configuration comments that tools rely on (e.g., `# noqa: E501` with reason)
- License headers and file-level legal notices

```python
# Compatibility layer: maps old module path for pickle compatibility
sys.modules['dataset_utils'] = dataset_utils

# Heuristic mapping: hate speech -> severe_toxic + identity_hate
# Real annotations would improve performance significantly
LABEL_MAPPING = {...}
```

#### ❌ Bad Comments (Remove These)

- Obvious statements: `# Load model`, `# Loop through items`, `# Create variable`
- Redundant explanations that the code already makes clear
- Duplications of function/class names right above definitions
- Section dividers like `# ====== UI ======`
- Comments that paraphrase the next 2-3 lines verbatim

```python
# ❌ BAD
# Load model
model = load_model()

# ❌ BAD
# This function loads the model
def load_model():
    ...

# ✅ GOOD
# Lazy loading to reduce memory footprint on startup
def load_model():
    ...
```

### TODOs

- Include owner and issue link: `# TODO(@username): #123 - reason`
- Remove TODOs with no owner or timeline unless tracked in issues
- Convert vague TODOs to actionable items

```python
# ❌ BAD
# TODO: fix later

# ✅ GOOD
# TODO(@alice): #456 - Refactor moderation thresholds into config file
```

### Docstrings

- Use consistent style (Google or NumPy) based on the file's existing style
- Keep docstrings for public APIs
- Focus on parameters, return values, and exceptions
- Remove docstrings that only restate the function name

### Infrastructure Files

- **GitHub Actions, Dockerfile, Makefile**: Keep concise comments explaining build matrix, caching, or deployment caveats
- Remove obvious comments that don't add value

## Formatting

- Follow the repository's linter settings (max-line-length = 127 per flake8 config)
- Run `make lint` or `flake8 .` before committing
- Ensure no trailing whitespace

## Testing

- Add tests for new features
- Keep test comments minimal; only explain non-obvious setup/fixtures or invariants
- Remove obvious test comments like `# Test that X works`

## Pull Requests

- Keep PRs focused and well-documented
- If removing commented-out code, link to commit hash in PR description if critical
- Ensure all linters pass before requesting review

## Questions?

If you're unsure whether a comment adds value, ask yourself:
1. Does this explain **why** something is done, not just **what**?
2. Would a new contributor be confused without this comment?
3. Does this document a non-obvious constraint or trade-off?

If the answer is "no" to all three, the comment can likely be removed.

