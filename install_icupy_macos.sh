#!/bin/sh
# (Re)build PyICU against Homebrew's icu4c on macOS.
#
# The build environment (PKG_CONFIG_PATH etc. pointing at the keg-only
# icu4c) lives in pyproject.toml under [tool.uv.extra-build-variables], so a
# plain `uv sync` already builds correctly. This script exists to force a
# rebuild when the cached wheel is linked against an ICU that no longer
# exists, e.g. after `brew upgrade` moved icu4c to a new major version
# (symptom: `import icu` fails with "Library not loaded:
# @rpath/libicui18n.NN.dylib" and trom warns about falling back to uconv).
set -ex

brew list icu4c >/dev/null

uv cache clean pyicu
uv sync --reinstall-package pyicu

# Verify the fresh build imports and report the linked ICU version
uv run python -c "import icu; print('PyICU OK, ICU version:', icu.ICU_VERSION)"
