"""Fetch the NASA C-MAPSS turbofan degradation dataset.

Tries a list of stable mirrors (the original NASA PCoE link rots periodically).
Validates by checking that the expected ``train_FD00{1..4}.txt`` files appear
after extraction. If every mirror fails, prints manual instructions.
"""
from __future__ import annotations
import io
import sys
import zipfile
from pathlib import Path

import requests

from ml.config import CMAPSS_DIR

# Known stable mirrors of the NASA C-MAPSS turbofan dataset.
# Order matters; we try them top-down.
MIRRORS = [
    # Official NASA PHM data repository (AWS S3, ~5.7 MB zip)
    'https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip',
]
# Per-file fallback if zip mirrors fail.
PER_FILE_BASE = 'https://raw.githubusercontent.com/hankroark/Turbofan-Engine-Degradation/master/CMAPSSData'

EXPECTED = [f'{kind}_FD00{i}.txt' for kind in ('train', 'test', 'RUL') for i in (1, 2, 3, 4)]


def _is_complete(target: Path) -> bool:
    return all((target / name).exists() for name in EXPECTED)


def _try_download(url: str, target: Path) -> bool:
    print(f'  → trying {url}')
    try:
        r = requests.get(url, stream=True, timeout=60)
        r.raise_for_status()
        buf = io.BytesIO(r.content)
        with zipfile.ZipFile(buf) as zf:
            # Some zips nest under CMaps/, some don't — extract files flat.
            for member in zf.namelist():
                base = Path(member).name
                if base in EXPECTED:
                    with zf.open(member) as src, (target / base).open('wb') as dst:
                        dst.write(src.read())
        return _is_complete(target)
    except Exception as e:  # noqa: BLE001 — try next mirror
        print(f'    ✗ {type(e).__name__}: {e}')
        return False


def main() -> int:
    target = CMAPSS_DIR / 'CMaps'
    target.mkdir(parents=True, exist_ok=True)

    if _is_complete(target):
        print(f'✓ C-MAPSS already present at {target}')
        return 0

    print('Fetching C-MAPSS...')
    for url in MIRRORS:
        if _try_download(url, target):
            print(f'✓ Extracted to {target}')
            return 0

    print('  → falling back to per-file download from hankroark mirror')
    ok = True
    for name in EXPECTED:
        url = f'{PER_FILE_BASE}/{name}'
        try:
            r = requests.get(url, timeout=30); r.raise_for_status()
            (target / name).write_bytes(r.content)
        except Exception as e:  # noqa: BLE001
            print(f'    ✗ {name}: {e}'); ok = False
    if ok and _is_complete(target):
        print(f'✓ Per-file download complete: {target}')
        return 0

    print('\n✗ All mirrors failed.')
    print('Manual fallback:')
    print('  1. Download the C-MAPSS / Turbofan Engine Degradation Simulation dataset')
    print('     from NASA PCoE (https://www.nasa.gov/intelligent-systems-division/).')
    print(f'  2. Extract train_FD00{{1..4}}.txt, test_FD00{{1..4}}.txt, RUL_FD00{{1..4}}.txt')
    print(f'     into: {target}')
    return 1


if __name__ == '__main__':
    sys.exit(main())
