"""Integrity manifest for the frozen V3.3 project (read-only).

Usage (from the project root):
    python -m visuograsp_v4.scripts.integrity_manifest snapshot   # write baseline manifest
    python -m visuograsp_v4.scripts.integrity_manifest verify     # compare against baseline

Every file outside ``visuograsp_v4/`` and ``.git/`` is covered. Files up to 64 MB are
hashed with SHA-256. Larger files (the CV datasets, about 4.3 GB in total) get a
size + mtime + partial hash (first and last MiB), which detects any rewrite in practice
without reading gigabytes on every check. Nothing in the V3.3 tree is ever written.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
V4_ROOT = PROJECT_ROOT / 'visuograsp_v4'
MANIFEST_DIR = V4_ROOT / 'outputs' / 'integrity'
BASELINE = MANIFEST_DIR / 'v33_manifest_baseline.json'
FULL_HASH_LIMIT = 64 * 1024 * 1024
EXCLUDE_TOP = {'.git', 'visuograsp_v4'}


def _file_record(path: Path) -> dict:
    stat = path.stat()
    record = {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        if stat.st_size <= FULL_HASH_LIMIT:
            for block in iter(lambda: handle.read(1 << 20), b''):
                digest.update(block)
            record['sha256'] = digest.hexdigest()
        else:
            digest.update(handle.read(1 << 20))
            handle.seek(max(0, stat.st_size - (1 << 20)))
            digest.update(handle.read(1 << 20))
            record['partial_sha256'] = digest.hexdigest()
    return record


def build_manifest() -> dict:
    files = {}
    for path in sorted(PROJECT_ROOT.rglob('*')):
        rel = path.relative_to(PROJECT_ROOT)
        if rel.parts[0] in EXCLUDE_TOP or not path.is_file():
            continue
        files[rel.as_posix()] = _file_record(path)
    return {'created_unix': time.time(), 'root': str(PROJECT_ROOT), 'n_files': len(files),
            'total_bytes': sum(r['size'] for r in files.values()), 'files': files}


def compare(baseline: dict, current: dict) -> dict:
    base, cur = baseline['files'], current['files']
    missing = sorted(set(base) - set(cur))
    added = sorted(set(cur) - set(base))
    changed = []
    for name in sorted(set(base) & set(cur)):
        a, b = base[name], cur[name]
        keys = ('size', 'sha256', 'partial_sha256')
        if any(a.get(k) != b.get(k) for k in keys):
            changed.append(name)
    return {'missing': missing, 'added_outside_v4': added, 'changed': changed,
            'n_baseline': len(base), 'n_current': len(cur),
            'untouched': not (missing or changed)}


def main(argv: list[str]) -> int:
    mode = argv[1] if len(argv) > 1 else 'verify'
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    started = time.time()
    current = build_manifest()
    if mode == 'snapshot':
        if BASELINE.exists():
            print(f'[integrity] baseline already exists, not overwritten: {BASELINE}')
            return 0
        BASELINE.write_text(json.dumps(current, indent=1), encoding='utf-8')
        print(f'[integrity] baseline written: {current["n_files"]} files, '
              f'{current["total_bytes"] / 1e9:.2f} GB, {time.time() - started:.1f} s')
        return 0
    baseline = json.loads(BASELINE.read_text(encoding='utf-8'))
    result = compare(baseline, current)
    out = MANIFEST_DIR / 'v33_verify_latest.json'
    out.write_text(json.dumps(result, indent=1), encoding='utf-8')
    print(f'[integrity] untouched={result["untouched"]} files={result["n_current"]} '
          f'changed={len(result["changed"])} missing={len(result["missing"])} '
          f'added_outside_v4={len(result["added_outside_v4"])} ({time.time() - started:.1f} s)')
    return 0 if result['untouched'] else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv))
