"""Create a review archive with explicit inclusions and per-file SHA256 hashes."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import zipfile


def main():
    root = Path(__file__).resolve().parents[1]
    files = set()
    for directory in ['src', 'scripts', 'configs', 'tests', 'experiments', 'plots']:
        files.update(p for p in (root / directory).rglob('*')
                     if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc')
    for name in ['README.md', 'pyproject.toml', 'requirements.txt', 'requirements.lock.txt', '.gitignore']:
        files.add(root / name)
    files.update(p for p in (root / 'docs').glob('*.md'))
    files.update(p for p in (root / 'docs' / 'reports').iterdir()
                 if p.is_file() and p.suffix in {'.md', '.docx', '.pdf'})
    files.add(root / 'docs' / 'gas_pipeline_source.json')
    excluded = {'.aux', '.log', '.out', '.toc', '.lof', '.lot', '.blg', '.bbl', '.tmp', '.fls', '.fdb_latexmk', '.synctex.gz'}
    files.update(p for p in (root / 'docs' / 'latex').rglob('*')
                 if p.is_file() and p.suffix not in excluded
                 and 'build_logs' not in p.parts and '__pycache__' not in p.parts
                 and not p.name.endswith('.synctex.gz'))
    # The exported repository already records its complete curated file set.
    submission_manifest = root / 'SUBMISSION_MANIFEST.json'
    if submission_manifest.exists():
        snapshot = json.loads(submission_manifest.read_text())
        for row in snapshot['files']:
            relative = Path(row['path'])
            path = (root / relative).resolve()
            if relative.is_absolute() or '..' in relative.parts or not path.is_relative_to(root):
                raise ValueError(f'Invalid submission manifest path: {relative}')
            if not path.is_file():
                raise FileNotFoundError(path)
            files.add(path)
        files.add(submission_manifest)
    manifest = [{'path': str(p.relative_to(root)), 'bytes': p.stat().st_size,
                 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(files)]
    archive = root / 'docs' / 'submission_review_bundle.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(files):
            z.write(p, p.relative_to(root))
        z.writestr('BUNDLE_MANIFEST.json', json.dumps(manifest, indent=2) + '\n')
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for row in manifest:
            assert hashlib.sha256(z.read(row['path'])).hexdigest() == row['sha256']
    print(f'{archive}: {len(manifest)} files, {archive.stat().st_size:,} bytes; CRC and SHA256 verified')


if __name__ == '__main__':
    main()
