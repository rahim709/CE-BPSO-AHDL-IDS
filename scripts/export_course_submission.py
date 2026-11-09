"""Export separate GitHub and self-contained Overleaf course submissions."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import zipfile


PROJECT_NAME = 'CE-BPSO-AHDL-IDS'

STUDIES = [
    'false_alarm_optimization', 'nmri_tradeoff', 'submission_final_evaluation',
    'research_extension', 'state_grouped_validation', 'ua_amf_prediction_20261108',
    'ua_amf_prediction_benchmark_20261108', 'ua_amf_prediction_final_20261108',
    'ua_amf_efficiency_20261108', 'ua_amf_online_20261108',
]
SKIP_PARTS = {'__pycache__', 'source_snapshot', 'manuscript_before', 'manuscript_final',
              'build_logs', 'execution_logs', 'matplotlib-cache', 'original_trace'}
SKIP_EVIDENCE = {'inventory.json', 'archive_verification.json'}
DOCS = ['datasets.md', 'gas_pipeline_source.json', 'paper_specification.md', 'protocol.md',
        'false_alarm_optimization.md', 'nmri_balance.md', 'ua_amf_implementation.md',
        'ua_amf_prediction_study.md', 'efficiency_optimization.md', 'archive_free_ua_amf.md',
        'current_execution_review.md', 'latest_gpu_verification.md', 'gpu_validation.md',
        'implementation_gap_review.md', 'research_diagnostics.md', 'submission_review.md',
        'validation_report.md', 'command_validation.md', 'synthetic_dataset_validation.md',
        'results_review.md']


def copy(source: Path, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def sha256(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def package_overleaf(output: Path):
    archive = output / (PROJECT_NAME + '-Overleaf.zip')
    members = [(p, str(p.relative_to(output / 'overleaf')))
               for p in sorted((output / 'overleaf').rglob('*')) if p.is_file()]
    paper = output / (PROJECT_NAME + '.pdf')
    members.append((paper, paper.name))
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for path, name in members:
            z.write(path, name)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for path, name in members:
            assert hashlib.sha256(z.read(name)).hexdigest() == sha256(path)
    return archive


def write_manifest(output: Path):
    github = output / PROJECT_NAME
    rows = []
    for path in sorted(github.rglob('*')):
        if not path.is_file() or set(path.relative_to(github).parts) & {'.git', '__pycache__', '.pytest_cache'}:
            continue
        if path.name == 'SUBMISSION_MANIFEST.json':
            continue
        rows.append({'path': str(path.relative_to(github)), 'bytes': path.stat().st_size,
                     'sha256': sha256(path)})
    overleaf = [{'path': str(p.relative_to(output)), 'sha256': sha256(p)}
                for p in sorted((output / 'overleaf').rglob('*')) if p.is_file()]
    manifest = {'files': rows, 'overleaf': overleaf, 'paper_sha256': sha256(output / (PROJECT_NAME + '.pdf'))}
    archive = output / (PROJECT_NAME + '-Overleaf.zip')
    if archive.exists():
        manifest['overleaf_zip'] = {'path': archive.name, 'sha256': sha256(archive)}
    (github / 'SUBMISSION_MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')


def export_overleaf(root: Path, output: Path):
    latex = root / 'docs/latex'
    images = output / 'images'
    images.mkdir(parents=True, exist_ok=True)
    active = set()
    image_names = {}

    def resolve(name, suffix):
        path = latex / name
        if not path.suffix:
            path = path.with_suffix(suffix)
        path = path.resolve()
        if not path.is_relative_to(root):
            raise ValueError(f'Document dependency is outside the project: {name}')
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def flatten(path):
        if path in active:
            raise ValueError(f'Cyclic TeX input: {path}')
        active.add(path)
        text = path.read_text()

        def include(match):
            source = resolve(match[1], '.tex')
            return '\n' + flatten(source) + '\n'

        text = re.sub(r'\\(?:input|include)\{([^}]+)\}', include, text)

        def graphic(match):
            name = match[2]
            if name.startswith('images/') and (output / name).is_file():
                return match[0]
            direct = latex / name
            source = direct if direct.is_file() else latex / 'figures' / name
            source = source.resolve()
            if not source.is_relative_to(root) or not source.is_file():
                raise FileNotFoundError(source)
            if source not in image_names:
                target_name = source.name
                if (images / target_name).exists() and sha256(images / target_name) != sha256(source):
                    target_name = source.stem + '_' + sha256(source)[:8] + source.suffix
                copy(source, images / target_name)
                image_names[source] = target_name
            return r'\includegraphics' + (match[1] or '') + '{images/' + image_names[source] + '}'

        text = re.sub(r'\\includegraphics(\[[^]]*\])?\{([^}]+)\}', graphic, text)
        active.remove(path)
        return text

    text = flatten((latex / 'manuscript.tex').resolve())
    text = text.replace(r'\graphicspath{{figures/}}', r'\graphicspath{{images/}}')
    (output / 'main.tex').write_text(text)
    copy(latex / 'references.bib', output / 'references.bib')
    # Bundle the IEEE style and its original license notices for portable builds.
    for name in ['IEEEtran.cls', 'IEEEtran.bst', 'README']:
        copy(latex / name, output / ('README.IEEEtran' if name == 'README' else name))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('course_submission'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f'Output already exists; choose a new directory: {output}')
    github = output / PROJECT_NAME
    github.mkdir(parents=True)

    for directory in ['src', 'scripts', 'configs', 'tests', 'experiments', 'plots']:
        for source in (root / directory).rglob('*'):
            if source.is_file() and not set(source.parts) & SKIP_PARTS and source.suffix != '.pyc':
                copy(source, github / source.relative_to(root))
    for name in ['README.md', 'pyproject.toml', 'requirements.txt', 'requirements.lock.txt']:
        copy(root / name, github / name)
    for name in DOCS:
        copy(root / 'docs' / name, github / 'docs' / name)
    for source in (root / 'docs/reports').glob('*.md'):
        copy(source, github / source.relative_to(root))
    allowed_latex = {'.tex', '.bib', '.cls', '.bst', '.pdf', '.png', '.json', '.csv', '.xml', '.npz', '.md'}
    for source in (root / 'docs/latex').rglob('*'):
        if not source.is_file() or set(source.parts) & SKIP_PARTS:
            continue
        if source.name.startswith('ua_amf_manuscript') or source.name in SKIP_EVIDENCE:
            continue
        if source.suffix in allowed_latex or source.name == 'README':
            copy(source, github / source.relative_to(root))

    allowed_results = {'.json', '.csv', '.md', '.tex', '.pdf', '.png', '.xml'}
    for study in STUDIES:
        for source in (root / 'results' / study).rglob('*'):
            if not source.is_file() or set(source.parts) & SKIP_PARTS:
                continue
            if source.name in {'artifact_manifest.json', 'files_created_or_modified.json'}:
                continue
            prediction = source.suffix == '.npz' and ('prediction' in source.name or source.name.startswith('validation_epoch_'))
            if source.suffix in allowed_results or prediction:
                copy(source, github / source.relative_to(root))
    for source in (root / 'results/learning_sanity/final_verification').rglob('*'):
        if source.is_file() and (source.suffix in allowed_results or 'predictions' in source.name):
            copy(source, github / source.relative_to(root))
    # Only metadata is exported for data views; raw and processed feature arrays stay local.
    for path in ['datasets/raw/.gitkeep', 'datasets/processed/.gitkeep']:
        copy(root / path, github / path)
    (github / '.gitignore').write_text('''# Local environments, credentials and caches
.venv/
.venv-*/
venv/
__pycache__/
*.py[cod]
.pytest_cache/
.mypy_cache/
.ruff_cache/
.coverage
htmlcov/
*.egg-info/
build/
dist/
.env
.env.*
!.env.example
.aws/
.codex/
.agents/
.idea/
.vscode/
.DS_Store

# Obtain official data separately; preserve the included measured results.
datasets/raw/*
datasets/processed/*
!datasets/raw/.gitkeep
!datasets/processed/.gitkeep
datasets/synthetic/
results/**/train.npz
results/**/val.npz
results/**/test.npz
*.pt
*.pth
*.ckpt
*.joblib
*.pkl

# Document intermediates and generated package copies
*.aux
*.log
*.out
*.toc
*.lof
*.lot
*.blg
*.bbl
*.fls
*.fdb_latexmk
*.synctex.gz
docs/latex/build_logs/
docs/submission_review_bundle.zip
course_submission/
course_submission.zip
''')
    export_overleaf(root, output / 'overleaf')
    with tempfile.TemporaryDirectory(prefix='bpso-overleaf-build-') as directory:
        build = Path(directory)
        shutil.copytree(output / 'overleaf', build, dirs_exist_ok=True)
        for command in [['pdflatex', '-interaction=nonstopmode', '-halt-on-error', 'main.tex'],
                        ['bibtex', 'main'],
                        ['pdflatex', '-interaction=nonstopmode', '-halt-on-error', 'main.tex'],
                        ['pdflatex', '-interaction=nonstopmode', '-halt-on-error', 'main.tex']]:
            result = subprocess.run(command, cwd=build, capture_output=True, text=True)
            if result.returncode:
                raise RuntimeError(result.stdout[-6000:] + result.stderr)
        log = (build / 'main.log').read_text()
        if 'undefined references' in log or 'Rerun to get cross-references right' in log:
            raise RuntimeError('Unresolved references in the Overleaf project')
        copy(build / 'main.pdf', output / (PROJECT_NAME + '.pdf'))
        copy(build / 'main.pdf', github / (PROJECT_NAME + '.pdf'))
        copy(build / 'main.pdf', github / 'docs/latex/manuscript.pdf')
        verification = github / 'results/verification'
        verification.mkdir(parents=True, exist_ok=True)
        (verification / 'overleaf_build.json').write_text(json.dumps({
            'compiler': 'pdfLaTeX + BibTeX + two pdfLaTeX passes',
            'unresolved_references': False, 'paper_sha256': sha256(output / (PROJECT_NAME + '.pdf')),
            'source': 'overleaf/main.tex', 'bibliography': 'overleaf/references.bib',
        }, indent=2) + '\n')
    subprocess.run(['git', 'init', '-b', 'main', str(github)], check=True, capture_output=True)
    package_overleaf(output)
    write_manifest(output)
    print(f'Exported to {output}')


if __name__ == '__main__':
    main()
