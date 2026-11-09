"""Compile the review report and IEEEtran manuscript; retain complete logs."""
from pathlib import Path
import subprocess


def main():
    root = Path(__file__).resolve().parents[1] / 'docs' / 'latex'
    logs = root / 'build_logs'
    logs.mkdir(exist_ok=True)
    for name in ['implementation_report', 'manuscript']:
        for index, command in enumerate([
            ['pdflatex', '-interaction=nonstopmode', '-halt-on-error', name + '.tex'],
            ['bibtex', name],
            ['pdflatex', '-interaction=nonstopmode', '-halt-on-error', name + '.tex'],
            ['pdflatex', '-interaction=nonstopmode', '-halt-on-error', name + '.tex'],
        ]):
            result = subprocess.run(command, cwd=root, capture_output=True, text=True)
            (logs / f'{name}_{index}.txt').write_text(result.stdout + result.stderr)
            if result.returncode:
                print(result.stdout[-5000:] + result.stderr)
                raise SystemExit(result.returncode)
        log = (root / (name + '.log')).read_text()
        if 'undefined references' in log or 'Citation `' in log or 'Rerun to get cross-references right' in log:
            raise SystemExit(f'Unresolved references in {name}; inspect the build log.')
        print(f'Compiled {root / (name + ".pdf")}')


if __name__ == '__main__':
    main()
