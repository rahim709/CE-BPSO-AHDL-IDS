"""Read-only repository/evidence inventory; does not certify historic measurements."""
import hashlib
import json
from pathlib import Path
from collections import Counter
from src.optimization.encoding import BinaryCodec
from src.utils.core import save_json


def main():
    rows, summaries, ledgers, errors = [], [], [], []
    for root in ['src', 'configs', 'tests', 'scripts', 'datasets', 'results', 'experiments', 'docs']:
        for path in sorted(Path(root).rglob('*')):
            if not path.is_file() or '__pycache__' in path.parts or 'ua_amf' in str(path):
                continue
            rows.append({'path': str(path), 'bytes': path.stat().st_size})
            if path.name == 'search_summary.json':
                try:
                    s = json.loads(path.read_text())
                    summaries.append({'path': str(path), 'method': s.get('method'), 'seed': s.get('seed'),
                                      'epochs': s.get('charged_epochs'), 'selection_complete': s.get('selection_complete'),
                                      'fitness': (s.get('best') or {}).get('fitness'), 'synthetic': s.get('synthetic_data')})
                except Exception as exc:
                    errors.append({'path': str(path), 'error': str(exc)})
            if path.name == 'evaluations.json':
                try:
                    events = json.loads(path.read_text())
                    ledgers.append({'path': str(path), 'events': len(events), 'charged_epochs': sum(e['charged_epochs'] for e in events),
                                    'failures': sum(bool(e.get('failure')) for e in events)})
                except Exception as exc:
                    errors.append({'path': str(path), 'error': str(exc)})
    paper = next(Path('.').glob('BPSO*.pdf'))
    save_json('results/ua_amf_validation/repository_audit.json', {'files': rows, 'historical_search_summaries': summaries,
              'historical_ledgers': ledgers, 'parse_errors': errors, 'paper_sha256': hashlib.sha256(paper.read_bytes()).hexdigest(),
              'paper_codec_bits': BinaryCodec('paper').length,
              'suffix_counts': dict(Counter(Path(r['path']).suffix for r in rows)),
              'scope': 'Structural audit, not a rerun of historic full experiments. Historical artifacts not modified.'})


if __name__ == '__main__':
    main()
