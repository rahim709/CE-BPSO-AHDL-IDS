"""Reproducible, schema-inspired software fixtures. Never original research data.

These simple stochastic generators are not fitted to the original corpora and
make no claim to reproduce their joint distributions, attack physics or labels.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from src.utils.core import save_json

TON_CLASSES = ('normal', 'backdoor', 'ddos', 'dos', 'injection', 'mitm', 'password', 'ransomware', 'scanning', 'xss')
BOT_CLASSES = ('Normal', 'DDoS', 'DoS', 'Reconnaissance', 'Theft')
SWAT_SENSORS = (
    'FIT101', 'LIT101', 'MV101', 'P101', 'P102',
    'AIT201', 'AIT202', 'AIT203', 'FIT201', 'MV201', 'P201', 'P202', 'P203', 'P204', 'P205', 'P206',
    'DPIT301', 'FIT301', 'LIT301', 'MV301', 'MV302', 'MV303', 'MV304', 'P301', 'P302',
    'AIT401', 'AIT402', 'FIT401', 'LIT401', 'P401', 'P402', 'P403', 'P404', 'UV401',
    'AIT501', 'AIT502', 'AIT503', 'AIT504', 'FIT501', 'FIT502', 'FIT503', 'FIT504',
    'P501', 'P502', 'PIT501', 'PIT502', 'PIT503', 'FIT601', 'P601', 'P602', 'P603')


def labels(rng: np.random.Generator, n: int, classes: tuple[str, ...], probabilities: list[float]) -> np.ndarray:
    # Guaranteed class coverage is a fixture-design choice, not real prevalence.
    minimum = 8
    if n < minimum * len(classes):
        raise ValueError('Too few records to guarantee class coverage in this fixture')
    counts = rng.multinomial(n - minimum * len(classes), np.asarray(probabilities) / sum(probabilities)) + minimum
    values = np.repeat(np.asarray(classes), counts)
    rng.shuffle(values)
    return values


def _missing(frame: pd.DataFrame, rng: np.random.Generator, columns: list[str]) -> None:
    for col in columns:
        mask = rng.random(len(frame)) < .002
        if mask.any():
            frame[col] = frame[col].astype(float)
            frame.loc[mask, col] = np.nan


def ton_network(n: int, rng: np.random.Generator, offset: int) -> pd.DataFrame:
    y = labels(rng, n, TON_CLASSES, [.65, .04, .08, .06, .04, .003, .04, .02, .04, .027])
    attack = (y != 'normal').astype(int)
    load = rng.lognormal(.3 + attack * .35, .9)
    duration = rng.lognormal(-1 + attack * .15, 1.2)
    src_pkts = rng.poisson(12 * load) + 1
    dst_pkts = rng.poisson(8 * load) + 1
    src_bytes = src_pkts * rng.integers(40, 1400, n)
    dst_bytes = dst_pkts * rng.integers(40, 1400, n)
    frame = pd.DataFrame({
        'src_ip': [f'192.0.2.{1 + (i + offset) % 200}' for i in range(n)],
        'src_port': rng.integers(1024, 65536, n),
        'dst_ip': [f'198.51.100.{1 + (i + offset) % 200}' for i in range(n)],
        'dst_port': rng.choice([22, 53, 80, 443, 502, 1883], n),
        'proto': rng.choice(['tcp', 'udp', 'icmp'], n, p=[.75, .23, .02]),
        'service': rng.choice(['-', 'dns', 'http', 'ssl', 'ssh'], n),
        'duration': duration, 'src_bytes': src_bytes, 'dst_bytes': dst_bytes,
        'conn_state': rng.choice(['SF', 'S0', 'REJ', 'OTH'], n),
        'missed_bytes': rng.poisson(.3 + attack, n),
        'src_pkts': src_pkts, 'src_ip_bytes': src_bytes + 40 * src_pkts,
        'dst_pkts': dst_pkts, 'dst_ip_bytes': dst_bytes + 40 * dst_pkts,
        'dns_query': rng.choice(['-', 'sensor.example', 'gateway.example'], n),
        'dns_qclass': rng.choice([0, 1], n), 'dns_qtype': rng.choice([0, 1, 28], n),
        'dns_rcode': rng.choice([0, 2, 3], n),
        'dns_AA': rng.choice(['-', 'T', 'F'], n), 'dns_RD': rng.choice(['-', 'T', 'F'], n),
        'dns_RA': rng.choice(['-', 'T', 'F'], n), 'dns_rejected': rng.choice(['-', 'T', 'F'], n),
        'ssl_version': rng.choice(['-', 'TLSv12', 'TLSv13'], n),
        'ssl_cipher': rng.choice(['-', 'TLS_AES_128_GCM_SHA256'], n),
        'ssl_resumed': rng.choice(['-', 'T', 'F'], n), 'ssl_established': rng.choice(['-', 'T', 'F'], n),
        'ssl_subject': rng.choice(['-', 'CN=synthetic.example'], n),
        'ssl_issuer': rng.choice(['-', 'CN=Synthetic Test CA'], n),
        'http_trans_depth': rng.integers(0, 5, n), 'http_method': rng.choice(['-', 'GET', 'POST'], n),
        'http_uri': rng.choice(['-', '/', '/status', '/api/sensor'], n),
        'http_version': rng.choice(['-', '1.1', '2.0'], n),
        'http_request_body_len': rng.poisson(80 + 35 * attack, n),
        'http_response_body_len': rng.poisson(500 + 60 * attack, n),
        'http_status_code': rng.choice([0, 200, 404, 500], n),
        'http_user_agent': rng.choice(['-', 'synthetic-client/1.0'], n),
        'http_orig_mime_types': rng.choice(['-', 'application/json'], n),
        'http_resp_mime_types': rng.choice(['-', 'text/html', 'application/json'], n),
        'weird_name': rng.choice(['-', 'bad_TCP_checksum'], n),
        'weird_addl': rng.choice(['-', 'synthetic'], n),
        'weird_notice': rng.choice(['-', 'T', 'F'], n), 'label': attack, 'type': y,
    })
    _missing(frame, rng, ['duration', 'src_bytes', 'dst_bytes'])
    return frame


def bot_network(n: int, rng: np.random.Generator, offset: int) -> pd.DataFrame:
    y = labels(rng, n, BOT_CLASSES, [.08, .40, .34, .16, .02])
    attack = (y != 'Normal').astype(int)
    flood = np.isin(y, ['DDoS', 'DoS']).astype(int)
    load = rng.lognormal(1.5 + .7 * flood, .85)
    spkts = rng.poisson(8 * load) + 1
    dpkts = rng.poisson(3 * load) + 1
    sbytes = spkts * rng.integers(40, 1100, n)
    dbytes = dpkts * rng.integers(40, 1100, n)
    duration = rng.lognormal(-.4 + .3 * flood, 1.1)
    ids = offset + np.arange(n)
    frame = pd.DataFrame({
        'pkSeqID': ids, 'seq': ids,
        'stime': 1735689600 + ids, 'ltime': 1735689600 + ids + duration,
        'saddr': [f'192.0.2.{1 + i % 200}' for i in ids],
        'daddr': [f'198.51.100.{1 + i % 200}' for i in ids],
        'sport': rng.integers(1024, 65536, n), 'dport': rng.choice([22, 53, 80, 443, 1883], n),
        'proto': rng.choice(['tcp', 'udp', 'icmp'], n), 'state': rng.choice(['CON', 'INT', 'FIN'], n),
        'pkts': spkts + dpkts, 'bytes': sbytes + dbytes, 'dur': duration,
        'spkts': spkts, 'dpkts': dpkts, 'sbytes': sbytes, 'dbytes': dbytes,
        'rate': (spkts + dpkts) / duration, 'srate': spkts / duration, 'drate': dpkts / duration,
        'attack': attack, 'category': y,
        'subcategory': np.where(flood, 'synthetic_flood', np.where(attack, 'synthetic_other', 'Normal')),
    })
    _missing(frame, rng, ['dur', 'rate', 'sbytes'])
    return frame


def swat_process(n: int, rng: np.random.Generator) -> pd.DataFrame:
    # Repeated blocks keep both labels in each chronological software-test split.
    attack = ((np.arange(n) // 48) % 4 == 3)
    state = np.zeros((n, 6))
    state[0] = rng.normal(0, .3, 6)
    noise = rng.normal(0, .12, (n, 6))
    for i in range(1, n):
        state[i] = .97 * state[i - 1] + noise[i]
    t = np.arange(n)
    measurements = {}
    for col in SWAT_SENSORS:
        stage = min(5, int(col[-3]) - 1)
        flow = np.sin(t / (40 + stage * 7)) + state[:, stage]
        if col.startswith(('MV', 'P', 'UV')) and not col.startswith('PIT'):
            values = (flow > -.2).astype(float) + 1
        elif col.startswith('LIT'):
            values = 55 + 12 * flow + 3 * rng.normal(size=n)
        elif col.startswith('AIT'):
            values = 7 + .5 * flow + .15 * rng.normal(size=n)
        else:
            values = 2.5 + .7 * flow + .1 * rng.normal(size=n)
        if col in {'LIT101', 'FIT201', 'AIT402', 'PIT501'}:
            values = values + attack * rng.normal(1.5, .6, n)
        measurements[col] = np.maximum(values, 0)
    frame = pd.DataFrame(measurements)
    frame.insert(0, 'Timestamp', pd.date_range('2025-01-01', periods=n, freq='s').strftime('%Y-%m-%d %H:%M:%S'))
    frame['Normal/Attack'] = np.where(attack, 'Attack', 'Normal')
    _missing(frame, rng, ['FIT101', 'LIT301', 'PIT501'])
    return frame


def generate_datasets(output: str | Path = 'datasets/synthetic', rows: int = 3000, seed: int = 381) -> dict:
    """Write segregated synthetic CSVs and per-file provenance/checksums."""
    if rows < 500:
        raise ValueError('Use at least500 rows for minority coverage and chronological fixtures')
    output = Path(output)
    children = np.random.SeedSequence(seed).spawn(3)
    result = {}
    for name, child in zip(['ton_iot', 'bot_iot', 'swat'], children):
        directory = output / name
        directory.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(child)
        if name == 'swat':
            frames = {'SYNTHETIC_sensor_log.csv': swat_process(rows, rng)}
            target = 'Normal/Attack'
        else:
            size = int(rows * .8)
            factory = ton_network if name == 'ton_iot' else bot_network
            frames = {'SYNTHETIC_train.csv': factory(size, rng, 0),
                      'SYNTHETIC_test.csv': factory(rows - size, rng, size)}
            target = 'type' if name == 'ton_iot' else 'category'
        files = []
        for filename, frame in frames.items():
            path = directory / filename
            frame.to_csv(path, index=False)
            files.append({'name': filename, 'rows': len(frame), 'columns': list(frame.columns),
                          'classes': frame[target].value_counts().to_dict(),
                          'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        manifest = {'synthetic': True, 'official_data': False, 'research_results': False,
                    'schema_inspired_by': name, 'generator': 'src.data.synthetic', 'generator_version': 1,
                    'seed': seed, 'rows': rows, 'files': files,
                    'limitations': 'Stochastic software fixtures; not sampled from or statistically fitted to original datasets. Not valid reproduction or intrusion-detection evidence.'}
        save_json(directory / 'SYNTHETIC_MANIFEST.json', manifest)
        (directory / 'README_SYNTHETIC.txt').write_text(
            'ARTIFICIALLY GENERATED SOFTWARE TEST DATA\n'
            'Not original ToN-IoT, Bot-IoT or SWaT observations. Do not report as real research data.\n')
        result[name] = {'directory': str(directory), 'manifest': str(directory / 'SYNTHETIC_MANIFEST.json'), 'rows': rows}
    save_json(output / 'SYNTHETIC_DATASETS.json', {'synthetic': True, 'datasets': result})
    return result
