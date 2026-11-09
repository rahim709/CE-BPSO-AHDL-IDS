import json
import pytest
from src.optimization.search import run_search
from src.optimization.evaluator import Evaluator, TargetReached
from src.optimization.encoding import BinaryCodec
from test_training_search import config


def test_target_only_at_terminal_and_persisted(fixture_data, architecture, tmp_path):
    data, _ = fixture_data
    cfg = config(data)
    cfg['search']['stop_f1'] = 0.
    evaluator = Evaluator(cfg,tmp_path/'trial',BinaryCodec('debug'))
    bits=evaluator.codec.encode(architecture)
    evaluator.evaluate(bits,1)
    with pytest.raises(TargetReached):
        evaluator.evaluate(bits,2)
    assert evaluator.events[-1]['fitness'] is not None
    assert evaluator.used_epochs==2


@pytest.mark.parametrize('method',['bpso','bpso_sh','hyperband','dehb','random'])
def test_native_controls_stop_on_target(fixture_data,tmp_path,method):
    data,_=fixture_data
    cfg=config(data)
    cfg['search'].update(method=method,stop_f1=0.)
    result=run_search(cfg,tmp_path/method)
    assert result['status']=='target_reached'
    assert result['best']['completed_budget']==2
    assert result['best']['fitness']>=0
