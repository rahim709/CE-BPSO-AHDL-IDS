import numpy as np
import pytest
from src.evaluation.specialist import route_specialist, calibrate_specialist


def test_routing_preserves_other_attack_predictions_and_threshold_ties():
    out=route_specialist([0,1,2,3],[.6,.5,.9,.1],.5)
    assert out.tolist()==[1,0,2,3]


def test_cap_includes_fixed_false_alarms_and_ties():
    base=np.array([2,0,0,0,0,1]);scores=np.array([.9,.9,.8,.8,.1,.95]);y=np.array([0,0,0,0,0,1])
    chosen=calibrate_specialist(base,scores,y,max_fpr=.4)
    out=route_specialist(base,scores,chosen['threshold'])
    assert np.mean(out[y==0]!=0)<=.4
    assert out[-1]==1
    with pytest.raises(ValueError,match='exceed'):
        calibrate_specialist(base,scores,y,max_fpr=.1)


def test_calibration_rejects_invalid_inputs():
    with pytest.raises(ValueError):calibrate_specialist([0],[np.nan],[0])
    with pytest.raises(ValueError):calibrate_specialist([0],[.5],[1])
