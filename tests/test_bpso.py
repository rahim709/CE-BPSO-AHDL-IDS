import numpy as np
from src.optimization.bpso import BPSO, SwarmSettings


def test_maximization_and_ties():
    swarm = BPSO(4, SwarmSettings(population=3), 4)
    swarm.tell(np.array([.1, .9, .3]))
    best = swarm.best_position.copy()
    swarm.step()
    swarm.tell(np.array([.2, .8, .4]))
    assert swarm.best_fitness == .9
    assert np.array_equal(swarm.best_position, best)
    assert np.allclose(swarm.personal_fitness, [.2, .9, .4])


def test_equations_and_rng_resume():
    settings = SwarmSettings(population=2, c1=0, c2=0, w_max=.8, w_min=.4)
    swarm = BPSO(3, settings, 42)
    swarm.velocity[:] = 100
    clone = BPSO(3, settings, 0)
    clone.load_state_dict(swarm.state_dict())
    swarm.step()
    clone.step()
    assert (swarm.velocity >= 0).all() and (swarm.velocity <= 1).all()
    assert np.array_equal(swarm.position, clone.position)
    assert np.array_equal(swarm.velocity, clone.velocity)


def test_inertia_equation():
    swarm = BPSO(5, SwarmSettings(population=2, c1=0, c2=0), 1)
    swarm.velocity[:] = 1.
    swarm.step()
    assert np.allclose(swarm.velocity, .8)
    swarm.step()
    assert np.allclose(swarm.velocity, .8 * (.8 - .4/30))
