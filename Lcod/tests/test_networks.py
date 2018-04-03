import numpy as np
import pytest

from Lcod.ista_tf import IstaTF
from Lcod.lista_network import LIstaNetwork
from Lcod.facto_network import FactoNetwork
from Lcod.fista_tf import FistaTF
from Lcod.lfista_network import LFistaNetwork
from Lcod.simple_problem_generator import SimpleProblemGenerator


class _TestNetwork:

    @pytest.mark.parametrize('n_layers', [1, 3, 7, 12, 19, 25])
    def test_init(self, n_layers):
        K = 10
        p = 5
        n = 100
        lmbd = .1
        D = np.random.normal(size=(K, p))

        D /= np.sqrt((D*D).sum(axis=1))[:, None]
        pb = SimpleProblemGenerator(D, lmbd)
        X, _, Z, lmbd = pb.get_batch(n)

        feed_test = {"Z": Z,
                     "X": X,
                     "lmbd": lmbd}

        classic = self.classic_class(D)
        classic.optimize(X, lmbd, Z, max_iter=n_layers, tol=-1)
        network = self.network_class(D, n_layers=n_layers)
        c = network.cost(**feed_test)
        assert np.isclose(c, classic.train_cost[n_layers])

        network.terminate()
        classic.terminate()


class MixtureLIsta:
    network_class = LIstaNetwork
    classic_class = IstaTF


class MixtureFactNet:
    network_class = FactoNetwork
    classic_class = IstaTF


class MixtureLFista:
    network_class = LFistaNetwork
    classic_class = FistaTF


class TestLFista(_TestNetwork, MixtureLFista):
    pass


class TestLIsta(_TestNetwork, MixtureLIsta):
    pass


class TestFacNet(_TestNetwork, MixtureFactNet):
    pass
