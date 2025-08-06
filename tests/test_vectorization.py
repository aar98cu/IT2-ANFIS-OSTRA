import numpy as np
import copy
import os
import sys

sys.path.append(os.path.abspath('.'))

from python.it2anfis import (
    IT2ANFIS,
    output7,
    get_kalman_data,
    update_de_do,
    dmf_dp,
    dconsequent_dp,
)


def create_dummy_anfis():
    ni, mf, nr = 1, 1, 1
    nn = ni + ni * mf + 3 * nr + 1
    config = np.zeros((nn, nn))
    config[0, 1] = 1  # input to membership
    nodes = np.random.rand(nn, 2)
    mfparams = np.random.rand(ni * mf, 5)
    cparams = np.random.rand(nr, ni + 1)
    anfis = IT2ANFIS(
        config=config,
        mfparams=mfparams,
        cparams=cparams,
        nodes=nodes.copy(),
        ni=ni,
        mf=mf,
        nr=nr,
        nn=nn,
    )
    anfis.mfparam_de_do = np.zeros((ni * mf, 5))
    anfis.cparam_de_do = np.zeros((nr, ni + 1))
    anfis.de_do = np.random.rand(nn, 2)
    return anfis


def test_output7_equiv():
    anfis = create_dummy_anfis()
    anfis_loop = copy.deepcopy(anfis)

    # baseline loop implementation
    st = anfis_loop.ni + anfis_loop.ni * anfis_loop.mf + 2 * anfis_loop.nr
    inp = anfis_loop.nodes[:anfis_loop.ni, 0]
    for i in range(anfis_loop.nr):
        wn = anfis_loop.nodes[st - anfis_loop.nr + i, 0]
        anfis_loop.nodes[st + i, 0] = wn * (
            np.sum(anfis_loop.cparams[i, :-1] * inp) + anfis_loop.cparams[i, -1]
        )

    # vectorized
    output7(anfis)

    assert np.allclose(anfis_loop.nodes, anfis.nodes)


def test_get_kalman_data_equiv():
    anfis = create_dummy_anfis()
    target = 0.5
    st = anfis.ni + anfis.ni * anfis.mf + anfis.nr

    # baseline
    kalman_loop = np.zeros((anfis.ni + 1) * anfis.nr + 1)
    j = 0
    for i in range(st, st + anfis.nr):
        for k in range(anfis.ni):
            kalman_loop[j] = anfis.nodes[i, 0] * anfis.nodes[k, 0]
            j += 1
        kalman_loop[j] = anfis.nodes[i, 0]
        j += 1
    kalman_loop[j] = target

    kalman_vec = get_kalman_data(anfis, target)
    assert np.allclose(kalman_loop, kalman_vec)


def test_update_de_do_equiv():
    anfis = create_dummy_anfis()
    anfis_loop = copy.deepcopy(anfis)

    # baseline update_de_do
    s = 0
    for i in range(anfis_loop.ni, anfis_loop.ni + anfis_loop.ni * anfis_loop.mf):
        for j in range(1, 5):
            do_dp = dmf_dp(anfis_loop, i, j)
            if j == 1:
                anfis_loop.mfparam_de_do[s, j - 1] += anfis_loop.de_do[i, 0] * do_dp[0]
                anfis_loop.mfparam_de_do[s, j] += anfis_loop.de_do[i, 1] * do_dp[1]
            else:
                anfis_loop.mfparam_de_do[s, j] += np.sum(
                    anfis_loop.de_do[i, :] * do_dp
                ) / 2
        s += 1
    s = 0
    start = 1 + anfis_loop.ni + anfis_loop.ni * anfis_loop.mf + 2 * anfis_loop.nr
    for i in range(start, anfis_loop.config.shape[0]):
        for j in range(anfis_loop.ni + 1):
            do_dp = dconsequent_dp(anfis_loop, i, j)
            anfis_loop.cparam_de_do[s, j] += anfis_loop.de_do[i, 0] * do_dp
        s += 1

    update_de_do(anfis)

    assert np.allclose(anfis_loop.cparam_de_do, anfis.cparam_de_do)
    assert np.allclose(anfis_loop.mfparam_de_do, anfis.mfparam_de_do)
