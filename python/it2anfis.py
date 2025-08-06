import numpy as np
from dataclasses import dataclass, field

@dataclass
class IT2ANFIS:
    config: np.ndarray
    mfparams: np.ndarray
    cparams: np.ndarray
    nodes: np.ndarray
    ni: int
    mf: int
    nr: int
    nn: int
    last_decrease_ss: int = 1
    last_increase_ss: int = 1
    rules: np.ndarray = field(default_factory=lambda: np.empty((0,)))
    P: np.ndarray = field(default=None)
    S: np.ndarray = field(default=None)
    mfparam_de_do: np.ndarray = field(default=None)
    cparam_de_do: np.ndarray = field(default=None)
    de_do: np.ndarray = field(default=None)


def output1(anfis: IT2ANFIS) -> IT2ANFIS:
    mfparams = anfis.mfparams
    for i in range(anfis.ni):
        x = anfis.nodes[i, 0]
        for j in range(anfis.mf):
            ind = anfis.ni + i * anfis.mf + j
            b = mfparams[ind - anfis.ni, 2]
            c = mfparams[ind - anfis.ni, 3]
            # lower membership degree
            ai = mfparams[ind - anfis.ni, 0]
            tmp1i = (x - c) / ai
            if tmp1i == 0:
                tmp2i = 0.0
            else:
                tmp2i = (tmp1i * tmp1i) ** b
            anfis.nodes[ind, 0] = mfparams[ind - anfis.ni, 4] / (1 + tmp2i)
            # upper membership degree
            as_ = mfparams[ind - anfis.ni, 1]
            tmp1s = (x - c) / as_
            if tmp1s == 0:
                tmp2s = 0.0
            else:
                tmp2s = (tmp1s * tmp1s) ** b
            anfis.nodes[ind, 1] = 1 / (1 + tmp2s)
    return anfis


def output2(anfis: IT2ANFIS) -> IT2ANFIS:
    st = anfis.ni + anfis.ni * anfis.mf
    for i in range(st, st + anfis.nr):
        idx = np.where(anfis.config[:, i] == 1)[0]
        tmpi = np.cumprod(anfis.nodes[idx, 0])
        tmps = np.cumprod(anfis.nodes[idx, 1])
        anfis.nodes[i, 0] = tmpi[-1]
        anfis.nodes[i, 1] = tmps[-1]
    return anfis


def output3_4_5_6(anfis: IT2ANFIS) -> IT2ANFIS:
    st = anfis.ni + anfis.ni * anfis.mf
    wi = anfis.nodes[st:st + anfis.nr, 0].copy()
    ws = anfis.nodes[st:st + anfis.nr, 1].copy()
    O1 = np.argsort(wi)
    wi = np.sort(wi)
    ws = np.sort(ws)
    wn = (wi + ws) / 2
    yi = np.sum(wi * wn) / np.sum(wn)
    ys = np.sum(ws * wn) / np.sum(wn)
    l = 0
    r = 0
    for i in range(anfis.nr - 1):
        if wi[i] <= yi <= wi[i + 1]:
            l = i + 1
        if ws[i] <= ys <= ws[i + 1]:
            r = i + 1
    Xl = np.concatenate([ws[:l], wi[l:]]) / np.sum(np.concatenate([ws[:l], wi[l:]]))
    Xr = np.concatenate([wi[:r], ws[r:]]) / np.sum(np.concatenate([wi[:r], ws[r:]]))
    X = (Xl + Xr) / 2
    X_unsorted = np.empty_like(X)
    X_unsorted[O1] = X
    st = anfis.ni + anfis.ni * anfis.mf + anfis.nr
    anfis.nodes[st:st + anfis.nr, 0] = X_unsorted
    anfis.nodes[st:st + anfis.nr, 1] = X_unsorted
    return anfis


def output7(anfis: IT2ANFIS) -> IT2ANFIS:
    st = anfis.ni + anfis.ni * anfis.mf + 2 * anfis.nr
    inp = anfis.nodes[:anfis.ni, 0]
    wn = anfis.nodes[st - anfis.nr:st, 0]
    consequent = np.einsum('ij,j->i', anfis.cparams[:, :-1], inp) + anfis.cparams[:, -1]
    anfis.nodes[st:st + anfis.nr, 0] = wn * consequent
    return anfis


def output8(anfis: IT2ANFIS) -> IT2ANFIS:
    st = anfis.nodes.shape[0]
    anfis.nodes[-1, 0] = np.sum(anfis.nodes[st - anfis.nr - 1:st - 1, 0])
    return anfis


def get_kalman_data(anfis: IT2ANFIS, target: float) -> np.ndarray:
    st = anfis.ni + anfis.ni * anfis.mf + anfis.nr
    w = anfis.nodes[st:st + anfis.nr, 0]
    x = anfis.nodes[:anfis.ni, 0]
    w_x = np.einsum('i,j->ij', w, x).reshape(-1)
    kalman_data = np.concatenate([w_x, w, [target]])
    return kalman_data


def kalman_filter(anfis: IT2ANFIS, kalman_data: np.ndarray, k: int) -> IT2ANFIS:
    k_p_n = (anfis.ni + 1) * anfis.nr
    alpha = 1000000.0
    if k == 1:
        anfis.P = np.zeros(k_p_n)
        anfis.S = alpha * np.eye(k_p_n)
    x = kalman_data[:-1]
    y = kalman_data[-1]
    tmp1 = anfis.S @ x
    denom = 1 + np.sum(tmp1 * x)
    tmp_m = np.outer(tmp1, tmp1) * (-1.0 / denom)
    anfis.S = anfis.S + tmp_m
    diff = y - np.sum(x * anfis.P)
    tmp1 = diff * (anfis.S @ x)
    anfis.P = anfis.P + tmp1
    anfis.cparams = anfis.P.reshape(anfis.nr, anfis.ni + 1)
    return anfis


def clear_de_dp(anfis: IT2ANFIS) -> IT2ANFIS:
    anfis.cparam_de_do = np.zeros((anfis.nr, anfis.ni + 1))
    anfis.mfparam_de_do = np.zeros((anfis.ni * anfis.mf, 5))
    return anfis


def update_parameter(anfis: IT2ANFIS, step_size: float) -> IT2ANFIS:
    tmp = anfis.mfparam_de_do
    len_ = np.sqrt(np.sum(tmp * tmp))
    if len_ == 0:
        return anfis
    anfis.mfparams = anfis.mfparams - step_size * anfis.mfparam_de_do / len_
    C = anfis.mfparams[:, 4]
    C[C > 1] = 1
    C[C < 0.1] = 0.1
    anfis.mfparams[:, 4] = C
    return anfis


def update_step_size(anfis: IT2ANFIS, error_array: np.ndarray, iter_: int, step_size: float, decrease_rate: float, increase_rate: float):
    def check_decrease_ss(error_array, last_change, current):
        if current - last_change < 4:
            return False
        if (error_array[current-1] < error_array[current-2] and
            error_array[current-2] > error_array[current-3] and
            error_array[current-3] < error_array[current-4] and
            error_array[current-4] > error_array[current-5]):
            return True
        return False

    def check_increase_ss(error_array, last_change, current):
        if current - last_change < 4:
            return False
        if (error_array[current-1] < error_array[current-2] and
            error_array[current-2] < error_array[current-3] and
            error_array[current-3] < error_array[current-4] and
            error_array[current-4] < error_array[current-5]):
            return True
        return False

    if iter_ >=5 and check_decrease_ss(error_array, anfis.last_decrease_ss, iter_+1):
        step_size *= decrease_rate
        anfis.last_decrease_ss = iter_ + 1
    elif iter_ >=5 and check_increase_ss(error_array, anfis.last_increase_ss, iter_+1):
        step_size *= increase_rate
        anfis.last_increase_ss = iter_ + 1
    return anfis, step_size


def update_de_do(anfis: IT2ANFIS) -> IT2ANFIS:
    s = 0
    for i in range(anfis.ni, anfis.ni + anfis.ni * anfis.mf):
        for j in range(1,5):
            do_dp = dmf_dp(anfis, i, j)
            if j == 1:
                anfis.mfparam_de_do[s, j-1] += anfis.de_do[i, 0] * do_dp[0]
                anfis.mfparam_de_do[s, j] += anfis.de_do[i, 1] * do_dp[1]
            else:
                anfis.mfparam_de_do[s, j] += np.sum(anfis.de_do[i,:] * do_dp) / 2
        s += 1
    s = 0
    start = 1 + anfis.ni + anfis.ni * anfis.mf + 2 * anfis.nr
    de = anfis.de_do[start:start + anfis.nr, 0]
    wn = anfis.nodes[start - anfis.nr:start, 0]
    inp = np.append(anfis.nodes[:anfis.ni,0], 1)
    anfis.cparam_de_do += np.outer(de * wn, inp)
    return anfis


def dmf_dp(anfis: IT2ANFIS, i: int, j: int):
    idx = np.where(anfis.config[:, i] == 1)[0]
    x = anfis.nodes[idx, 0]
    ai = anfis.mfparams[i - anfis.ni, 0]
    as_ = anfis.mfparams[i - anfis.ni, 1]
    b = anfis.mfparams[i - anfis.ni, 2]
    c = anfis.mfparams[i - anfis.ni, 3]
    B = anfis.mfparams[i - anfis.ni, 4]
    tmp1i = (x - c) / ai
    if tmp1i == 0:
        tmp2i = 0.0
    else:
        tmp2i = (tmp1i * tmp1i) ** b
    denomi = (1 + tmp2i) * (1 + tmp2i)
    tmp1s = (x - c) / as_
    if tmp1s == 0:
        tmp2s = 0.0
    else:
        tmp2s = (tmp1s * tmp1s) ** b
    denoms = (1 + tmp2s) * (1 + tmp2s)
    if j == 1:
        return np.array([2*b*B*tmp2i/(ai*denomi), 2*b*tmp2s/(as_*denoms)])
    elif j == 2:
        if tmp1i == 0 and tmp1s == 0:
            return np.array([0.0, 0.0])
        return np.array([-np.log(tmp1i*tmp1i)*B*tmp2i/denomi, -np.log(tmp1s*tmp1s)*tmp2s/denoms])
    elif j == 3:
        if np.allclose(x, c):
            return np.array([0.0,0.0])
        return np.array([2*b*B*tmp2i/((x - c)*denomi), 2*b*tmp2s/((x - c)*denoms)])
    elif j == 4:
        return np.array([B,1.0])
    return np.array([0.0,0.0])


def dconsequent_dp(anfis: IT2ANFIS, i: int, j: int):
    wn = anfis.nodes[i - anfis.nr, 0]
    inp = np.append(anfis.nodes[:anfis.ni,0], 1)
    return wn * inp[j]


def calculate_de_do(anfis: IT2ANFIS, de_dout: float) -> IT2ANFIS:
    anfis.de_do = np.zeros_like(anfis.nodes)
    anfis.de_do[-1, :] = de_dout
    for i in range(len(anfis.nodes) - 2, anfis.ni, -1):
        de_do = np.array([0.0, 0.0])
        II = np.where(anfis.config[i, :] == 1)[0]
        I = II[II > i]
        for jj in I:
            tmp1 = anfis.de_do[jj, :]
            tmp2 = derivative_o_o(anfis, i, jj)
            de_do += tmp1 * tmp2
        anfis.de_do[i, :] = de_do
    return anfis


def derivative_o_o(anfis: IT2ANFIS, i: int, j: int):
    if i > anfis.ni + anfis.ni * anfis.mf + 2 * anfis.nr:
        return np.array([1.0, 1.0])
    elif i > anfis.ni + anfis.ni * anfis.mf + anfis.nr:
        return do4_do3(anfis, i, j)
    elif i > anfis.ni + anfis.ni * anfis.mf:
        return do3_do2(anfis, i, j)
    elif i > anfis.ni:
        return np.array([
            anfis.nodes[j,0]/anfis.nodes[i,0],
            anfis.nodes[j,1]/anfis.nodes[i,1]
        ])
    return np.array([0.0,0.0])


def do4_do3(anfis: IT2ANFIS, i: int, j: int):
    inp = anfis.nodes[:anfis.ni,0]
    jj = j - (anfis.ni + anfis.ni * anfis.mf + 2 * anfis.nr)
    if jj < 0 or jj >= anfis.nr:
        return np.array([0.0, 0.0])
    val = np.sum(anfis.cparams[jj, :-1] * inp) + anfis.cparams[jj, -1]
    return np.array([val, val])


def do3_do2(anfis: IT2ANFIS, i: int, j: int):
    II = np.where(anfis.config[:, j] == 1)[0]
    I = II[II < j]
    totali = np.sum(anfis.nodes[I,0])
    totals = np.sum(anfis.nodes[I,1])
    if j - i == anfis.nr:
        return np.array([(totali - anfis.nodes[i,0])/(totali*totali),
                         (totals - anfis.nodes[i,1])/(totals*totals)])
    else:
        return np.array([-anfis.nodes[j - anfis.nr,0]/(totali*totali),
                         -anfis.nodes[j - anfis.nr,1]/(totals*totals)])


def train_anfis(data, epoch_n, mf, step_size, decrease_rate, increase_rate, B):
    inputs = data[:, :-1]
    output = data[:, -1]
    ndata = data.shape[0]
    ni = inputs.shape[1]
    nr = mf ** ni
    nn = ni + ni * mf + 3 * nr + 1
    min_RMSE = np.inf
    mn = inputs.min(axis=0)
    mx = inputs.max(axis=0)
    mm = mx - mn
    mfparams = []
    for i in range(ni):
        tmp = np.column_stack((np.linspace(mn[i], mx[i], mf), np.ones(mf)*0.5))
        mfparams.append(np.hstack((np.tile([mm[i]/5, mm[i]*0.24, 1], (mf,1)), tmp)))
    mfparams = np.vstack(mfparams)
    cparams = np.zeros((nr, ni + 1))
    config = np.zeros((nn, nn))
    nodes = np.zeros((nn,2))
    st = ni
    for i in range(ni):
        config[i, st:st+mf] = 1
        st += mf
    st = ni + ni * mf + 1
    n = ni
    x = np.arange(ni+1, ni+mf+1)
    d = []
    while n > 1:
        c = []
        for xi in x:
            for j in range(mf):
                c.append([xi, j + ni + mf*(ni - n + 1)])
        n -= 1
        if n > 1:
            x = np.array(c)
            d = []
        else:
            d = np.array(c)
    for i in range(mf**ni):
        for j in range(ni):
            config[d[i,j]-1, st-1] = 1
        st += 1
    for i in range(nr):
        config[ni+ni*mf+i, ni+ni*mf+nr+i] = 1
    for i in range(nr):
        config[ni+ni*mf+nr+i, ni+ni*mf+2*nr+i] = 1
    for i in range(nr):
        config[ni+ni*mf+2*nr+i, -1] = 1
    for i in range(ni):
        for j in range(nr):
            config[i, ni+ni*mf+2*nr+j] = 1
    anfis = IT2ANFIS(config=config, mfparams=mfparams, cparams=cparams, nodes=nodes,
                     ni=ni, mf=mf, nr=nr, nn=nn, rules=d)
    y_anfis = np.zeros((ndata,1))
    RMSE = np.zeros((epoch_n,1))
    for i in range(epoch_n):
        output_1_to_6 = np.zeros((anfis.nn, ndata*2))
        for j in range(ndata):
            anfis.nodes[:anfis.ni,0] = inputs[j,:]
            anfis.nodes[:anfis.ni,1] = inputs[j,:]
            anfis = output1(anfis)
            anfis = output2(anfis)
            anfis = output3_4_5_6(anfis)
            output_1_to_6[:, j*2:(j+1)*2] = anfis.nodes
            kalman_data = get_kalman_data(anfis, output[j])
            anfis = kalman_filter(anfis, kalman_data, j+1)
        anfis = clear_de_dp(anfis)
        for j in range(ndata):
            anfis.nodes = output_1_to_6[:, j*2:(j+1)*2]
            anfis = output7(anfis)
            anfis = output8(anfis)
            y_anfis[j,0] = anfis.nodes[-1,0]
            target = output[j]
            de_dout = -2*(target - y_anfis[j,0])
            anfis = calculate_de_do(anfis, de_dout)
            anfis = update_de_do(anfis)
        diff = y_anfis[:,0] - output
        total_sq_error = np.sum(diff*diff)
        RMSE[i,0] = np.sqrt(total_sq_error/ndata)
        if RMSE[i,0] < min_RMSE:
            it2anfis = IT2ANFIS(**vars(anfis))
            min_RMSE = RMSE[i,0]
        anfis = update_parameter(anfis, step_size)
        anfis, step_size = update_step_size(anfis, RMSE[:i+1,0], i, step_size, decrease_rate, increase_rate)
    anfis = it2anfis
    for j in range(ndata):
        anfis.nodes[:anfis.ni,0] = inputs[j,:]
        anfis.nodes[:anfis.ni,1] = inputs[j,:]
        anfis = output1(anfis)
        anfis = output2(anfis)
        anfis = output3_4_5_6(anfis)
        anfis = output7(anfis)
        anfis = output8(anfis)
        y_anfis[j,0] = anfis.nodes[-1,0]
    return it2anfis, y_anfis, RMSE


def evalmyanfis(anfis: IT2ANFIS, inputs: np.ndarray) -> np.ndarray:
    ndata = inputs.shape[0]
    yhat = np.zeros((ndata,1))
    for j in range(ndata):
        anfis.nodes[:anfis.ni,0] = inputs[j,:]
        anfis.nodes[:anfis.ni,1] = inputs[j,:]
        anfis = output1(anfis)
        anfis = output2(anfis)
        anfis = output3_4_5_6(anfis)
        anfis = output7(anfis)
        anfis = output8(anfis)
        yhat[j,0] = anfis.nodes[-1,0]
    return yhat
