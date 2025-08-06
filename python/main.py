import numpy as np
from pathlib import Path
from .it2anfis import train_anfis, evalmyanfis

def main():
    name = 'ex2'
    epoch_n = 100
    mf_n = 2
    step_size = 0.1
    decrease_rate = 0.5
    increase_rate = 1.1
    B = 0.5
    data_path = Path('Input') / f'{name}.txt'
    data = np.loadtxt(data_path)
    it2anfis, y_anfis, RMSE = train_anfis(data, epoch_n, mf_n, step_size, decrease_rate, increase_rate, B)
    y_anfis = evalmyanfis(it2anfis, data[:, :-1])
    rmse = np.sqrt(np.sum((y_anfis[:,0] - data[:, -1])**2)/data.shape[0])
    print('rmse =', rmse)

if __name__ == '__main__':
    main()
