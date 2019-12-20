import pickle
from os import listdir
import numpy as np
results_folder = './Log/logfiles/'

files = [file for file in listdir(path=results_folder)]
epochs_n = 100
output = {}
for file in files:
    if file.endswith('.log'):
        test_elbo = np.zeros(shape=epochs_n)
        test_elbo_closed = np.zeros(shape=epochs_n)
        with open(file=results_folder + file, mode='r') as f:
            counter = 0
            lines = f.readlines()
            for l in lines:
                split = l.split(sep='||')
                if len(split) > 1:
                    # test_elbo[counter] = float(split[1].split()[2])
                    test_elbo[counter] = float(split[2].split()[1])
                    test_elbo_closed[counter] = float(split[2].split()[1])
                    counter += 1

        output.update({str(file): test_elbo})

results_file = 'igr.pkl'
with open(file=results_folder + results_file, mode='wb') as f:
    pickle.dump(obj=output, file=f)
