import pickle

results_file = './Results/gradients_10.pkl'
with open(file=results_file, mode='rb') as f:
    import pdb
    pdb.set_trace()
    grad_monitor_dict = pickle.load(f)
    grad_monitor_dict.keys()
    grad_monitor_dict[9370].shape
    grad_monitor_dict[9370]
    grad_monitor_dict[1]
