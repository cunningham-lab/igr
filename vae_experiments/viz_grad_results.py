import pickle
from matplotlib import pyplot as plt
from Utils.general import count_zeros_in_gradient

path_to_files_gs_15 = './Results/gradients/grad_gs_25/gradients_100.pkl'
path_to_files_gs_67 = './Results/gradients/grad_gs_67/gradients_100.pkl'
path_to_files_igr_50 = './Results/gradients/grad_igr_50/gradients_100.pkl'
path_to_files_igr_10 = './Results/gradients/grad_igr_10/gradients_100.pkl'

with open(file=path_to_files_gs_15, mode='rb') as f:
    gs_15 = pickle.load(f)

with open(file=path_to_files_gs_67, mode='rb') as f:
    gs_67 = pickle.load(f)

with open(file=path_to_files_igr_10, mode='rb') as f:
    igr_10 = pickle.load(f)

with open(file=path_to_files_igr_50, mode='rb') as f:
    igr_50 = pickle.load(f)

grad_gs_15 = count_zeros_in_gradient(grad_dict=gs_15)
grad_gs_67 = count_zeros_in_gradient(grad_dict=gs_67)
grad_igr_10 = count_zeros_in_gradient(grad_dict=igr_10)
grad_igr_50 = count_zeros_in_gradient(grad_dict=igr_50)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ===========================================================================================================
# Plots
# ===========================================================================================================
plt.style.use(style='ggplot')
plt.figure(dpi=150)
plt.plot(grad_igr_50, label='IGR(0.50)', color='blue')
plt.plot(grad_gs_67, label='GS(0.67)', color='green')
plt.plot(grad_igr_10, label='IGR(0.10)', color='blue', linestyle='--')
plt.plot(grad_gs_15, label='GS(0.25)', color='green', linestyle='--')
plt.ylabel('% of Vanished Gradients in Batch')
plt.xlabel('Epochs')
plt.legend()
plt.savefig('./Results/Outputs/gradients.png')
plt.tight_layout()
plt.show()
# ===========================================================================================================
