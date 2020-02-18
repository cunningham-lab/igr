import pickle
from matplotlib import pyplot as plt
from Utils.general import count_zeros_in_gradient

path = './Results/grads/'
models = {
    # 1: {'model': 'igr_planar_cv', 'label': 'IGR_Planar(0.85)', 'color': '#e41a1c'},
    2: {'model': 'igr_planar_20', 'label': 'IGR-Planar(0.3)', 'color': '#e41a1c'},
    # 3: {'model': 'igr_sb_cv', 'label': 'IGR_SB(0.1)', 'color': '#377eb8'},
    4: {'model': 'igr_sb_20', 'label': 'IGR-SB(0.02)', 'color': '#377eb8'},
    # 5: {'model': 'igr_i_cv', 'label': 'IGR_I(1.0)', 'color': '#4daf4a'},
    6: {'model': 'igr_i_20', 'label': 'IGR-I(0.1)', 'color': '#4daf4a'},
    # 7: {'model': 'gs_cv', 'label': 'GS(1.0)', 'color': '#984ea3'},
    8: {'model': 'gs_20', 'label': 'GS(0.25)', 'color': '#984ea3'},
}
for key, val in models.items():
    dirs = path + val['model'] + '/gradients_100.pkl'
    with open(file=dirs, mode='rb') as f:
        output = pickle.load(f)
        models[key]['grads'] = count_zeros_in_gradient(grad_dict=output) * 100

# Plots
# ===========================================================================================================
plt.style.use(style='ggplot')
plt.figure(dpi=150)
for _, model in models.items():
    linestyle = '-' if model['model'].find('cv') > 0 else '--'
    plt.plot(model['grads'], label=model['label'],
             color=model['color'], linestyle=linestyle)

plt.ylabel('% of Vanished Gradients in Batch')
plt.xlabel('Epochs')
# plt.ylim([-0.01, 0.01])
plt.legend()
plt.tight_layout()
plt.savefig('./Results/grads/gradients.png')
plt.show()
