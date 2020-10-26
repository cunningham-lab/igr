from matplotlib import pyplot as plt
from os import listdir
from Utils.general import make_np_of_var_from_log_files, add_mean_std_plot_line

scale = 100
path = './results/elbo/mnist/'
models = {
    1: {'model': 'igr', 'label': 'IGR-I', 'color': '#4daf4a'},
    2: {'model': 'pf', 'label': 'IGR-Planar', 'color': '#e41a1c'},
    3: {'model': 'sb', 'label': 'IGR-SB', 'color': '#377eb8'},
    4: {'model': 'gs', 'label': 'GS', 'color': '#984ea3'},
}
for key, val in models.items():
    dirs = path + val['model'] + '/'
    variable_name = 'TeELBO '
    output = make_np_of_var_from_log_files(variable_name=variable_name,
                                           files_list=[file for file in listdir(path=dirs)],
                                           path_to_files=dirs)
    models[key]['results'] = output / scale

# Plots
# ====================================================================================================
plt.style.use(style='ggplot')
plt.figure(dpi=150)

offset = 0
plt.ylabel('Test ELBO (1.e+2)')
for _, model in models.items():
    linestyle = '-' if model['model'].find('cv') > 0 else '--'
    add_mean_std_plot_line(runs=model['results'], label=model['label'],
                           color=model['color'], linestyle=linestyle, offset=offset)

plt.xlabel('Epochs')
plt.legend()
plt.tight_layout()
plt.savefig('./Results/elbo.pdf')
plt.show()
