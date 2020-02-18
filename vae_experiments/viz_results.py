from matplotlib import pyplot as plt
from os import listdir
from Utils.general import make_np_of_var_from_log_files, add_mean_std_plot_line

scale = 100
path = './Results/elbo/mnist/'
models = {
    1: {'model': 'igr_planar_cv', 'label': 'IGR-Planar(0.85)', 'color': '#e41a1c'},
    3: {'model': 'igr_sb_cv', 'label': 'IGR-SB(0.15)', 'color': '#377eb8'},
    5: {'model': 'igr_i_cv', 'label': 'IGR-I(1.0)', 'color': '#4daf4a'},
    7: {'model': 'gs_cv', 'label': 'GS(1.0)', 'color': '#984ea3'},
    2: {'model': 'igr_planar_20', 'label': 'IGR-Planar(0.3)', 'color': '#e41a1c'},
    4: {'model': 'igr_sb_20', 'label': 'IGR-SB(0.02)', 'color': '#377eb8'},
    6: {'model': 'igr_i_20', 'label': 'IGR-I(0.1)', 'color': '#4daf4a'},
    8: {'model': 'gs_20', 'label': 'GS(0.25)', 'color': '#984ea3'},
}
for key, val in models.items():
    dirs = path + val['model'] + '/'
    variable_name = 'TeELBOC ' if val['model'].find('igr') > 0 else 'TeELBO '
    output = make_np_of_var_from_log_files(variable_name=variable_name,
                                           files_list=[file for file in listdir(path=dirs)],
                                           path_to_files=dirs)
    models[key]['results'] = output / scale

# Plots
# ===========================================================================================================
plt.style.use(style='ggplot')
plt.figure(dpi=150)

offset = 5
plt.ylabel('Test ELBO (1.e+2)')
for _, model in models.items():
    linestyle = '-' if model['model'].find('cv') > 0 else '--'
    add_mean_std_plot_line(runs=model['results'], label=model['label'],
                           color=model['color'], linestyle=linestyle, offset=offset)

plt.xlabel('Epochs')
plt.legend()
plt.tight_layout()
plt.savefig('./Results/elbo.png')
plt.show()
