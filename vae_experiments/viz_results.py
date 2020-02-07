from matplotlib import pyplot as plt
from os import listdir
from Utils.general import make_np_of_var_from_log_files, add_mean_std_plot_line

# path_to_files = './Results/elbo/jv_celeb_a/sb/'
# sb = make_np_of_var_from_log_files(variable_name='TeJVC',
#                                    files_list=[file for file in listdir(path=path_to_files)],
#                                    path_to_files=path_to_files)
# path_to_files = './Results/elbo/jv_celeb_a/igr/'
# igr = make_np_of_var_from_log_files(variable_name='TeJVC',
#                                    files_list = [file for file in listdir(path=path_to_files)],
#                                    path_to_files = path_to_files)
# path_to_files = './Results/elbo/jv_celeb_a/gs/'
# gs = make_np_of_var_from_log_files(variable_name='TeJV ',
#                                    files_list=[file for file in listdir(path = path_to_files)],
#                                    path_to_files=path_to_files)
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
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ===========================================================================================================
# Plots
# ===========================================================================================================
plt.style.use(style='ggplot')
plt.figure(dpi=150)

offset = 5
# add_mean_std_plot_line(runs=-igr, label='IGR(0.1)', color='green', offset=offset)
# add_mean_std_plot_line(runs=-gs, label='GS(0.15)', color='blue', offset=offset)
# add_mean_std_plot_line(runs=-sb, label='SB(0.1)', color='orange', offset=offset)
# plt.ylabel('Test Joint VAE loss')

plt.ylabel('Test ELBO (1.e+2)')
for _, model in models.items():
    linestyle = '-' if model['model'].find('cv') > 0 else '--'
    add_mean_std_plot_line(runs=model['results'], label=model['label'],
                           color=model['color'], linestyle=linestyle, offset=offset)

plt.xlabel('Epochs')
plt.legend()
# plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 6})
# plt.tight_layout(rect=(0, 0, 0.95, 1))
plt.tight_layout()
# plt.savefig('./Results/Outputs/jv_comp.png')
# plt.savefig('./Results/Outputs/elbo.eps', format='eps')
plt.savefig('./Results/Outputs/elbo.png')
plt.show()
# ===========================================================================================================
