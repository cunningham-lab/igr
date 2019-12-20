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

path_to_files = './Results/elbo/mnist/gs_high/'
gs_high = make_np_of_var_from_log_files(variable_name='TeELBO ',
                                        files_list=[file for file in listdir(path=path_to_files)],
                                        path_to_files=path_to_files)

path_to_files = './Results/elbo/mnist/gs_low/'
gs_low = make_np_of_var_from_log_files(variable_name='TeELBO ',
                                       files_list=[file for file in listdir(path=path_to_files)],
                                       path_to_files=path_to_files)

path_to_files = './Results/elbo/mnist/igr_high/'
igr_high = make_np_of_var_from_log_files(variable_name='TeELBOC',
                                         files_list=[file for file in listdir(path=path_to_files)],
                                         path_to_files=path_to_files)

path_to_files = './Results/elbo/mnist/igr_low/'
igr_low = make_np_of_var_from_log_files(variable_name='TeELBOC',
                                        files_list=[file for file in listdir(path=path_to_files)],
                                        path_to_files=path_to_files)
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

plt.ylabel('Test ELBO')
add_mean_std_plot_line(runs=igr_high, label='IGR(0.50)', color='blue', offset=offset)
add_mean_std_plot_line(runs=gs_high, label='GS(0.67)', color='green', offset=offset)
add_mean_std_plot_line(runs=igr_low, label='IGR(0.1)', color='blue', offset=offset, linestyle='--')
add_mean_std_plot_line(runs=gs_low, label='GS(0.25)', color='green', offset=offset, linestyle='--')

plt.xlabel('Epochs')
plt.legend()
# plt.savefig('./Results/Outputs/jv_comp.png')
plt.savefig('./Results/Outputs/elbo.png')
plt.tight_layout()
plt.show()
# ===========================================================================================================
