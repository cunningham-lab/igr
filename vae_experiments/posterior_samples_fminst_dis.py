import numpy as np
from matplotlib import pyplot as plt
from Utils.posterior_sample_funcs import sample_from_posterior
import seaborn as sns
import pandas as pd

dataset_name = 'fmnist'
run_with_sample = False
plots_path = './Results/Outputs/'
path_to_results = './Results/Current_Model/'

diff_igr = sample_from_posterior(path_to_results=path_to_results, hyper_file='hyper_igr10_fmnist.pkl',
                                 dataset_name=dataset_name, weights_file='vae_igr10_fmnist.h5',
                                 model_type='GSMDis', run_with_sample=run_with_sample)
diff_igr_high = sample_from_posterior(path_to_results=path_to_results, hyper_file='hyper_igr50_fmnist.pkl',
                                      dataset_name=dataset_name, weights_file='vae_igr50_fmnist.h5',
                                      model_type='GSMDis', run_with_sample=run_with_sample)
diff_gs = sample_from_posterior(path_to_results=path_to_results, hyper_file='hyper_gs25_fmnist.pkl',
                                dataset_name=dataset_name, weights_file='vae_gs25_fmnist.h5',
                                model_type='ExpGSDis', run_with_sample=run_with_sample)
diff_gs_high = sample_from_posterior(path_to_results=path_to_results, hyper_file='hyper_gs67_fmnist.pkl',
                                     dataset_name=dataset_name, weights_file='vae_gs67_fmnist.h5',
                                     model_type='ExpGSDis', run_with_sample=run_with_sample)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ===========================================================================================================
# Make boxplot
# ===========================================================================================================
plt.style.use(style='ggplot')
plt.figure(dpi=100)

rows_list = []
model_name = ['IGR(0.5)', 'GS(0.67)', 'IGR(0.1)', 'GS(0.25)']
total_test_images = diff_igr.shape[0]
diff_list = [np.median(np.mean(diff_igr_high, axis=2), axis=1),
             np.median(np.mean(diff_gs_high, axis=2), axis=1),
             np.median(np.mean(diff_igr, axis=2), axis=1),
             np.median(np.mean(diff_gs, axis=2), axis=1)]
for i in range(len(diff_list)):
    for s in range(total_test_images):
        entry = {'Model': model_name[i], 'Distance': diff_list[i][s]}
        rows_list.append(entry)
df = pd.DataFrame(rows_list)
ax = sns.boxplot(x='Model', y='Distance', data=df, color='royalblue', boxprops={'alpha': 0.5})

plt.ylabel('Euclidean Distance')
# plt.ylim([0.0, 0.5])
plt.xlabel('Models')
plt.legend()
plt.savefig('./Results/Outputs/posterior_samples_fminst_dis.png')
plt.tight_layout()
plt.show()
# ===========================================================================================================
