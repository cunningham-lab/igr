import numpy as np
from matplotlib import pyplot as plt
from Utils.posterior_sampling_funcs import sample_from_posterior
import seaborn as sns
import pandas as pd

dataset_name = 'mnist'
run_with_sample = False
plots_path = './Results/'
path = './Results/posterior_samples/' + dataset_name + '/'
models = {
    1: {'model': 'igr_planar_cv', 'label': 'IGR-Planar(0.85)', 'type': 'IGR_Planar_Dis'},
    3: {'model': 'igr_sb_cv', 'label': 'IGR-SB(0.15)', 'type': 'IGR_SB_Finite_Dis'},
    5: {'model': 'igr_i_cv', 'label': 'IGR-I(1.0)', 'type': 'IGR_I_Dis'},
    7: {'model': 'gs_cv', 'label': 'GS(1.0)', 'type': 'GS_Dis'},
    # 2: {'model': 'igr_planar_20', 'label': 'IGR-Planar(0.3)', 'type': 'IGR_Planar_Dis'},
    # 4: {'model': 'igr_sb_20', 'label': 'IGR-SB(0.02)', 'type': 'IGR_SB_Finite_Dis'},
    # 6: {'model': 'igr_i_20', 'label': 'IGR-I(0.1)', 'type': 'IGR_I_Dis'},
    # 8: {'model': 'gs_20', 'label': 'GS(0.25)', 'type': 'GS_Dis'},
}
output = 0
for key, val in models.items():
    dirs = path + val['model'] + '/'
    output = sample_from_posterior(path_to_results=dirs, hyper_file='hyper.pkl',
                                   dataset_name=dataset_name, weights_file='vae.h5',
                                   model_type=val['type'], run_with_sample=run_with_sample)
    models[key]['results'] = output
    # noinspection PyTypeChecker
    models[key]['diff'] = np.median(np.mean(output, axis=2), axis=1)

# Make boxplot
# ===========================================================================================================
plt.style.use(style='ggplot')
plt.figure(dpi=100)

rows_list = []
total_test_images = output.shape[0]
for _, mod in models.items():
    for s in range(total_test_images):
        entry = {'Model': mod['label'], 'Distance': mod['diff'][s]}
        rows_list.append(entry)
df = pd.DataFrame(rows_list)
ax = sns.boxplot(x='Model', y='Distance', data=df, color='royalblue', boxprops={'alpha': 0.5})

plt.ylabel('Euclidean Distance')
# plt.ylim([0.0, 0.5])
plt.xlabel('Models')
plt.legend()
plt.savefig('./Results/posterior_samples_' + dataset_name + '.png')
plt.tight_layout()
plt.show()
