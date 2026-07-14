import matplotlib.pyplot as plt
import pandas as pd
import sklearn
import pickle
from dataclasses import dataclass
import os


data = pd.read_csv("ml_results.csv")

model_set = set([c[:-3] for c in data.columns if c.endswith("_R2")])

pca = pickle.load(open(os.path.expanduser("~/Documents/P10/hot3d/pca.pkl"), "rb"))

@dataclass
class ModelResults:
    mse: float
    r2: float
    R2: float
weights = pca.explained_variance_ratio_
print(weights)
data_dict = {}
for model in model_set:
    mse = data[f'{model}_mse']
    r2 = data[f'{model}_r2']
    R2 = data[f'{model}_R2']
    data_dict[model] = ModelResults(mse=mse, r2=r2, R2=R2)

fig, ax = plt.subplots(3,2)
ax[0][0].bar(list(data_dict.keys()), list(map(lambda x: (weights*getattr(x, 'mse')).sum(),data_dict.values())))
ax[0][1].bar(list(data_dict.keys()), list(map(lambda x: getattr(x, 'mse')[0],data_dict.values())))
ax[1][0].bar(list(data_dict.keys()), list(map(lambda x: (weights*getattr(x, 'R2')).sum(),data_dict.values())))
ax[1][1].bar(list(data_dict.keys()), list(map(lambda x: getattr(x, 'R2')[0],data_dict.values())))
ax[2][0].bar(list(data_dict.keys()), list(map(lambda x: (weights*getattr(x, 'r2')).sum(),data_dict.values())))
ax[2][1].bar(list(data_dict.keys()), list(map(lambda x: (getattr(x, 'r2'))[0],data_dict.values())))
ax[1][0].set_ylim(ymin = -0.1, ymax=1)
ax[1][1].set_ylim(ymin = -0.1, ymax=1)
ax[1][0].grid()
ax[1][1].grid()
ax[2][0].set_ylim(ymin= 0, ymax=1)
ax[2][1].set_ylim(ymin= 0, ymax=1)

plt.show()
