from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
import os
import pandas as pd
import pickle as pkl
from pathlib import Path

LDA_PATH = Path("~/Documents/P10/hot3d/grasp_lda_mano_pca.pkl").expanduser()
DATA_PATH = Path("results").absolute()
PARAM_PATH = Path("params").absolute()
data_files = [f for f in DATA_PATH.iterdir() if f.is_file() and "MANO" in f.stem and f.suffix == ".csv" and f.stem.startswith("results_260601")]
unique_models = set(
    [f.stem.split("_")[-2] for f in data_files]
)

lda:LDA = pkl.load(open(LDA_PATH, "rb"))["lda"]
for model in sorted(unique_models):
    precision = []
    model_files = [f for f in data_files if f.stem.split("_")[-2] == model]
    for f in model_files:
        df = pd.read_csv(f)
        data = df.values
        true = data[:,:3]
        pred = data[:,3:]
        lbl_true = lda.predict(true)
        lbl_pred = lda.predict(pred)
        for x in "power lateral pinch".split():
            print((lbl_true == x).mean(), (lbl_pred == x).mean())

        exit()