from graspcnn.data import PrecachedMANOGraspDataset
import torch
from torch_geometric.data import Batch
from graspcnn.features import FeatureCalculator
from tqdm import tqdm

from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, HuberRegressor
from sklearn.metrics import mean_squared_error, r2_score

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_set = torch.load(".data_cache/data_src_train.pt", weights_only=False)
val_set = torch.load(".data_cache/data_src_val.pt", weights_only=False)
test_set = torch.load(".data_cache/data_src_test.pt", weights_only=False)

def get_formatted_dataset(dataset, batch_size=16):
    feature_calculator = FeatureCalculator()
    x, y = [], []
    for i  in tqdm(range(0, len(dataset), batch_size)):
        samples = [dataset[j] for j in range(i, min(i + batch_size, len(dataset)))]
        batch = Batch.from_data_list(samples).to(DEVICE)
        feat_vecs = feature_calculator.forward(batch.x, batch.edge_index, batch.meta, batch=batch.batch)
        x.append(feat_vecs.cpu())
        y.append(batch.y.cpu())
    return torch.cat(x, dim=0), torch.cat(y, dim=0)

x_train, y_train = get_formatted_dataset(train_set)
x_val, y_val = get_formatted_dataset(val_set)
x_test, y_test = get_formatted_dataset(test_set)

print(x_train.shape, y_train.shape)
print(x_val.shape, y_val.shape)
print(x_test.shape, y_test.shape)
x_test = torch.cat([x_test, x_val], dim=0)
y_test = torch.cat([y_test, y_val], dim=0)
print(x_test.shape, y_test.shape)
data = [{"PC": i+1} for i in range(15)]
for model in (SVR, RandomForestRegressor, LinearRegression, Ridge):
    m_name = model.__name__
    print(m_name)
    s = f"\t\tMSE\t\tR²\t\tr²"
    print(s)
    for i in range(y_test.shape[1]):
        reg = model().fit(x_train, x_train[:,i])
        y_pred_test = torch.Tensor(reg.predict(x_test))
        res = y_pred_test - y_test[:,i]
        mse = mean_squared_error(res, y_pred_test)
        cov = torch.cov(torch.cat([y_pred_test.unsqueeze(0), y_test[:,i].unsqueeze(0)], dim=0))
        assert cov.shape[0] == 2
        s_pred = cov[0,0]**0.5
        s_val = cov[1,1]**0.5
        cov = cov[0,1]
        R2 = r2_score(y_test[:,i], y_pred_test)
        r2 = cov/(s_pred * s_val)
        data[i][f"{m_name}_R2"] = R2
        data[i][f"{m_name}_mse"] = mse
        data[i][f"{m_name}_r2"] = r2.item()

        s = f"PC{i+1}{'\t' if i < 9 else ''}\t{mse:+.3}\t{R2:+.3}\t{r2:+.3}"
        print(s)
    print(list(data[0].keys()))
import pandas as pd

pd.DataFrame(data).to_csv('ml_results.csv', index=False)