import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import seaborn as sb
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import normalize 
from sklearn.decomposition import PCA
from copy import deepcopy

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
d_t = map(lambda x: {x:type(train[x][0])}, train)
d_t_str = filter(lambda x: x.values()[0]==str, d_t)

d_t_str_keys = map(lambda x: x.keys()[0], d_t_str)
d_int_keys = list(test.keys())
d_int_keys = list(set(d_int_keys).difference(set(d_t_str_keys)))

data_str_test = test[d_t_str_keys]
data_str_train = train[d_t_str_keys]
#data_str.append(test[d_t_str_keys])
unique_str_vals = list(set(list(set(data_str_test.values.ravel()))+ list(set(data_str_train.values.ravel()))))
lbl = LabelEncoder() 
lbl.fit(unique_str_vals)

for i in d_t_str_keys:
    train[i] = lbl.transform(train[i]) 
train[d_t_str_keys] = train[d_t_str_keys] / np.linalg.norm(train[d_t_str_keys])
train= train.iloc[np.random.permutation(len(train))]

for i in d_t_str_keys:
    test[i] = lbl.transform(test[i]) 
test[d_t_str_keys] = test[d_t_str_keys] / np.linalg.norm(test[d_t_str_keys])


# apply PCA 
pca = PCA(n_components=10, svd_solver='full')

y_train = train['y']
p_train = pca.fit_transform(train.drop(['y'], axis=1))
p_test = pca.fit_transform(test)

train_size = int(0.8*len(p_train))
dtrain = xgb.DMatrix( p_train[:train_size], label=y_train[:train_size])
deval = xgb.DMatrix( p_train[train_size:], label=y_train[train_size:])
dtest = xgb.DMatrix( p_test)

param = {'max_depth':10, 'eta':1, 'silent':1, 'objective':'reg:linear' }
param['nthread'] = 4
#param['eval_metric'] = 'auc'
plst = param.items()

evallist  = [(deval,'eval'), (dtrain,'train')]
num_round = 2000
bst = xgb.train( plst, dtrain, num_round, evallist, early_stopping_rounds=10 )
bst.save_model('model/0001.model')

ypred = bst.predict(dtest)
ID = test['ID']

dic = {'ID':ID, 'y':ypred}
df = pd.DataFrame(dic, columns=["ID",'y'])
df.to_csv('data/Sample_submission_1.csv', index=False)
