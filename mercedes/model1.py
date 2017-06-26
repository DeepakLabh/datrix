import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import seaborn as sb
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import normalize 
from sklearn.decomposition import PCA, FastICA
from copy import deepcopy
from sklearn.metrics import r2_score

# evaluation metric
def the_metric(y_pred, y):
    y_true = y.get_label()
    return 'my_r2', r2_score(y_true, y_pred)

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
unique_str_vals = sorted(list(set(list(set(data_str_test.values.ravel()))+ list(set(data_str_train.values.ravel())))))
lbl = LabelEncoder() 
lbl.fit(unique_str_vals)

for i in d_t_str_keys:
    train[i] = lbl.transform(train[i]) 

train[d_t_str_keys] = train[d_t_str_keys] / np.linalg.norm(train[d_t_str_keys])
train= train.iloc[np.random.permutation(len(train))]
train_int_data = train[d_int_keys]
test_int_data = test[d_int_keys]

for i in d_t_str_keys:
    test[i] = lbl.transform(test[i]) 
test[d_t_str_keys] = test[d_t_str_keys] / np.linalg.norm(test[d_t_str_keys])

n_comp = 12
# apply PCA 
pca = PCA(n_components=n_comp, svd_solver='full')
pca = PCA(n_components=n_comp, random_state=42)
ica = FastICA(n_components=n_comp, random_state=42)

y_train = train['y']
y_mean = y_train.mean()
p_train = pca.fit_transform(train_int_data)
ica2_train = ica.fit_transform(train.drop(["y"], axis=1))
p_train = np.concatenate((p_train,train[d_t_str_keys], ica2_train), axis=1)

p_test = pca.fit_transform(test_int_data)
ica2_test = ica.transform(test)
p_test = np.concatenate((p_test,test[d_t_str_keys], ica2_test), axis=1)

train_size = int(0.80*len(p_train))
dtrain = xgb.DMatrix( p_train[:train_size], label=y_train[:train_size])
deval = xgb.DMatrix( p_train[train_size:], label=y_train[train_size:])
dtest = xgb.DMatrix( p_test)

#dtrain = xgb.DMatrix( p_train, label=y_train)
#param = {'max_depth':10, 'eta':1, 'silent':1, 'objective':'reg:linear' }
# prepare dict of params for xgboost to run with
xgb_params = {
    'n_trees': 500, 
    'eta': 0.01,
    'max_depth': 5,
    'subsample': 0.55,
    'colsample_bytree': 0.6,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'base_score': y_mean,
    'silent': 1,
    'nthread': 4
}

evallist  = [(deval,'eval'), (dtrain,'train')]

# xgboost, cross-validation
#cv_result = xgb.cv(xgb_params, 
#                   dtrain, 
#                   num_boost_round=2000, # increase to have better results (~700)
#                   nfold = 3,
#                   early_stopping_rounds=50,
#                   feval=the_metric,
#                   verbose_eval=100, 
#                   show_stdv=True
#                  )
#
num_boost_rounds = 3500#len(cv_result)
print('num_boost_rounds=' + str(num_boost_rounds))

# train model
model = xgb.train(dict(xgb_params, silent=1), dtrain, num_boost_rounds, evallist, feval=the_metric)

# Predict on trian and test
y_train_pred = model.predict(dtrain)
y_pred = model.predict(dtest)

print('First 5 predicted test values:', y_pred[:5])
#plst = xgb_params.items()
#
#num_round = 2000
#bst = xgb.train( plst, dtrain, num_round, evallist, early_stopping_rounds=10 )
#bst.save_model('models/0001.model')
#
#ypred = bst.predict(dtest)
ID = test['ID']

dic = {'ID':ID, 'y':y_pred}
df = pd.DataFrame(dic, columns=["ID",'y'])
df.to_csv('data/Sample_submission_1.csv', index=False)
