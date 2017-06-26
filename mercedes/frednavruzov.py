import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import r2_score
from sklearn.decomposition import PCA, FastICA

# evaluation metric
def the_metric(y_pred, y):
    y_true = y.get_label()
    return 'my_r2', r2_score(y_true, y_pred)

# read datasets
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

# process columns, apply LabelEncoder to categorical features
#for c in train.columns:
#    if train[c].dtype == 'object':
#        lbl = LabelEncoder() 
#        lbl.fit(list(train[c].values) + list(test[c].values)) 
#        train[c] = lbl.transform(list(train[c].values))
#        test[c] = lbl.transform(list(test[c].values))

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
# shape        
print('Shape train: {}\nShape test: {}'.format(train.shape, test.shape))

n_comp = 10

# PCA
pca = PCA(n_components=n_comp, random_state=42)
pca2_results_train = pca.fit_transform(train.drop(["y"], axis=1))
pca2_results_test = pca.transform(test)

# ICA
ica = FastICA(n_components=n_comp, random_state=42)
ica2_results_train = ica.fit_transform(train.drop(["y"], axis=1))
ica2_results_test = ica.transform(test)

# Append decomposition components to datasets
for i in range(1, n_comp+1):
    train['pca_' + str(i)] = pca2_results_train[:,i-1]
    test['pca_' + str(i)] = pca2_results_test[:, i-1]
    
    train['ica_' + str(i)] = ica2_results_train[:,i-1]
    test['ica_' + str(i)] = ica2_results_test[:, i-1]
    
y_train = train["y"]
y_mean = np.mean(y_train)

import xgboost as xgb

# prepare dict of params for xgboost to run with
xgb_params = {
    'n_trees': 500, 
    'eta': 0.005,
    'max_depth': 4,
    'subsample': 0.95,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'base_score': y_mean, # base prediction = mean(target)
    'silent': 1
}

# form DMatrices for Xgboost training
train_size = int(0.80*len(train))
dtrain = xgb.DMatrix( train.drop(['y'], axis=1)[:train_size], label=y_train[:train_size])
deval = xgb.DMatrix( train.drop(['y'], axis=1)[train_size:], label=y_train[train_size:])
dtest = xgb.DMatrix( test)

evallist  = [(deval,'eval'), (dtrain,'train')]
# xgboost, cross-validation
cv_result = xgb.cv(xgb_params, 
                   dtrain, 
                   num_boost_round=1900, # increase to have better results (~700)
                   early_stopping_rounds=50,
                   verbose_eval=50, 
                   show_stdv=False
                  )

num_boost_rounds = len(cv_result)
print(num_boost_rounds)

# train model
model = xgb.train(dict(xgb_params, silent=1), dtrain,num_boost_rounds,evallist, feval=the_metric, verbose_eval=50)


# now fixed, correct calculation
print(r2_score(dtrain.get_label(), model.predict(dtrain)))

#u make predictions and save results
y_pred = model.predict(dtest)
output = pd.DataFrame({'id': test['ID'].astype(np.int32), 'y': y_pred})
output.to_csv('xgboost-depth{}-pca-ica.csv'.format(xgb_params['max_depth']), index=False)
