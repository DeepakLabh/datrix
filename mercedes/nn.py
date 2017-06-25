import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import seaborn as sb
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import normalize 
from sklearn.decomposition import PCA
from copy import deepcopy
from sklearn.metrics import r2_score
from keras.callbacks import ModelCheckpoint
import model_arch as ma

# evaluation metric
def the_metric(y_pred, y):
    y_true = y
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


# apply PCA 
pca = PCA(n_components=10, svd_solver='full')

y_train = train['y']
y_mean = y_train.mean()
p_train = pca.fit_transform(train_int_data)
p_train = np.concatenate((p_train,train[d_t_str_keys]), axis=1)

p_test = pca.fit_transform(test_int_data)
p_test = np.concatenate((p_test,test[d_t_str_keys]), axis=1)

train_size = int(0.9*len(p_train))
dtrain = p_train[:train_size]
label_train=y_train[:train_size]
deval = p_train[train_size:]
label_val=y_train[train_size:]

print (p_train.shape)
model = ma.fcnn(p_train.shape[1])
history = model.fit(dtrain, label_train, validation_data=(deval, label_val), nb_epoch=100, batch_size=8, verbose=1)

result = model.predict(deval)

r2_score = the_metric(result, label_val)
print r2_score