import re
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
import pickle as pk
import gzip

# evaluation metric
def the_metric(y_pred, y):
    y_true = y
    return 'my_r2', r2_score(y_true, y_pred)
try:
    t_train,t_test,label_train,label_val = pk.load(gzip.open('data/train_test_data.zip', 'rb'))
    print 'Compressed Pickle loading done'
except:
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    ####################3 Remove the columns whis has only 0 #####################
    #singularity_train = map(lambda x: {x: len(set(train[x]))},train.keys())
    #keys_with_single_train = filter(lambda x: list(x.values())[0]==1, list(singularity_train))
    #keys_with_single_train = map(lambda x: x.keys()[0], keys_with_single_train)
    #for i in keys_with_single_train:
    #    del train[i]
    #    del test[i]
    ##del train['X4'], train['X5'], test['X4'], test['X5']
    ####################3 Remove the columns whis has only 0 #####################
    
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
    
    train[d_t_str_keys] = train[d_t_str_keys] / np.linalg.norm(train[d_t_str_keys]) # normalize values
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
    p_traini_all = np.concatenate((p_train,train[d_t_str_keys]), axis=1)
    
    p_test = pca.fit_transform(test_int_data)
    p_test_all = np.concatenate((p_test,test[d_t_str_keys]), axis=1)
    
    train_size = int(0.90*len(p_train))
    dtrain = p_train[:train_size]
    label_train=y_train[:train_size]
    deval = p_train[train_size:]
    label_val=y_train[train_size:]
    
    t_train = np.array(train.drop(['y'], axis=1))
    #t_train = np.concatenate((t_train,p_train), axis=1)
    print (t_train.shape, train.shape)
    t_test = np.array(test)
    #t_test = np.concatenate((t_test, p_test), axis=1)
    try:
        pk.dump([t_train,t_test,label_train,label_val], gzip.open('data/train_test_data.zip', 'wb'), -1)
        print 'Compressed pickling done'
    except Exception as e:
        print e, 'pickling error'
        pass
model = ma.fcnn(t_train.shape[1])
checkpoint = ModelCheckpoint('weights/best.weights', monitor='val_loss', save_best_only=True, verbose=2)

train_size = int(0.90*len(t_train))
history = model.fit(t_train[:train_size], label_train, validation_data=(t_train[train_size:], label_val), nb_epoch=400, batch_size=8, verbose=1, callbacks=[checkpoint])

model_best = ma.fcnn(t_train.shape[1])
model_best.load_weights('weights/best.weights')
result = model.predict(t_train[train_size:])

r2_score = the_metric(result, label_val)
y_pred_best = model_best.predict(t_test)
y_pred = model.predict(t_test)

print r2_score, y_pred_best.shape, test.shape
ID = test['ID']
dic = {'ID':ID, 'y':y_pred_best.reshape((len(test)))}
df = pd.DataFrame(dic, columns=["ID",'y'])
df.to_csv('data/nn_submission_best.csv', index=False)

dic = {'ID':ID, 'y':y_pred.reshape((len(test)))}
df = pd.DataFrame(dic, columns=["ID",'y'])
df.to_csv('data/nn_submission.csv', index=False)
#########################3 PLOT loss and accuracy ##############
print(history.history.keys())

# summarize history
pars = history.history.keys()
for par in pars:
    if 'val_' in par:
        act_par = re.findall('val_(.*)', par)[0]
        plt.plot(history.history[act_par])
        plt.plot(history.history[par])
        plt.title('model '+act_par)
        plt.ylabel(act_par)
        plt.xlabel('epoch')
        plt.legend(['train', 'validate'], loc='upper right')
        # plt.show()
        plt.savefig(str(act_par+'.png'))
        plt.cla()
        #plt.close()
#########################3 PLOT loss and accuracy ##############
