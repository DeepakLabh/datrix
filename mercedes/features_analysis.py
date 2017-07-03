import pandas as pd
from collections import Counter
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')
#data_types = map(lambda x: {x: type(train_df[x][0])},train_df.keys())
#print (list(data_types))
singularity_train = map(lambda x: {x: len(set(train_df[x]))},train_df.keys())
singularity_test = map(lambda x: {x: len(set(test_df[x]))},test_df.keys())
#print (list(singularity_train))
keys_with_single_train = filter(lambda x: list(x.values())[0]==1, list(singularity_train))
keys_with_single_test = filter(lambda x: list(x.values())[0]==1, list(singularity_test))
print ('these Keys should be ignored:- ',list(keys_with_single_train), list(keys_with_single_test))
y_counter = Counter(train_df['y'])
sorted_y_counter = sorted(y_counter, key=y_counter.get, reverse=True)
print (sorted_y_counter[:10], y_counter[sorted_y_counter[0]])


