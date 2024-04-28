import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np


data_dict = pickle.load(open('./data.pickle', 'rb'))

data_list = np.asarray(data_dict['data'])

data = []
for  i in data_list:
    while(len(i)!=84):
        i.append(0) 
    data.append(i)

# for i in data:
#     print(i)
# print("____________________________________________")

labels = np.asarray(data_dict['labels'])[:len(data)]
# print(data)
# print(labels)














# x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=5, shuffle=True, stratify=labels)
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# print(data)
# print(x_train.shape)
# print("---------------------------------------")
# print(labels)











model = RandomForestClassifier()

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly !'.format(score * 100))

f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
# f.close()
