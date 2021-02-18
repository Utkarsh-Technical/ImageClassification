import os
from cv2 import data
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import random
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# dir = ''

# categories = ['cat', 'dog']

# for c in categories:
#     path = os.path.join(dir,c)
#     label = categories.index(c)

#     for img in os.lostdir(path):
#         imgpath = os.path.join(path,img)
#         try:
#             pet_img = cv2.imread(imgpath,0)
#             cv2.resize(pet_img,(100,100))
#             image = np.array(pet_img).flatten()
#             data.append([image,label])
#         except Exception as e:
#             pass

# print(len(data))

# pick_in = open('data1.pickle','wb')
# pickle.dump(data,pick_in)
# pick_in.close()

pick_in = open('data1.pickle','r')
pickle.load(pick_in)
pick_in.close()

random.shuffle(data)
features = []
labels =[]

for fe,la in data:
    features.append(fe)
    labels.append(la)

xtrain, xtest, ytrain, ytest = train_test_split(features,labels,test_size=0.25)

model = SVC(C=1,kernel='poly',gamma='auto')
model.fit(xtrain, ytrain)

predict = model.predict(xtest)
accuracy = model.score(xtest, ytest)
categories = ['cat', 'dog']

print("Accuracy = ",accuracy)
print('Prediction is: ',categories[predict[0]])

mypet = xtest[0].reshape(50,50)
plt.show(mypet, cmap='gray')
plt.show()