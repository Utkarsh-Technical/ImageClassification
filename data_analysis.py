import os
from cv2 import data
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle


dir = ''

categories = ['cat', 'dog']

for c in categories:
    path = os.path.join(dir,c)
    label = categories.index(c)

    for img in os.lostdir(path):
        imgpath = os.path.join(path,img)
        try:
            pet_img = cv2.imread(imgpath,0)
            cv2.resize(pet_img,(100,100))
            image = np.array(pet_img).flatten()
            data.append([image,label])
        except Exception as e:
            pass

print(len(data))

pick_in = open('data1.pickle','wb')
pickle.dump(data,pick_in)
pick_in.close()