# Bruno Martens & Timo Vijn

import os
os.system('clear')

import cv2
import time
import numpy as np
import imutils

directory = "./Training/Images/"
  
for filename in os.listdir(directory):
    img_path = "./Training/Images/{}".format(filename)
    img = cv2.imread(img_path)
    (img_h, img_w) = img.shape[:2]

    filename = filename.replace(".png","")
    print(filename)
    
    file = open('./Training/Labels/{}.txt'.format(filename),'w')

    file.write('1 \n')
    file.write('0 0 ')
    file.write("{} ".format(img_h))
    file.write("{}".format(img_w))

    file.close()
    continue


