# Bruno Martens & Timo Vijn

import os
os.system('clear')

import cv2
import time
import numpy as np
import imutils

directory = "./Training/Images/Test/"
  
for filename in os.listdir(directory):
    img_path = "./Training/Images/Test/{}".format(filename)
    img = cv2.imread(img_path)
    (img_h, img_w) = img.shape[:2]

    filename = filename.replace(".png","")
    print(filename)
    
    label_file = open('./Training/Labels/Test/{}.txt'.format(filename),'w')

    label_file.write('0 ')
    label_file.write('0.5 0.5 ')
    label_file.write('1 1')

    # file.write("{} ".format(img_h))
    # file.write("{}".format(img_w))

    label_file.close()

    list_file = open('./Training/Labels/Test/Test.txt', 'a')
    
    list_file.write('{} \n'.format(img_path))
    
    list_file.close()

    continue