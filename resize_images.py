

import cv2
from PIL import Image
import torchvision.transforms as transforms
import os
import numpy as np

def generate_new_dataset():

    # root path depends on your computer
    # for test: "../datasets/celeba_small/celeba/img_align_celeba/"
    root =   "../datasets/celeba/img_align_celeba/" #"../datasets/celeba/img_align_celeba/" #'data/celebA/celebA/'
    save_root =  "../datasets/resized_celebA3/"  #'data/resized_celebA/'


    if not os.path.isdir(save_root):
        os.mkdir(save_root)
    if not os.path.isdir(save_root + 'celebA'):
        os.mkdir(save_root + 'celebA')
    img_list = os.listdir(root)

    #print("img_list", img_list)

    # ten_percent = len(img_list) // 10

    img_size=128
    img_resize=64
    x=25
    y=45

    if not False:
        for i in range(len(img_list)):
            #img = plt.imread(root + img_list[i])
            #print(type(img))
            #img = imresize(img, (resize_size, resize_size))
            #img = Image.fromarray(img).resize(size=(resize_size, resize_size))
            #plt.imsave(fname=save_root + 'celebA/' + img_list[i], arr=img)

            img = cv2.imread(root + img_list[i])
            img = img[y:y + img_size, x:x + img_size]
            img = cv2.resize(img, (img_resize, img_resize))
            #img = np.array(img, dtype='float32')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            #img /= 255.0  # Normalization to [0.,1.]
            new_image = Image.fromarray(img)
            new_image.save(save_root + 'celebA/' + img_list[i])

            if i % 1000 == 0: #(i % 10) == 0:
                print('%d images complete' % i)


    if not True:
        for i in range(len(img_list)):
            img = Image.open(root + img_list[i])

            w, h = img.size
            size = min(w, h)
            # print(w, h)
            # print(size)

            transform = transforms.CenterCrop(size)
            new_image = transform(img)

            new_image = new_image.resize((64, 64))

            new_image.save(save_root + 'celebA/' + img_list[i])
            if i == 20: #(i % 10) == 0:
                print('%d images complete' % i)
                break


generate_new_dataset()