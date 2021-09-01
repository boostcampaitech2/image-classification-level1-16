import numpy as np
import os
import shutil

from PIL import Image

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

class DataDoubling:
    def __init__(self, image_dir_path):
        self.data_gen = ImageDataGenerator(
                                      rotation_range = 25,
                                      shear_range = 0.5,
                                      horizontal_flip = True,
                                      brightness_range = [0.7, 1.3],
                                      fill_mode = 'nearest')
        
        self.dir_path = image_dir_path
        
    #한명의 사람들 dir 들어가서 incorrect_mask, normal 두개의 데이터를 가지고 doubling 작업해서 해당 dir에 저장시킨다.


    def copy_dir(self):
        shutil.copytree(self.dir_path, '/opt/ml/input/data/train/images2')
        print('copy complete')
        


    def data_doubling(self):
        profiles = os.listdir(self.dir_path)
        masks = ['mask1.jpg','mask2.jpg','mask3.jpg','mask4.jpg','mask5.jpg']
        for profile in profiles:
            if profile.startswith("."):
                continue
#            print(profile)
            img_folder = os.path.join(self.dir_path, profile)
            for file_name in os.listdir(img_folder):
                _file_name, ext = os.path.splitext(file_name)
                if _file_name.startswith("."):
                    continue
                if file_name in masks:
                    continue
                
                img_path = os.path.join(self.dir_path, profile, file_name)
                img = load_img(img_path)
                x = img_to_array(img)
                x = x.reshape((1,) + x.shape)
                
                idx = 0
                
                for batch in self.data_gen.flow(x, save_to_dir = os.path.join(self.dir_path, profile), save_prefix = _file_name + str(idx) , save_format = 'jpg'):
                    idx += 1
                    if idx > 3:
                        break
        print('doubling completed!')
                    