import numpy as np
import os
import glob

## setting the path
path_labels = '/home/nuke07/Downloads/gtFine_trainvaltest/gtFine'
path_images = '/home/nuke07/Downloads/leftImg8bit_trainvaltest/leftImg8bit'
dir_list_train = os.listdir(path_labels + '/train')
dir_list_test = os.listdir(path_labels + '/test')
dir_list_val = os.listdir(path_labels + '/val')

img_str = '/**leftImg8bit.png'
lab_str = '/**gtFine_labelIds.png'
inst_str = '/**gtFine_instanceIds.png'
col_str = '/**gtFine_color.png'

## class
class datasetSorter:
    def __init__(self, type):
        self.path_images = path_images
        self.path_labels = path_labels
        self.dir_list_train = os.listdir(path_labels + '/train')
        self.dir_list_test = os.listdir(path_labels + '/test')
        self.dir_list_val = os.listdir(path_labels + '/val')
        self.img_str = img_str
        self.lab_str = lab_str
        self.inst_str = inst_str
        self.col_str = col_str
        self.image_list = self.get_image_list(type)

    def get_image_list(self, type):
        img_dict = dict()
        if type == 'train':
            dirs = self.dir_list_train
        elif type == 'test':
            dirs = self.dir_list_test
        else:
            dirs = self.dir_list_val
        img_list = list()
        lab_list = list()
        for dir in dirs:
            img_path = self.path_images + '/' + type + '/' + dir + self.img_str
            img_list_temp = glob.glob(img_path)
            for lab in img_list_temp:
                lab = lab.replace(self.path_images, self.path_labels)
                lab = lab.replace(self.img_str[3:], self.lab_str[3:])
                lab_list.append(lab)
            img_list += img_list_temp
        img_dict['images'] = img_list
        img_dict['labels'] = lab_list
        return img_dict

    def __getitem__(self, index):
        return self.image_list['images'][index], self.image_list['labels'][index]

    def __len__(self):
        return len(self.image_list['images'])
