import os
import cv2
import numpy as np
from paddle.io import Dataset


class Place365Dataset256(Dataset):
    """
    步骤一：继承 paddle.io.Dataset 类
    """

    def __init__(self, root_dir, mode, transform=None):
        """
        步骤二：实现 __init__ 函数，初始化数据集，将样本和标签映射到列表中
        """
        super(Place365Dataset256, self).__init__()
        catetgory_file_path = os.path.join(root_dir, 'filelist', 'categories_places365.txt')
        self.id2categories_name = dict()
        # 读标签
        with open(catetgory_file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                category_name, id = line.split(' ')
                id = int(id)
                category_name = category_name[3:]
                self.id2categories_name[id] = category_name
        # 读数据
        self.data_path_list = []
        self.label_list = []

        if mode=='train':
            data_txt='places365_train_standard.txt'
            data_root='train/data_256'
        elif  mode=='val':
            data_txt='places365_train_standard.txt'
            data_root='train/data_256'
        else:
            raise 'Error mode! (support: train or val)'

        data_txt_path = os.path.join(root_dir, data_root, data_txt)
        with open(data_txt_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                data_path, category_id = line.split(' ')


        self.data_list = []
        with open(label_path, encoding='utf-8') as f:
            for line in f.readlines():
                image_path, label = line.strip().split('\t')
                image_path = os.path.join(data_dir, image_path)
                self.data_list.append([image_path, label])
        # 2. 传入定义好的数据处理方法，作为自定义数据集类的一个属性
        self.transform = transform

    def __getitem__(self, index):
        """
        步骤三：实现 __getitem__ 函数，定义指定 index 时如何获取数据，并返回单条数据（样本数据、对应的标签）
        """
        image_path, label = self.data_list[index]
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = image.astype('float32')
        # 3. 应用数据处理方法到图像上
        if self.transform is not None:
            image = self.transform(image)
        label = int(label)
        return image, label

    def __len__(self):
        """
        步骤四：实现 __len__ 函数，返回数据集的样本总数
        """
        return len(self.data_list)


if __name__ == '__main__':
    datasets = Place365Dataset256('/home/public/datasets/place365/')
    # # 1. 定义随机旋转和改变图片大小的数据处理方法
    # from paddle.vision.transforms import Compose, RandomRotation
    #
    # # 定义待使用的数据处理方法，这里包括随机旋转、改变图片大小两个组合处理
    # from paddle.vision.transforms import Resize
    #
    # transform = Compose([RandomRotation(10), Resize(size=32)])
    #
    # custom_dataset = Place365Dataset('mnist/train', 'mnist/train/label.txt', transform)
