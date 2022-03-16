from torch.utils.data import Dataset
from torch.utils.data import TensorDataset
import os
import torch
import json
from PIL import Image
import numpy as np
#import transforms
from torchvision.transforms import functional as F

#import matplotlib.pyplot as plt


class my_DataSet(Dataset):

    def __init__(self, coco_root, transforms,train_set=True ):
        self.transforms = transforms
        self.annotations_root = os.path.join(coco_root,"erase_with_uncertain_dataset/annotations/annotations")

        if train_set:
            self.annotations_json = os.path.join(self.annotations_root,"tiny_set_train.json")         #选择训练集
            self.image_root = os.path.join(coco_root,"erase_with_uncertain_dataset/train")
        else:
            self.annotations_json = os.path.join(self.annotations_root, "tiny_set_test.json")         #选择验证集
            self.image_root = os.path.join(coco_root, "erase_with_uncertain_dataset/test")

        assert os.path.exists(self.annotations_json), "{} file not exist.".format( self.annotations_json)
        assert os.path.exists(self.image_root), "{} file not exist.".format( self.image_root)
        json_file = open(self.annotations_json, 'r')
        self.coco_dict = json.load(json_file)
        self.bbox_image = {}                                                                 
        bbox_img = self.coco_dict["annotations"]            
        for temp in bbox_img:   
            temp_append=[]
            pic_id = temp["image_id"] - 1
            #pic_id = pic_id-1
            bbox = temp["bbox"]                     #这边我用的mask_rcnn的标注，换成faster需要参照json文件改一下
            class_id = temp["category_id"]             #id
            #temp_append.append(pic_id)
            temp_append.append(class_id)                     
            temp_append.append(bbox)              #id和边界框一起存
            if self.bbox_image.__contains__(pic_id): # equals to -> if pic_id in self.bbox_image 
                self.bbox_image[pic_id].append(temp_append)
            else:
                self.bbox_image[pic_id] = []
                self.bbox_image[pic_id].append(temp_append)  




    def __len__(self):
        return len(self.coco_dict["images"])

    def __getitem__(self,idx):                                          
        image_list = self.coco_dict["images"]
        pic_name= image_list[idx]["file_name"]
        pic_path = os.path.join(self.image_root,pic_name)
        image = Image.open(pic_path)
        image = np.array(image)
        bboxes = []      
        labels = []
        target = {}
        if self.bbox_image.__contains__(idx):
            for annotations in self.bbox_image[idx]:
                bboxes.append(annotations[1])
                labels.append(annotations[0])
            bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            target["boxes"] = bboxes
            target["labels"] = labels
        else:
            bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            target["boxes"] = bboxes
            target["labels"] = labels

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target


    def collate_fn(self,batch):
        return tuple(zip(*batch))

class Compose(object):
    """组合多个transform函数"""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class ToTensor(object):
    """将PIL图像转为Tensor"""
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


data_transform = {
    "train": Compose([ToTensor()]),
    "val": Compose([ToTensor()])
}


cocodaset=my_DataSet(r"datasets/tiny_set",data_transform["train"])
#re_image = cocodaset.__getitem__(2859)
#print("re_target",re_target)

dataloader = torch.utils.data.DataLoader(cocodaset, batch_size=2, shuffle=True,
                        collate_fn= cocodaset.collate_fn   )  #lambda batch: tuple(zip(*batch)))
for step, (batch_x, batch_y) in enumerate(dataloader):
    print("steop:{}, batch_x:{}, batch_y:{}".format(step, batch_x, batch_y))
