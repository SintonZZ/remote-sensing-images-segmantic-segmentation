import os
import numpy as np
import random
import torch
import cv2
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset

matches = [100, 200, 300, 400, 500, 600, 700, 800]
images_path = './train/image/'
labels_path = './train/label/'
img_name_list = os.listdir(images_path)
label_name_list = os.listdir(labels_path)
training_samples = int(len(img_name_list) * 0.99)

def get_img_label_paths(images_path, labels_path):
    res = []
    for dir_entry in os.listdir(images_path):
        if os.path.isfile(os.path.join(images_path, dir_entry)):
            file_name, _ = os.path.splitext(dir_entry)
            res.append((os.path.join(images_path, file_name+".tif"),
                        os.path.join(labels_path, file_name+".png")))
    return res


img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


class MaskToTensor(object):
    def __call__(self, mask):
        return torch.from_numpy(np.array(mask, dtype=np.int32)).long()


label_transform = MaskToTensor()


class RSDataset(Dataset):
    def __init__(self, img_label_pairs, img_transform, label_transform, train=True):
        train_img_label_pairs = img_label_pairs[:training_samples]
        val_img_label_pairs = img_label_pairs[training_samples:]

        if train:
            self.img_label_path = train_img_label_pairs
        else:
            self.img_label_path = val_img_label_pairs

        self.img_transform = img_transform
        self.label_transform = label_transform

    def __getitem__(self, index):
        img = cv2.imread(self.img_label_path[index][0], cv2.IMREAD_UNCHANGED)
        label = cv2.imread(self.img_label_path[index][1], cv2.IMREAD_UNCHANGED)

        for m in matches:
            label[label == m] = matches.index(m)
        '''
        pytorch中使用CrossEntropyLoss时不需要进行one-hot编码
        nn.CrossEntropyLoss()函数内部会将数值型label转换为one-hot编码后的label
        seg_labels = np.zeros((256, 256, nClasses))
        for c in range(nClasses):
            seg_labels[:, :, c] = (label == c).astype(int)
        '''

        return self.img_transform(img), self.label_transform(label)

    def __len__(self):
        return len(self.img_label_path)

img_label_pairs = get_img_label_paths(images_path, labels_path)
random.shuffle(img_label_pairs)

train_loader = torch.utils.data.DataLoader(
    RSDataset(img_label_pairs, img_transform, label_transform, train=True),
    batch_size=8, shuffle=True, num_workers=0, pin_memory=True
)

val_loader = torch.utils.data.DataLoader(
    RSDataset(img_label_pairs, img_transform, label_transform, train=False),
    batch_size=8, shuffle=False, num_workers=0, pin_memory=True
)
