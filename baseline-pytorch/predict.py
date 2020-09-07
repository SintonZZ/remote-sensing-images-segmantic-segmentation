from model import unet
import cv2
import numpy as np
import os
import torch
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
from tqdm import tqdm
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'

matches = [100, 200, 300, 400, 500, 600, 700, 800]

input_path = './test_A/'
output_path = './results/'


class TestDataset(Dataset):
    def __init__(self, input_path, transform):
        self.filename = os.listdir(input_path)
        self.transform = transform

    def __getitem__(self, index):
        image_path = os.path.join(input_path, self.filename[index])
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        img_index = os.path.splitext(self.filename[index])[0]
        return self.transform(img), img_index  # 返回单张测试图片和其对应的文件名编号， 编号用于保存预测出的标签图

    def __len__(self):
        return len(self.filename)



def predict(weights_path):
    model = unet(num_classes=len(matches)).cuda()
    model.load_state_dict(torch.load(weights_path))
    model.eval()
    with torch.no_grad():
        for input, img_index in tqdm(test_loader):   # type(img_index): list
            input = input.cuda()
            outputs = model(input)
            outputs = outputs.data.cpu().numpy() # outputs： (batch_size, 8,256,256)

            for idx, output in enumerate(outputs):
                output = output.argmax(axis=0)  # (256,256)
                save_img = np.zeros((256, 256), dtype=np.uint16)
                for i in range(256):
                    for j in range(256):
                        save_img[i][j] = matches[int(output[i][j])]
                cv2.imwrite(os.path.join(output_path, img_index[idx] + '.png'), save_img)

if __name__ == "__main__":

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_loader = torch.utils.data.DataLoader(
        TestDataset(input_path, transform),
        batch_size=16, shuffle=False, num_workers=8, pin_memory=True
    )

    weights_path = './baseline.pt'
    predict(weights_path=weights_path)



