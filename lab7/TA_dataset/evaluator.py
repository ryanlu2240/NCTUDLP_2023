import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
from torchvision import transforms
import numpy as np
import json

'''===============================================================
1. Title:     

DLP spring 2023 Lab7 classifier

2. Purpose:

For computing the classification accruacy.

3. Details:

The model is based on ResNet18 with only chaning the
last linear layer. The model is trained on iclevr dataset
with 1 to 5 objects and the resolution is the upsampled 
64x64 images from 32x32 images.

It will capture the top k highest accuracy indexes on generated
images and compare them with ground truth labels.

4. How to use

You should call eval(images, labels) and to get total accuracy.
images shape: (batch_size, 3, 64, 64)
labels shape: (batch_size, 24) where labels are one-hot vectors
e.g. [[1,1,0,...,0],[0,1,1,0,...],...]

==============================================================='''


class evaluation_model():
    def __init__(self):
        #modify the path to your own path
        checkpoint = torch.load('./checkpoint.pth')
        # checkpoint = torch.load('./TA_dataset/checkpoint.pth')
        self.resnet18 = models.resnet18(pretrained=False)
        self.resnet18.fc = nn.Sequential(
            nn.Linear(512,24),
            nn.Sigmoid()
        )
        self.resnet18.load_state_dict(checkpoint['model'])
        self.resnet18 = self.resnet18.cuda()
        self.resnet18.eval()
        self.classnum = 24
    def compute_acc(self, out, onehot_labels):
        batch_size = out.size(0)
        acc = 0
        total = 0
        for i in range(batch_size):
            curr_acc = 0
            k = int(onehot_labels[i].sum().item())
            total += k
            outv, outi = out[i].topk(k)
            lv, li = onehot_labels[i].topk(k)
            for j in outi:
                if j in li:
                    acc += 1
                    curr_acc +=1
            # if curr_acc == k:
            #     print(i)
        return acc / total
    def eval(self, images, labels):
        with torch.no_grad():
            #your image shape should be (batch, 3, 64, 64)
            out = self.resnet18(images)
            # print('predict:', out)
            # print('gt:', labels)
            acc = self.compute_acc(out.cpu(), labels.cpu())
            return acc
        

default_transform = transforms.Compose([
    transforms.Resize([64, 64]),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

if __name__ == '__main__':
    model = evaluation_model()
    device = 'cuda'
    #test training data

    obj_data = json.load(open('./objects.json'))
    test_labels = []
    data = json.load(open('./test.json'))
    imgs = []
    # output_img_ema = []
    idx = 0 
    for idx, i in enumerate(data):
        onehot = torch.zeros((1,24))
        for cls in i:
            onehot[0][obj_data[cls]] = 1
        test_labels.append(onehot[0].detach().cpu().numpy())

        img_arr = Image.open('../sampling/test/'+ str(idx) + '.png').convert('RGB')
        img_tensor = default_transform(img_arr)
        img = np.array(img_tensor.float())
        imgs.append(img)
        # output_img_ema.append(ema_sampled_images[0])
    # print(np.shape(imgs))
    test_labels = torch.tensor(test_labels).to(device)
    imgs = torch.tensor(imgs).to(device)

    acc = model.eval(imgs, test_labels)
    print('test_acc:', acc)

    test_labels = []
    data = json.load(open('./new_test.json'))
    imgs = []
    # output_img_ema = []
    idx = 0 
    for idx, i in enumerate(data):
        onehot = torch.zeros((1,24))
        for cls in i:
            onehot[0][obj_data[cls]] = 1
        test_labels.append(onehot[0].detach().cpu().numpy())

        img_arr = Image.open('../sampling/new_test/'+ str(idx) + '.png').convert('RGB')
        img_tensor = default_transform(img_arr)
        img = np.array(img_tensor.float())
        imgs.append(img)
    # print(np.shape(imgs))
    test_labels = torch.tensor(test_labels).to(device)
    imgs = torch.tensor(imgs).to(device)

    acc = model.eval(imgs, test_labels)
    print('new test_acc:', acc)
