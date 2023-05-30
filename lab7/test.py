import os
import copy
import numpy as np
import json
import torch
import argparse
import torch.nn as nn
from tqdm import tqdm
from torch import optim
from utils import *
from modules import UNet_conditional, EMA
import logging
from torch.utils.tensorboard import SummaryWriter
from loader import load_train_data
from TA_dataset.evaluator import  evaluation_model
from torchvision import transforms

transform = transforms.Compose([
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


def parse_args():
    parser = argparse.ArgumentParser()

	## What to do
    parser.add_argument("--train", default=False, action="store_true")
    parser.add_argument("--test" , default=False, action="store_true")

	## Hyper-parameters

    parser.add_argument("--seed"      , default=1     , type=int,    help="manual seed")
    parser.add_argument("--num_workers"      , default=4     , type=int)
    parser.add_argument("--batch_size"      , default=2     , type=int)
    parser.add_argument("--lr"        , default=3e-4 , type=float,  help="learning rate")
    parser.add_argument("--device"    , type=str      , default="cuda:0")
    parser.add_argument("--dataset_root", type=str, default='./iclevr/')
    parser.add_argument("--train_data_json", type=str, default='./TA_dataset/train.json')
    parser.add_argument("--test_data_json", type=str, default='./TA_dataset/test.json')
    parser.add_argument("--newtest_data_json", type=str, default='./TA_dataset/new_test.json')
    parser.add_argument("--objects_json", type=str, default='./TA_dataset/objects.json')
    parser.add_argument("--num_classes"      , default=24     , type=int)

    parser.add_argument("--epochs"      , default=300     , type=int)
    
    
    args = parser.parse_args()
    return args


class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=(240, 320), device="cuda:0"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.img_size = img_size
        self.device = device

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n, labels, cfg_scale=3):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size[0], self.img_size[1])).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t, labels)
                if cfg_scale > 0:
                    uncond_predicted_noise = model(x, t, None)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x
    
if __name__ == '__main__':
    
    args = parse_args()
    # test_idx = [6, 11, 12, 13, 14, 15, 26, 29, 30]
    # new_test_idx = [0, 1, 2, 4, 6, 7, 8, 10, 11, 25, 26, 27]
    device = args.device
    # evaluator = evaluation_model()
    diffusion = Diffusion(img_size=(64, 64), device=device)
    model = UNet_conditional(num_classes=args.num_classes, device=device).to(device)
    checkpoint = torch.load('./models/epoch/500_ckpt64.pt')
    model.load_state_dict(checkpoint)

    test_labels = []
    obj_data = json.load(open(args.objects_json))
    data = json.load(open(args.test_data_json))
    output_img = []
    # output_img_ema = []
    idx = 0 
    for idx, i in enumerate(tqdm(data)):
        # if idx in test_idx:
        onehot = torch.zeros((1,24))
        for cls in i:
            onehot[0][obj_data[cls]] = 1
        # test_labels.append(onehot[0].detach().cpu().numpy())

        onehot = onehot.float().to(device)
        sampled_images = diffusion.sample(model, n=len(onehot), labels=onehot)
        arr_img = sampled_images[0].detach().cpu().numpy()
        # print(arr_img.shape)
        im = Image.fromarray(arr_img.transpose(1, 2, 0))
        im.save(f"./sampling/test/{idx}.png")
        print(f"./sampling/test/{idx}.png")


    obj_data = json.load(open(args.objects_json))
    data = json.load(open(args.newtest_data_json))
    for idx, i in enumerate(tqdm(data)):
        # if idx in new_test_idx:
        onehot = torch.zeros((1, 24))
        for cls in i:
            onehot[0][obj_data[cls]] = 1
        # newtest_labels.append(onehot[0].detach().cpu().numpy())

        onehot = onehot.float().to(device)
        sampled_images = diffusion.sample(model, n=len(onehot), labels=onehot)
        arr_img = sampled_images[0].detach().cpu().numpy()
        im = Image.fromarray(arr_img.transpose(1, 2, 0))
        im.save(f"./sampling/new_test/{idx}.png")
        print(f"./sampling/new_test/{idx}.png")

