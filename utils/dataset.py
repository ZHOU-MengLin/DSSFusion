# coding:utf-8
import os
import torch
# from torch.utils.data import DataLoader
import numpy as np
import cv2
import glob
import os.path as osp
from utils.transform import *
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.transforms import Resize, ToTensor, Compose
from PIL import Image, ImageFile
from arguments import args


ImageFile.LOAD_TRUNCATED_IMAGES = True
class MSRS(Dataset):
    def __init__(
            self,
            rootpth,
            cropsize=(512, 384),
            mode='train',
            Method='SDNet',
            *args,
            **kwargs
    ):
        super(MSRS, self).__init__()
        assert mode in ('train', 'val', 'test')
        self.mode = mode
        self.ignore_lb = 255
        self.filenames = []
        self.img_suffix = '.png'
        self.label_suffix = '.png'
        source_name = os.path.join(rootpth, self.mode + '.txt')
        with open(source_name) as f:
            files = f.readlines()
            for item in files:
                file_name = item.strip()
                self.filenames.append(file_name)
        # parse img directory
        self.im_dir = os.path.join(rootpth, Method)
        self.label_dir = os.path.join(rootpth, 'Label')

        self.len = len(self.filenames)
        # pre-processing
        self.to_tensor = transforms.Compose(
            [
                transforms.ToTensor(),
                # transforms.Normalize((0.234, 0.267, 0.231), (0.177, 0.178, 0.176)),
            ]
        )
        self.trans_train = Compose(
            [
                ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                HorizontalFlip(),
                RandomScale((0.75, 1.0, 1.25, 1.5, 1.75, 2.0)),
                RandomCrop(cropsize),
            ]
        )
        print('Total number of images: {}, Total number of labels: {}'.format(self.len, len(self.filenames)))

    def __getitem__(self, idx):
        img_fn = self.filenames[idx] + self.img_suffix
        label_fn = self.filenames[idx] + self.label_suffix
        impth = os.path.join(self.im_dir, img_fn)
        lbpth = os.path.join(self.label_dir, label_fn)
        img = Image.open(impth)
        label = Image.open(lbpth).convert('L')
        if self.mode == 'train':
            im_lb = dict(im=img, lb=label)
            im_lb = self.trans_train(im_lb)
            img, label = im_lb['im'], im_lb['lb']
        img = self.to_tensor(img)

        # label = np.array(label).astype(np.int64)[np.newaxis, :]
        label = np.array(label).astype(np.int64)
        if label.ndim == 4 and label.shape[-1] == 3:
            label = np.squeeze(label)
            label = label[..., 0]
        elif label.ndim == 3 and label.shape[-1] == 3:
            label = label[..., 0]
        label = torch.from_numpy(label).unsqueeze(0)
        # print(img.shape, label.shape)

        return img, label, img_fn

    def __len__(self):
        return self.len

    def convert_labels(self, label):
        for k, v in self.lb_map.items():
            label[label == k] = v
        return label

class CityScapes(Dataset):
    def __init__(self, num, cropsize=(512, 384), mode='train'):
        super(CityScapes, self).__init__()
        assert mode in ('train', 'val', 'test')
        self.mode = mode
        self.ignore_lb = 255

        # parse img directory
        self.imgs = {}
        imgnames = []
        fusion_dir = f'Dataset/Fusion/train/MSRS_black_{num}'
        im_names = os.listdir(fusion_dir)
        names = [el.replace('.png', '') for el in im_names]
        impths = [osp.join(fusion_dir, el) for el in im_names]
        imgnames.extend(names)
        self.imgs.update(dict(zip(names, impths)))

        # parse gt directory
        self.labels = {}
        gtnames = []
        gtpth = osp.join(args.lab_path, mode)
        # gtpth = args.lab_path

        lbnames = os.listdir(gtpth)
        # lbnames = [el for el in lbnames if 'labelIds' in el]
        names = [el.replace('.png', '') for el in lbnames]
        lbpths = [osp.join(gtpth, el) for el in lbnames]
        gtnames.extend(names)
        self.labels.update(dict(zip(names, lbpths)))

        self.imnames = imgnames
        self.len = len(self.imnames)
        print(set(imgnames) - set(gtnames))  # Images without ground truth
        print(set(gtnames) - set(imgnames))  # Ground truths without images
        assert set(imgnames) == set(gtnames)
        assert set(self.imnames) == set(self.imgs.keys())
        assert set(self.imnames) == set(self.labels.keys())

        # pre-processing
        self.to_tensor = transforms.Compose(
            [
                transforms.ToTensor(),
                # transforms.Normalize(
                #     (0.2254, 0.2585, 0.2225), (0.0521, 0.0529, 0.0546)
                # ),
            ]
        )
        self.trans_train = Compose([ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                                    HorizontalFlip(),
                                    RandomScale((0.75, 1.0, 1.25, 1.5, 1.75, 2.0)),
                                    RandomCrop(cropsize)
                                    ])
    def __getitem__(self, idx):
        fn = self.imnames[idx]
        impth = self.imgs[fn]
        lbpth = self.labels[fn]
        img = Image.open(impth)
        label = Image.open(lbpth).convert('L')
        if self.mode == 'train':
            im_lb = dict(im=img, lb=label)
            im_lb = self.trans_train(im_lb)
            img, label = im_lb['im'], im_lb['lb']
        # test_image = np.array(img)
        img = self.to_tensor(img)
        label = np.array(label).astype(np.int64)
        if label.ndim == 4 and label.shape[-1] == 3:
            label = np.squeeze(label)  # 结果是 (H, W, 3)
            label = label[..., 0]  # 取第一个通道，变为 (H, W)
        elif label.ndim == 3 and label.shape[-1] == 3:
            label = label[..., 0]  # 变为 (H, W)
        label = torch.from_numpy(label).unsqueeze(0)  #变成 (1, H, W)
        # print(label.shape)
        return img, label, fn

    def __len__(self):
        return self.len

def prepare_data_path(dataset_path):
    filenames = os.listdir(dataset_path)
    data_dir = dataset_path
    datapath = glob.glob(os.path.join(data_dir, "*.bmp"))
    datapath.extend(glob.glob(os.path.join(data_dir, "*.tif")))
    datapath.extend(glob.glob((os.path.join(data_dir, "*.jpg"))))
    datapath.extend(glob.glob((os.path.join(data_dir, "*.png"))))
    datapath.sort()
    filenames.sort()
    return datapath, filenames



class Fusion_dataset(Dataset):
    def __init__(self, split, ir_path=None, vi_path=None, la_path=None):
        super(Fusion_dataset, self).__init__()
        if split == 'train':
            # M3DF
            self.vis_paths, self.vis_names = prepare_data_path(os.path.join(vi_path, split))
            self.ir_paths, self.ir_names = prepare_data_path(os.path.join(ir_path, split))
            self.label_paths, self.label_names = prepare_data_path(os.path.join(la_path, split))

            # FBM
            # self.vis_paths, self.vis_names = prepare_data_path(os.path.join(vi_path))
            # self.ir_paths, self.ir_names = prepare_data_path(os.path.join(ir_path))
            # self.label_paths, self.label_names = prepare_data_path(os.path.join(la_path))

            self.split = split
            self.length = min(len(self.vis_names), len(self.ir_names))


        elif split == 'test':
            self.vis_paths, self.vis_names = prepare_data_path(os.path.join(vi_path))
            self.ir_paths, self.ir_names = prepare_data_path(os.path.join(ir_path))
            self.split = split
            self.length = min(len(self.vis_names), len(self.ir_names))



    def __getitem__(self, index):
        if self.split == 'train':
            vis_path = self.vis_paths[index]
            ir_path = self.ir_paths[index]
            label_path = self.label_paths[index]

            image_inf = cv2.imread(ir_path, 0)
            image_inf = cv2.resize(image_inf, (512, 384))
            image_vis = cv2.cvtColor(cv2.imread(vis_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            image_vis = cv2.resize(image_vis, (512, 384))
            label = cv2.imread(label_path, 0)
            label = cv2.resize(label, (512, 384))

            image_vis = image_vis.astype(np.float32).transpose((2, 0, 1)) / 255.0
            image_inf = np.expand_dims(image_inf.astype(np.float32) / 255.0, axis=0)

            name = self.vis_names[index]
            if image_vis.shape!=(3,384,512):
                print(vis_path)

            return (
                torch.tensor(image_vis),
                torch.tensor(image_inf),
                torch.tensor(label),
                name,
            )
        elif self.split == 'test':
            vis_path = self.vis_paths[index]
            ir_path = self.ir_paths[index]
            image_vis = np.array(Image.open(vis_path))
            # image_vis = cv2.resize(image_vis, (512, 384))
            image_inf = cv2.imread(ir_path, 0)
            # image_inf = cv2.resize(image_inf, (512, 384))


            image_vis = (np.asarray(Image.fromarray(image_vis), dtype=np.float32).transpose((2, 0, 1)) / 255.0)
            image_ir = np.asarray(Image.fromarray(image_inf), dtype=np.float32) / 255.0
            image_ir = np.expand_dims(image_ir, axis=0)

            name = self.vis_names[index]
            return (
                torch.tensor(image_vis),
                torch.tensor(image_ir),
                name,
            )

    def __len__(self):
        return self.length




if __name__ == '__main__':
    path = r'Dataset/Visible/test/00000.png'
    image_vis = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    imf = image_vis.astype(np.float32) / 255.0
    a = imf.transpose((2, 0, 1))
