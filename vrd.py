import json
import operator
from collections import defaultdict
import cv2
import pickle
import torch.utils.data as data
import torchvision.transforms as transforms
import itertools
import torch
import argparse
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import os
from torchvision import models
from operator import itemgetter
from utils import Model, plot_confusion_matrix, to_one_hot, SQLiteDatabase
from evaluate import top_recall_relationship, top_recall_phrase
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from time import time
from multiprocessing import Process

torch.manual_seed(1991)

image_folder = '/scratch/datasets/Data1/sg_dataset/'
class_model = models.densenet161(pretrained=True).cuda()

with open('data/annotations_train.json', 'r') as f:
    train_relationship_data = json.load(f)

with open('data/annotations_test.json', 'r') as f:
    test_relationship_data = json.load(f)

with open('data/objects.json', 'r') as f:
    list_objects = json.load(f)

with open('data/predicates.json', 'r') as f:
    list_predicates = json.load(f)

with open('data/mapping.pkl', 'rb') as f:
    mapping_id = pickle.load(f)

needed_object_list = [0, 1, 11, 14, 15, 16, 17, 31, 33, 46, 47, 48, 53, 54, 57, 58, 60, 70, 77, 79, 82, 83, 87, 89, 90,
                      91, 92, 96, 97, 98, 99]
new_to_old = {i: v for i, v in enumerate(needed_object_list)}
old_to_new = {v: i for i, v in enumerate(needed_object_list)}

# load imagenet metadata
# Load Imagenet Synsets
with open('data/imagenet_synsets.txt', 'r') as f:
    synsets = f.readlines()

# len(synsets)==1001
# sysnets[0] == background
synsets = [x.strip() for x in synsets]
splits = [line.split(' ') for line in synsets]
key_to_classname = {spl[0]: ' '.join(spl[1:]) for spl in splits}

with open('data/imagenet_classes.txt', 'r') as f:
    class_id_to_key = f.readlines()

class_id_to_key = [x.strip() for x in class_id_to_key]

id_to_object = {i: o for i, o in enumerate(list_objects)}
object_to_id = {o: i for i, o in enumerate(list_objects)}
id_to_predicate = {i: p for i, p in enumerate(list_predicates)}

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

transform_reg_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomSizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])

transform_reg_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Scale(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])

spatial_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Scale(size=(32, 32)),
    transforms.ToTensor(),
])


class Special_RN(nn.Module):
    def __init__(self, args, model=None):
        super(Special_RN, self).__init__()
        self.isGPU = torch.cuda.is_available()
        self.args = args

        self.visual_feature = getattr(models, args.reg_model_name)(pretrained=True).features

        self.spatial_feature = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=(5, 5), stride=(2, 2), padding=2),
            nn.ReLU(),
            nn.Conv2d(96, 128, kernel_size=(5, 5), stride=(2, 2), padding=2),
            nn.Conv2d(128, 64, kernel_size=(8, 8)),
            nn.ReLU()
        )

        self.reg_net = RegNet(args)

        if args.rel_model_name[0] == 'c':
            self.base_model = nn.Sequential(
                nn.Conv2d(3, 24, 3, stride=2, padding=1),
                nn.BatchNorm2d(24),
                nn.Conv2d(24, 24, 3, stride=2, padding=1),
                nn.BatchNorm2d(24),
                nn.Conv2d(24, 24, 3, stride=2, padding=1),
                nn.BatchNorm2d(24),
                nn.Conv2d(24, 24, 3, stride=2, padding=1),
                nn.BatchNorm2d(24),
                nn.Conv2d(24, 24, 3, stride=2, padding=1),
                nn.BatchNorm2d(24))

        else:
            self.base_model = getattr(models, args.rel_model_name)(pretrained=True).features
        #
        # in_size = list(self.base_model.modules())[-1].num_features * 2
        in_size = 2524  # 2324

        self.g_fc1 = nn.Linear(in_size, args.rn_size)

        if args.rn_same_layer:
            self.g_fcs = nn.Linear(args.rn_size, args.rn_size)
            self.f_fcs = nn.Linear(args.rn_size, args.rn_size)
        else:
            self.g_fcs = nn.ModuleList([nn.Linear(args.rn_size, args.rn_size)] * args.rn_num_layers[0])
            self.f_fcs = nn.ModuleList([nn.Linear(args.rn_size, args.rn_size)] * args.rn_num_layers[1])

        size = 7
        self.coord_size = size * size

        # prepare coord tensor
        np_coord_tensor = np.zeros((args.batch_size, self.coord_size, 2), dtype=np.float32)
        for i in range(self.coord_size):
            np_coord_tensor[:, i, :] = np.array([i // size, i % size])

        np_coord_tensor /= (size - 1)
        self.coord_tensor = torch.from_numpy(np_coord_tensor)
        if self.isGPU:
            self.coord_tensor = self.coord_tensor.cuda()
        self.coord_tensor = Variable(self.coord_tensor, requires_grad=False)

    def forward(self, inputs):
        img, x_s, x_o, x_r, mask, _, _ = inputs

        # """convolution"""
        x = self.base_model(img)

        x_r_visual_raw = self.visual_feature(x_r)
        out = F.relu(x_r_visual_raw, inplace=True)
        x_r_visual = F.avg_pool2d(out, kernel_size=7).view(x_r_visual_raw.size(0), -1)
        x_r_spatial = self.spatial_feature(mask).view(-1, 64)
        x_s = self.reg_net(x_s)
        x_o = self.reg_net(x_o)
        qst = torch.cat([x_r_visual, x_r_spatial, x_s, x_o], 1)

        qst_ = torch.unsqueeze(qst, 1)
        qst_ = qst_.repeat(1, self.coord_size, 1)
        qst = torch.unsqueeze(qst_, 2)

        # x = inputs

        mb = x.size(0)
        n_channels = x.size(1)
        d = x.size(2)
        # x_flat = (64 x 64 x 24)
        x_flat = x.view(mb, n_channels, d * d).permute(0, 2, 1)
        # add coordinates
        x_flat = torch.cat([x_flat, self.coord_tensor[:x_flat.size(0), :, :]], 2)

        # add question everywhere
        # cast all pairs against each other
        x_i = torch.unsqueeze(x_flat, 1)  # (64x1x25x26+11)
        x_i = x_i.repeat(1, self.coord_size, 1, 1)  # (64x25x25x26+11)
        x_j = torch.unsqueeze(x_flat, 2)  # (64x25x1x26+11)
        x_j = torch.cat([x_j, qst], 3)
        x_j = x_j.repeat(1, 1, self.coord_size, 1)  # (64x25x25x26+11)
        x_full = torch.cat([x_i, x_j], 3)  # (64x25x25x2*26+11)

        """g"""

        # reshape for passing through network
        x_ = x_full.view(-1, x_full.size(-1))
        x_ = self.g_fc1(x_)
        x_ = F.relu(x_)

        for i in range(self.args.rn_num_layers[0]):
            x_ = self.g_fcs(x_) if self.args.rn_same_layer else self.g_fcs[i](x_)
            x_ = F.relu(x_)

        # reshape again and sum
        x_g = x_.view(mb, -1, args.rn_size)
        x_g = x_g.sum(1).squeeze()
        x_f = x_g.unsqueeze(0) if len(x_g.size()) == 1 else x_g

        """f"""

        for i in range(self.args.rn_num_layers[1]):
            x_f = self.f_fcs(x_f) if self.args.rn_same_layer else self.f_fcs[i](x_f)
            x_f = F.relu(x_f)

        return x_f


class RN(nn.Module):
    def __init__(self, args, model=None):
        super(RN, self).__init__()
        self.isGPU = torch.cuda.is_available()
        self.args = args

        if model:
            self.base_model = model
        else:
            if args.rel_model_name[0] == 'c':
                self.base_model = nn.Sequential(
                    nn.Conv2d(3, 24, 3, stride=2, padding=1),
                    nn.BatchNorm2d(24),
                    nn.Conv2d(24, 24, 3, stride=2, padding=1),
                    nn.BatchNorm2d(24),
                    nn.Conv2d(24, 24, 3, stride=2, padding=1),
                    nn.BatchNorm2d(24),
                    nn.Conv2d(24, 24, 3, stride=2, padding=1),
                    nn.BatchNorm2d(24),
                    nn.Conv2d(24, 24, 3, stride=2, padding=1),
                    nn.BatchNorm2d(24))

            else:
                self.base_model = getattr(models, args.rel_model_name)(pretrained=True).features

        in_size = list(self.base_model.modules())[-1].num_features * 2
        # in_size = (2208 + 64) * 2

        self.g_fc1 = nn.Linear(in_size, args.rn_size)

        if args.rn_same_layer:
            self.g_fcs = nn.Linear(args.rn_size, args.rn_size)
            self.f_fcs = nn.Linear(args.rn_size, args.rn_size)
        else:
            self.g_fcs = nn.ModuleList([nn.Linear(args.rn_size, args.rn_size)] * args.rn_num_layers[0])
            self.f_fcs = nn.ModuleList([nn.Linear(args.rn_size, args.rn_size)] * args.rn_num_layers[1])

        size = 7
        self.coord_size = size * size
        #
        # # prepare coord tensor
        # np_coord_tensor = np.zeros((args.batch_size, self.coord_size, 2), dtype=np.float32)
        # for i in range(self.coord_size):
        #     np_coord_tensor[:, i, :] = np.array([i // size, i % size])
        #
        # np_coord_tensor /= (size - 1)
        # self.coord_tensor = torch.from_numpy(np_coord_tensor)
        # if self.isGPU:
        #     self.coord_tensor = self.coord_tensor.cuda()
        # self.coord_tensor = Variable(self.coord_tensor, requires_grad=False)

    def forward(self, inputs):
        img = inputs
        #
        # """convolution"""
        x = self.base_model(img)

        # x = inputs

        mb = x.size(0)
        n_channels = x.size(1)
        d = x.size(2)
        # x_flat = (64 x 64 x 24)
        x_flat = x.view(mb, n_channels, d * d).permute(0, 2, 1)
        # add coordinates
        # x_flat = torch.cat([x_flat, self.coord_tensor[:x_flat.size(0), :, :]], 2)

        # add question everywhere
        # cast all pairs against each other
        x_i = torch.unsqueeze(x_flat, 1)  # (64x1x25x26+11)
        x_i = x_i.repeat(1, self.coord_size, 1, 1)  # (64x25x25x26+11)
        x_j = torch.unsqueeze(x_flat, 2)  # (64x25x1x26+11)
        x_j = x_j.repeat(1, 1, self.coord_size, 1)  # (64x25x25x26+11)
        x_full = torch.cat([x_i, x_j], 3)  # (64x25x25x2*26+11)

        """g"""

        # reshape for passing through network
        x_ = x_full.view(-1, x_full.size(-1))
        x_ = self.g_fc1(x_)
        x_ = F.relu(x_)

        for i in range(self.args.rn_num_layers[0]):
            x_ = self.g_fcs(x_) if self.args.rn_same_layer else self.g_fcs[i](x_)
            x_ = F.relu(x_)

        # reshape again and sum
        x_g = x_.view(mb, -1, args.rn_size)
        x_g = x_g.sum(1).squeeze()
        x_f = x_g.unsqueeze(0) if len(x_g.size()) == 1 else x_g

        """f"""

        for i in range(self.args.rn_num_layers[1]):
            x_f = self.f_fcs(x_f) if self.args.rn_same_layer else self.f_fcs[i](x_f)
            x_f = F.relu(x_f)

        return x_f


class RegNet(nn.Module):
    def __init__(self, args):
        super(RegNet, self).__init__()

        self.model = getattr(models, args.reg_model_name)(pretrained=True)

        if args.reg_model_name[0] == 'd':  # for densenet, alexnet and vggnet
            num_features = self.model.classifier.in_features
            self.model.classifier = nn.Linear(num_features, args.num_objects)
        elif args.reg_model_name[0] == 'a':
            num_features = self.model.classifier[1].in_features
            self.model.classifier = nn.Linear(num_features, args.num_objects)
        elif args.reg_model_name[0] == 'v':
            num_features = self.model.classifier[0].in_features
            self.model.classifier = nn.Linear(num_features, args.num_objects)
        else:  # for resnet and inception
            num_features = self.model.fc.in_features
            self.model.fc = nn.Linear(num_features, args.num_objects)

    def forward(self, x):
        if isinstance(x, (list, tuple)):
            x = x[0]
        x = self.model(x)

        return x


class ImprovedVRDDataset(data.Dataset):
    def __init__(self, mode, args=None):

        self.image_path = image_folder + 'sg_' + mode + '_images/{}'
        if mode == 'train':
            self.transform_fn = transform_reg_train
        else:
            self.transform_fn = transform_reg_test

        data_file = 'data/vrd_' + mode + '_data.pkl'
        self.args = args

        with open(data_file, 'rb') as f:
            self.examples = pickle.load(f)
            # self.examples = self.examples[:100]

    def __getitem__(self, index):
        img_id, img_bbox, object_bbox, subject_bbox, o1_id, o2_id, value = self.examples[index]
        img = cv2.imread(self.image_path.format(img_id), cv2.IMREAD_COLOR)

        u_size_h, u_size_w = img_bbox[1] - img_bbox[0], img_bbox[3] - img_bbox[2]
        o_size_h, o_size_w = object_bbox[1] - object_bbox[0], object_bbox[3] - object_bbox[2]
        s_size_h, s_size_w = subject_bbox[1] - subject_bbox[0], subject_bbox[3] - subject_bbox[2]

        h, w = img.shape[0], img.shape[1]
        range_ = 0.1

        u_range_h, u_range_w = int(range_ * u_size_h), int(range_ * u_size_w)
        o_range_h, o_range_w = int(range_ * o_size_h), int(range_ * o_size_w)
        s_range_h, s_range_w = int(range_ * s_size_h), int(range_ * s_size_w)

        new_u_bbox = max(0, img_bbox[0] - u_range_h), min(h, img_bbox[1] + u_range_h), \
                     max(0, img_bbox[2] - u_range_w), min(w, img_bbox[3] + u_range_w)

        new_o_bbox = max(0, object_bbox[0] - o_range_h), min(h, object_bbox[1] + o_range_h), \
                     max(0, object_bbox[2] - o_range_w), min(w, object_bbox[3] + o_range_w)

        new_s_bbox = max(0, subject_bbox[0] - s_range_h), min(h, subject_bbox[1] + s_range_h), \
                     max(0, subject_bbox[2] - s_range_w), min(w, subject_bbox[3] + s_range_w)

        image = self.transform_fn(img)

        union_img = img[new_u_bbox[0]:new_u_bbox[1], new_u_bbox[2]:new_u_bbox[3]]
        union_img = self.transform_fn(union_img)

        object_img = img[new_o_bbox[0]:new_o_bbox[1], new_o_bbox[2]:new_o_bbox[3]]
        object_img = self.transform_fn(object_img)

        subject_img = img[new_s_bbox[0]:new_s_bbox[1], new_s_bbox[2]:new_s_bbox[3]]
        subject_img = self.transform_fn(subject_img)

        union_mask = torch.zeros(*img.shape[:2])
        union_mask[img_bbox[0]: img_bbox[1], img_bbox[2]:img_bbox[3]] = 1

        subject_mask = torch.zeros(*img.shape[:2])
        subject_mask[subject_bbox[0]:subject_bbox[1], subject_bbox[2]:subject_bbox[3]] = 1

        object_mask = torch.zeros(*img.shape[:2])
        object_mask[object_bbox[0]:object_bbox[1], object_bbox[2]:object_bbox[3]] = 1

        mask = torch.cat([union_mask.unsqueeze(0), subject_mask.unsqueeze(0), object_mask.unsqueeze(0)], 0)
        mask = spatial_transform(mask)

        object_id = torch.LongTensor([o1_id])
        subject_id = torch.LongTensor([o2_id])
        object_dist = torch.FloatTensor(to_one_hot(o1_id, args.num_objects))
        subject_dist = torch.FloatTensor(to_one_hot(o2_id, args.num_objects))
        predicate_id = torch.LongTensor([value])

        return (image, subject_img, object_img, union_img, mask, subject_dist, object_dist), (
        subject_id, predicate_id, object_id)

    def __len__(self):
        return len(self.examples)


class ImprovedEvaluateDataset(data.Dataset):
    def __init__(self, data_load, image_id, level, args=None):
        self.args = args
        self.level = level
        self.image_path = image_folder + 'sg_test_images/{}'
        self.transform_fn = transform_reg_test
        self.examples = data_load
        self.img = cv2.imread(self.image_path.format(image_id), cv2.IMREAD_COLOR)

    def __getitem__(self, index):
        if level in [1, 2]:
            _, union_bbox, object_bbox, subject_bbox, object_id, subject_id, _ = self.examples[index]
            object_dist = torch.FloatTensor(to_one_hot(object_id, args.num_objects))
            subject_dist = torch.FloatTensor(to_one_hot(subject_id, args.num_objects))
        else:
            subject_bbox, object_bbox, score = self.examples[index]

            union_bbox = [min(object_bbox[0], subject_bbox[0]), max(object_bbox[1], subject_bbox[1]),
                          min(object_bbox[2], subject_bbox[2]), max(object_bbox[3], subject_bbox[3])]

            subject_bbox = [int(round(x)) for x in subject_bbox]
            object_bbox = [int(round(x)) for x in object_bbox]
            union_bbox = [int(round(x)) for x in union_bbox]

            object_dist = torch.FloatTensor([0] * args.num_objects)
            subject_dist = torch.FloatTensor([0] * args.num_objects)

        u_size_h, u_size_w = union_bbox[1] - union_bbox[0], union_bbox[3] - union_bbox[2]
        o_size_h, o_size_w = object_bbox[1] - object_bbox[0], object_bbox[3] - object_bbox[2]
        s_size_h, s_size_w = subject_bbox[1] - subject_bbox[0], subject_bbox[3] - subject_bbox[2]

        h, w = self.img.shape[0], self.img.shape[1]
        range_ = 0.1

        u_range_h, u_range_w = int(range_ * u_size_h), int(range_ * u_size_w)
        o_range_h, o_range_w = int(range_ * o_size_h), int(range_ * o_size_w)
        s_range_h, s_range_w = int(range_ * s_size_h), int(range_ * s_size_w)

        new_u_bbox = max(0, union_bbox[0] - u_range_h), min(h, union_bbox[1] + u_range_h), \
                     max(0, union_bbox[2] - u_range_w), min(w, union_bbox[3] + u_range_w)

        new_o_bbox = max(0, object_bbox[0] - o_range_h), min(h, object_bbox[1] + o_range_h), \
                     max(0, object_bbox[2] - o_range_w), min(w, object_bbox[3] + o_range_w)

        new_s_bbox = max(0, subject_bbox[0] - s_range_h), min(h, subject_bbox[1] + s_range_h), \
                     max(0, subject_bbox[2] - s_range_w), min(w, subject_bbox[3] + s_range_w)

        try:
            image = self.transform_fn(self.img)

            union_img = self.img[new_u_bbox[0]:new_u_bbox[1], new_u_bbox[2]:new_u_bbox[3]]
            union_img = self.transform_fn(union_img)

            object_img = self.img[new_o_bbox[0]:new_o_bbox[1], new_o_bbox[2]:new_o_bbox[3]]
            object_img = self.transform_fn(object_img)

            subject_img = self.img[new_s_bbox[0]:new_s_bbox[1], new_s_bbox[2]:new_s_bbox[3]]
            subject_img = self.transform_fn(subject_img)

            union_mask = torch.zeros(*self.img.shape[:2])
            union_mask[union_bbox[0]: union_bbox[1], union_bbox[2]:union_bbox[3]] = 1

            subject_mask = torch.zeros(*self.img.shape[:2])
            subject_mask[subject_bbox[0]:subject_bbox[1], subject_bbox[2]:subject_bbox[3]] = 1

            object_mask = torch.zeros(*self.img.shape[:2])
            object_mask[object_bbox[0]:object_bbox[1], object_bbox[2]:object_bbox[3]] = 1
        except ValueError:
            return None

        mask = torch.cat([union_mask.unsqueeze(0), subject_mask.unsqueeze(0), object_mask.unsqueeze(0)], 0)
        mask = spatial_transform(mask)

        if self.level == 1:
            return (image, subject_img, object_img, union_img, mask, subject_dist, object_dist), (
            subject_bbox, object_bbox, subject_id, object_id)
        elif self.level == 2:
            return (image, subject_img, object_img, union_img, mask, subject_dist, object_dist), (
            subject_bbox, object_bbox, None)
        else:
            return (image, subject_img, object_img, union_img, mask, subject_dist, object_dist), (
            subject_bbox, object_bbox, score)

    def __len__(self):
        return len(self.examples)


class ImprovedRN(nn.Module):
    def __init__(self, args):
        super(ImprovedRN, self).__init__()

        self.args = args
        self.activate_fn = getattr(F, args.activation)

        if args.exp_num in [1, 2, 4, 5, 6, 9, 10, 11, 12, 15, 16, 19, 20, 21,
                            22] or args.reg_model_name == args.rel_model_name:
            self.visual_feature = getattr(models, args.reg_model_name)(pretrained=True).features
        else:
            self.visual_feature = None

        if args.exp_num in [2, 4, 5, 6, 9, 10, 11, 12, 15, 16, 19, 20, 21, 22]:
            self.spatial_feature = nn.Sequential(
                nn.Conv2d(3, 96, kernel_size=(5, 5), stride=(2, 2), padding=2),
                nn.ReLU(),
                nn.Conv2d(96, 128, kernel_size=(5, 5), stride=(2, 2), padding=2),
                nn.Conv2d(128, 64, kernel_size=(2, 2)) if args.exp_num in [21, 22] else nn.Conv2d(128, 64,
                                                                                                  kernel_size=(8, 8)),
                nn.ReLU()
            )

        if args.exp_num in [3, 4, 7, 8, 9, 10, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]:
            self.RN = RN(args)

        if args.exp_num in [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22]:
            self.reg_net = RegNet(args)

        if args.exp_num in [6, 8, 10, 12, 14, 16, 17, 18, 19, 20, 22]:
            self.Wrr = nn.Linear(args.num_predicates, args.num_predicates)
            self.Wro = nn.Linear(args.num_objects, args.num_predicates)
            self.Wrs = nn.Linear(args.num_objects, args.num_predicates)

            # self.Wsr = nn.Linear(args.num_predicates, args.num_objects)
            # self.Wso = nn.Linear(args.num_objects, args.num_objects)
            # self.Wos = nn.Linear(args.num_objects, args.num_objects)
            # self.Wor = nn.Linear(args.num_predicates, args.num_objects)

        if args.exp_num == 1:
            self.predicate_features = nn.Linear(2208, args.num_predicates)
        elif args.exp_num in [2, 6, 11, 12]:
            self.predicate_features = nn.Linear(2208 + 64, args.num_predicates)
        elif args.exp_num in [3, 8, 14, 17, 18, 21, 22]:
            self.predicate_features = nn.Linear(args.rn_size, args.num_predicates)
        elif args.exp_num in [4, 10, 16, 19, 20]:
            self.predicate_features = nn.Linear(args.rn_size + 2208 + 64, args.num_predicates)
        elif args.exp_num in [5]:
            self.predicate_features = nn.Linear(2208 + 64 + 200, args.num_predicates)
        elif args.exp_num in [7, 13]:
            self.predicate_features = nn.Linear(args.rn_size + 200, args.num_predicates)
        elif args.exp_num in [9, 15]:
            self.predicate_features = nn.Linear(args.rn_size + 2208 + 64 + 200, args.num_predicates)

    def forward(self, x):
        _, x_s, x_o, x_r, mask, subject_dist, object_dist = x

        if self.args.exp_num in [1, 2, 4, 5, 6, 9, 10, 11, 12, 15, 16, 19, 20, 21, 22] or \
                        self.args.reg_model_name == self.args.rel_model_name:
            x_r_visual_raw = self.visual_feature(x_r)
            out = F.relu(x_r_visual_raw, inplace=True)
            x_r_visual = F.avg_pool2d(out, kernel_size=7).view(x_r_visual_raw.size(0), -1)

        if self.args.exp_num in [2, 4, 5, 6, 9, 10, 11, 12, 15, 16, 19, 20, 21, 22]:
            x_r_spatial = self.spatial_feature(mask).view(-1, 64)

        if self.args.exp_num in [21, 22]:
            x_r_spatial = self.spatial_feature(mask)

        if self.args.exp_num in [3, 4, 7, 8, 9, 10, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]:
            x_r_rel = self.activate_fn(self.RN(x_r))

        if self.args.exp_num in [21, 22]:
            x_r_rel = self.activate_fn(self.RN(torch.cat([x_r_visual_raw, x_r_spatial], 1)))

        if self.args.exp_num in [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22]:
            x_s = self.reg_net(x_s)
            x_o = self.reg_net(x_o)
        elif self.args.exp_num in [5, 6, 7, 8, 9, 10]:
            x_s = subject_dist
            x_o = object_dist

        # Handle x_r
        if self.args.exp_num == 1:
            x_r = self.predicate_features(x_r_visual)
        elif self.args.exp_num in [2, 6, 11, 12]:
            x_r = self.predicate_features(torch.cat((x_r_visual, x_r_spatial), 1))
        elif self.args.exp_num in [3, 8, 14, 17, 18, 21, 22]:
            x_r = self.predicate_features(x_r_rel)
        elif self.args.exp_num in [4, 10, 16, 19, 20]:
            x_r = self.predicate_features(torch.cat((x_r_visual, x_r_spatial, x_r_rel), 1))
        elif self.args.exp_num in [5]:
            x_r = self.predicate_features(torch.cat((x_r_visual, x_r_spatial, x_s, x_o), 1))
        elif self.args.exp_num in [7, 13]:
            x_r = self.predicate_features(torch.cat((x_r_rel, x_s, x_o), 1))
        elif self.args.exp_num in [9, 15]:
            x_r = self.predicate_features(torch.cat((x_r_visual, x_r_spatial, x_r_rel, x_s, x_o), 1))

        if self.args.exp_num in [6, 8, 10, 12, 14, 16, 17, 18, 19, 20, 22]:  # Message Passing
            x_rs = self.Wrs(x_s)
            x_ro = self.Wro(x_o)

            for i in range(self.args.num_iterations):
                x_r = self.activate_fn(self.Wrr(x_r) + x_rs + x_ro)

            # q_s, q_o, q_r = x_s.clone(), x_o.clone(), x_r.clone()
            #
            # for i in range(self.args.num_iterations):
            #     q_s_new = self.activate_fn(x_s + self.Wsr(q_r) + self.Wso(q_o))
            #     q_r_new = self.activate_fn(x_r + self.Wrs(q_s) + self.Wro(q_o))
            #     q_o_new = self.activate_fn(x_o + self.Wos(q_s) + self.Wor(q_r))
            #
            #     q_s, q_r, q_o = q_s_new, q_r_new, q_o_new
            #
            # x_r, x_o, x_s = q_r, q_o, q_s

        if self.args.exp_num in [6, 11, 12, 14, 16, 17, 18, 19, 20, 22]:
            return x_s, x_r, x_o
        else:
            return x_r


def save_to_db(sqlite_db, batch):
    sqlite_db.insert_batch(batch)


if __name__ == '__main__':

    # Options to run
    parser = argparse.ArgumentParser(description='VRD Dataset Relational Network')

    parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                        help='vrd_input batch size for training (default: 64)')
    parser.add_argument('--reg-model-name', type=str, default='densenet161', metavar='N',
                        help='Recognition Model')
    parser.add_argument('--rel-model-name', type=str, default='conv', metavar='N',
                        help='Relational Model')
    parser.add_argument('--activation', type=str, default='relu', metavar='N',
                        help='Activation to train')
    parser.add_argument('--loss', type=str, default='CrossEntropyLoss', metavar='N',
                        help='Loss to train')
    parser.add_argument('--env-name', type=str, default='Experiment {}',
                        metavar='N', help='Environment name for displaying plot')
    parser.add_argument('--optimizer', type=str, default='Adam',
                        metavar='N', help='Optimizer to use')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 2.5e-4)')
    parser.add_argument('--mode', type=str, default='train', metavar='N',
                        help='mode: train or test (default: train)')
    # parser.add_argument('--version', type=int, default=1, metavar='N',
    #                     help='version of relation network (default: 1)')
    parser.add_argument('--num_objects', type=int, default=100, metavar='N',
                        help='num of object class')
    parser.add_argument('--num_iterations', type=int, default=10, metavar='N',
                        help='num of iteration in message passing')
    parser.add_argument('--num_predicates', type=int, default=70, metavar='N',
                        help='num of predicate class')
    parser.add_argument('--num-workers', type=int, default=4, metavar='N',
                        help='num of workers to fetch data')
    parser.add_argument('--rn-size', type=int, default=256, metavar='N',
                        help='relation network intermediate layer size')
    parser.add_argument('--rn-num-layers', type=list, default=[3, 4], metavar='N',
                        help='relation network num of layer')
    parser.add_argument('--rn-same-layer', type=bool, default=True, metavar='N',
                        help='relation network num of layer')
    parser.add_argument('--exp-num', type=int, default=6, metavar='N',
                        help='Experiment number to run')
    parser.add_argument('--resume-train', type=bool, default=False, metavar='N',
                        help='Resume training')

    """ Experiments' Number:
    0: Regnet
    1: Visual
    2: Visual + Spatial
    3: RN
    4: Visual + Spatial + RN32.21
    5: Visual + Spatial + Class
    6: Visual + Spatial + Class + MP (Message Passing)
    7: RN + Class
    8: RN + Class + MP
    9: Visual + Spatial + RN + Class
    10: Visual + Spatial + RN + Class + MP
    11: Visual + Spatial + Bbox 
    12: Visual + Spatial + Bbox + MP
    13: RN + Bbox 
    14: RN + Bbox + MP
    15: Visual + Spatial + RN + Bbox
    16: Visual + Spatial + RN + Bbox + MP
    17: = 14 + Bbox from RPN (Relationship Proposal Network)
    18: = 17 + Refine
    19: = 16 + Bbox from RPN
    20: = 19 + Refine
    21: New suggested model: RN based on concat of Visual and Spatial Feature + MP
    22: = 21 + MP
    23: Special RN
    
    (2),      (3),      (4): Base
    (5, 6),   (7, 8),   (9, 10): Add Class Info
    (11, 12), (13, 14), (15, 16): Replace Class Info with Bbox Info
    (),       (17, 18), (19, 20): Test RPN and Refine Module
    """

    args = parser.parse_args()
    args.env_name = args.env_name.format(args.exp_num)
    print(args)

    net = ImprovedRN(args) if args.exp_num < 23 else Special_RN(args)

    model = Model(net, args)

    if args.mode == 'train':
        if args.resume_train:
            model.load('model_checkpoints/Experiment_' + str(args.exp_num)
                       + '/weights.51-train_loss:0.36-train_acc:0.85-val_loss:0.58-val_acc:0.70.pkl')
        model.train()

        # Load datasets
        train_dataset = ImprovedVRDDataset('train', args)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

        val_dataset = ImprovedVRDDataset('test', args)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

        dset_loaders = {'train': train_loader, 'val': val_loader}
        dset_sizes = {'train': len(train_loader), 'val': len(val_dataset)}

        criterion = getattr(nn, args.loss)()
        optimizer = getattr(optim, args.optimizer)(net.parameters(), args.lr)
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2, verbose=True)

        model.compile(criterion, optimizer, scheduler=scheduler)

        folder_checkpoint = "model_checkpoints/" + args.env_name.replace(' ', '_') + '/'

        if not os.path.exists(folder_checkpoint):
            os.mkdir(folder_checkpoint)

        model_checkpoint = folder_checkpoint + "weights.{epoch:02d}-train_loss:{train_loss:.2f}" \
            + "-train_acc:{train_acc:.2f}-val_loss:{val_loss:.2f}-val_acc:{val_acc:.2f}.pkl"

        monitor_outputs = [0, 1, 2] if args.exp_num in [6, 11, 12, 14, 16, 17, 18, 19, 20, 22] else None

        model.fit_loader(dset_loaders, dset_sizes, env=args.env_name, batch_size=args.batch_size, evaluate_fn=None,
                         num_epochs=args.epochs, model_checkpoint=model_checkpoint, monitor_outputs=monitor_outputs,
                         display_acc=1)

    else:
        result_file = 'result2/experiment_' + str(args.exp_num) + '.pkl'
        with open('data/list_test_images.pkl', 'rb') as f:
            list_test_image = pickle.load(f)

        from vrd_2 import VRDDataset

        val_dataset = VRDDataset('test', args)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

        top_hs = [1, 2, 5, 8]
        last_top_h = top_hs[-1]
        top_h_test = 5

        if os.path.isfile(result_file):
            with open(result_file, 'rb') as f:
                results = pickle.load(f)[top_h_test]
        else:
            model.load('model_checkpoints/Experiment_' + str(args.exp_num) + '/official_weights.pkl')
            model.eval()

            test_image_dict = {image_id: i for i, image_id in enumerate(list_test_image)}
            results = {top_k: {image_id: {'list_confs': [], 'list_tuples': [],
                                          'list_tuple_confs': [],
                                          'list_sub_bboxes': [], 'list_obj_bboxes': []}
                               for image_id in list_test_image}
                       for top_k in top_hs}

            for k, example in enumerate(val_loader):
                print(k, len(val_loader))
                list_inputs, (image_ids, subject_bboxes_, object_bboxes_) = example

                subject_bboxes = torch.cat([x.unsqueeze(1) for x in subject_bboxes_], 1)
                object_bboxes = torch.cat([x.unsqueeze(1) for x in object_bboxes_], 1)

                list_inputs = [Variable(x.cuda(), volatile=True) for x in list_inputs]

                subject, predicate, object_ = model.net.forward(list_inputs)

                conf_ss, index_ss = subject.sort(1, descending=True)
                conf_ps, index_ps = predicate.sort(1, descending=True)
                conf_os, index_os = object_.sort(1, descending=True)

                for i, image_id in enumerate(image_ids):
                    data_s_ = torch.cat((conf_ss[i, :last_top_h].view(-1, 1),
                                         index_ss[i, :last_top_h].view(-1, 1).float()), 1).data.cpu().numpy()
                    data_p_ = torch.cat((conf_ps[i, :last_top_h].view(-1, 1),
                                         index_ps[i, :last_top_h].view(-1, 1).float()), 1).data.cpu().numpy()
                    data_o_ = torch.cat((conf_os[i, :last_top_h].view(-1, 1),
                                         index_os[i, :last_top_h].view(-1, 1).float()), 1).data.cpu().numpy()

                    data_s = data_s_[data_s_[:, 0] > 0, :]
                    data_p = data_p_[data_p_[:, 0] > 0, :]
                    data_o = data_o_[data_o_[:, 0] > 0, :]

                    if data_s.shape[0] == 0:
                        data_s = data_s_[0:1, :]

                    if data_p.shape[0] == 0:
                        data_p = data_p_[0:1, :]

                    if data_o.shape[0] == 0:
                        data_o = data_o_[0:1, :]

                    s_p_o_indices_triplets = [(s, p, o) for s in data_s[:, 1].astype(int)
                                              for p in data_p[:, 1].astype(int)
                                              for o in data_o[:, 1].astype(int)]
                    s_p_o_scores_triplets = [float(s * p * o) if s > 0 and p > 0 and o > 0 else 0
                                             for s in data_s[:, 0]
                                             for p in data_p[:, 0]
                                             for o in data_o[:, 0]]
                    s_p_o_scores_tuples = [(s, p, o) for s in data_s[:, 0] for p in data_p[:, 0] for o in
                                           data_o[:, 0]]

                    s_p_o_scores, s_p_o_indices = torch.FloatTensor(s_p_o_scores_triplets).sort(0, descending=True)

                    for top_k in top_hs:
                        triplets = [s_p_o_indices_triplets[m] for m in s_p_o_indices[:top_k]]
                        tuple_confs = [s_p_o_scores_tuples[m] for m in s_p_o_indices[:top_k]]
                        confs = s_p_o_scores[:top_k].tolist()

                        num_triplets = len(confs)
                        s_boxes = [subject_bboxes[i, :]] * num_triplets
                        o_boxes = [object_bboxes[i, :]] * num_triplets

                        results[top_k][image_id]['list_confs'].extend(confs)
                        results[top_k][image_id]['list_tuple_confs'].extend(tuple_confs)
                        results[top_k][image_id]['list_tuples'].extend(triplets)
                        results[top_k][image_id]['list_sub_bboxes'].extend(s_boxes)
                        results[top_k][image_id]['list_obj_bboxes'].extend(o_boxes)

            with open(result_file, 'wb') as f:
                pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)

            results = results[top_h_test]

        print('Complete Collect data for evaluation')

        top_ks = [50, 100, 1000]

        from copy import deepcopy

        for top_k in top_ks:
            result_ = deepcopy(results)
            result = top_recall_phrase(top_k, result_)
            print('R@{}, phrase'.format(top_k), result)

            result_ = deepcopy(results)
            result = top_recall_relationship(top_k, result_)
            print('R@{}, relationship'.format(top_k), result)
