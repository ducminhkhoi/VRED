import torch
import json
import pickle
from utils import Model, to_one_hot
from torch import nn
import torch.utils.data as data
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.transforms as transforms
from torchvision import models
from torch.autograd import Variable
import cv2
import csv
import os
import argparse
import numpy as np
from itertools import chain
from extract_data import get_rcnn_results
from evaluate import top_recall_phrase, top_recall_relationship, compute_score, \
    get_sublist, compute_recall_phrase_and_relationship
from collections import defaultdict
from copy import deepcopy
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from visdom import Visdom
import torchvision

viz = Visdom()

image_folder = '/scratch/datasets/Data1/sg_dataset/'

with open('data/objects.json', 'r') as f:
    list_objects = json.load(f)

with open('data/predicates.json', 'r') as f:
    list_predicates = json.load(f)

with open('data/annotations_train.json', 'r') as f:
    train_relationship_data = json.load(f)

with open('data/annotations_test.json', 'r') as f:
    test_relationship_data = json.load(f)

id_to_object = {i: o for i, o in enumerate(list_objects)}
object_to_id = {o: i for i, o in enumerate(list_objects)}
id_to_predicate = {i: p for i, p in enumerate(list_predicates)}
predicate_to_id = {p: i for i, p in enumerate(list_predicates)}

triplets = [(s, p, o) for s in range(len(list_objects)) for p in range(len(list_predicates))
            for o in range(len(list_objects))]
dict_triplets_to_indices = {x: i for i, x in enumerate(triplets)}
dict_indices_to_triplets = {i: x for i, x in enumerate(triplets)}

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

embedding_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Scale(size=(93, 93)),
    transforms.ToTensor(),
])


class PairFiltering(nn.Module):
    """Classify a pair of bounding boxes is a relationship or not"""

    def __init__(self):
        super(PairFiltering, self).__init__()

        self.spatial_feature = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=(5, 5), stride=(2, 2), padding=2),
            nn.ReLU(),
            nn.Conv2d(96, 128, kernel_size=(5, 5), stride=(2, 2), padding=2),
            nn.Conv2d(128, 64, kernel_size=(8, 8)),
            nn.ReLU()
        )

        self.fc1 = nn.Linear(64 + 2 * 100, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 2)

    def forward(self, x):
        mask, subject_dist, object_dist = x

        x_spatial = self.spatial_feature(mask).view(-1, 64)

        x = torch.cat([subject_dist, x_spatial, object_dist], 1)

        x = self.fc1(x)
        x = self.fc2(F.relu(x))
        x = self.fc3(F.relu(x))

        return x


class RedesignedModel(nn.Module):
    """Classify a pair of bounding boxes is a relationship or not"""

    def __init__(self, args):
        super(RedesignedModel, self).__init__()
        self.args = args
        self.activate_fn = getattr(F, args.activation)

        self.visual_feature = VisualModel(args, num_out=2208)
        self.spatial_feature = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=(5, 5), stride=(2, 2), padding=2),
            nn.ReLU(),
            nn.Conv2d(96, 128, kernel_size=(5, 5), stride=(2, 2), padding=2),
            nn.Conv2d(128, 64, kernel_size=(8, 8)),
            nn.ReLU()
        )

        self.predicate_feature = nn.Linear(2208 + 64, args.num_predicates)

        self.semantic = nn.Linear(args.num_objects * 2 + args.num_predicates, 4096)

        self.subject_class = nn.Linear(4096, args.num_objects)
        self.predicate_class = nn.Linear(4096, args.num_predicates)
        self.object_class = nn.Linear(4096, args.num_objects)

        self.Wrr = nn.Linear(args.num_predicates, args.num_predicates)
        self.Wrs = nn.Linear(args.num_objects, args.num_predicates)
        self.Wro = nn.Linear(args.num_objects, args.num_predicates)

    def forward(self, x):
        subject_image, union_image, object_image, mask, subject_dist, object_dist = x

        x_visual = self.visual_feature(union_image)
        x_spatial = self.spatial_feature(mask).view(-1, 64)

        x_r = self.predicate_feature(torch.cat([x_visual, x_spatial], 1))
        x_s = subject_dist
        x_o = object_dist

        # Project to semantic space
        x_semantic = self.semantic(torch.cat([x_s, x_r, x_o], 1))
        x_s_ = self.subject_class(x_semantic)
        x_r = self.predicate_class(x_semantic)
        x_o_ = self.object_class(x_semantic)

        x_rs = self.Wrs(x_s_)
        x_ro = self.Wro(x_o_)
        # Message Passing

        for i in range(self.args.num_iterations):
            x_r = self.activate_fn(self.Wrr(x_r) + x_rs + x_ro)

        if self.args.mode == 'train':
            return x_s, x_r, x_o, x_semantic
        else:
            return x_s, x_r, x_o


class VisualModel(nn.Module):
    def __init__(self, args, num_out=None):
        super(VisualModel, self).__init__()

        self.model = getattr(models, args.reg_model_name)(pretrained=True)
        self.args = args
        num_features = self.model.classifier.in_features

        if num_out is None:
            num_out = args.num_objects

        self.use_classifier = not (num_features == num_out)

        self.model.classifier = nn.Linear(num_features, num_out)

    def forward(self, x):
        if self.use_classifier:
            x = self.model(x)
        else:
            x = self.model.features(x)
            x = F.avg_pool2d(F.relu(x, inplace=True), kernel_size=7).view(x.size(0), -1)
        return x


class VRDDataset(data.Dataset):
    def __init__(self, mode, args=None, triplet=None):

        self.image_path = image_folder + 'sg_' + mode + '_images/{}'
        self.image_path_2 = 'data/unrel-dataset/images/{}.jpg'
        if mode == 'train':
            self.transform_fn = transform_reg_train
            if args.exp_num in {64, 65}:
                data_file = 'data/vrd_train_data_pair_filtering.pkl'
            elif args.exp_num in {57, 67}:
                data_file = 'data/vrd_train_data_faster_rcnn_detector.pkl'
            else:
                data_file = 'data/vrd_train_data.pkl'
        else:
            self.transform_fn = transform_reg_test

            if args.exp_num in {46, 47, 48, 53, 66}:
                data_file = 'data/vrd_test_data_rcnn.pkl'
            elif args.exp_num in {64, 65}:
                data_file = 'data/vrd_test_data_pair_filtering.pkl'
            elif args.exp_num in {57, 67}:
                data_file = 'data/vrd_test_data_faster_rcnn_detector.pkl'
            elif args.exp_num in {68, 69, 79, 80}:
                data_file = 'data/vrd_test_data_combine_detector_filter.pkl'
            elif args.exp_num in {76}:
                data_file = 'data/unrel_test_data.pkl'
            else:
                data_file = 'data/vrd_test_data.pkl'

        self.args = args

        with open(data_file, 'rb') as f:
            self.examples = pickle.load(f)

        if triplet:
            self.examples = self.examples[triplet]

        print(len(self.examples))

    def __getitem__(self, index):
        if self.args.exp_num in {46, 47, 48, 53, 57, 58, 59, 66, 67, 68, 69, 79, 80}:
            img_id, union_bbox, object_bbox, subject_bbox, object_id, subject_id, object_conf, subject_conf = \
                self.examples[index]
        else:
            img_id, union_bbox, object_bbox, subject_bbox, object_id, subject_id, pred_id = self.examples[index]

        if img_id.endswith('.jpg') or img_id.endswith('.png'):
            img = cv2.imread(self.image_path.format(img_id), cv2.IMREAD_COLOR)
        else:
            img = cv2.imread(self.image_path_2.format(img_id), cv2.IMREAD_COLOR)

        u_size_h, u_size_w = union_bbox[1] - union_bbox[0], union_bbox[3] - union_bbox[2]
        o_size_h, o_size_w = object_bbox[1] - object_bbox[0], object_bbox[3] - object_bbox[2]
        s_size_h, s_size_w = subject_bbox[1] - subject_bbox[0], subject_bbox[3] - subject_bbox[2]

        h, w = img.shape[0], img.shape[1]
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

        union_img_ = img[new_u_bbox[0]:new_u_bbox[1], new_u_bbox[2]:new_u_bbox[3]]
        union_img = self.transform_fn(union_img_)

        object_img_ = img[new_o_bbox[0]:new_o_bbox[1], new_o_bbox[2]:new_o_bbox[3]]
        object_img = self.transform_fn(object_img_)

        subject_img_ = img[new_s_bbox[0]:new_s_bbox[1], new_s_bbox[2]:new_s_bbox[3]]
        subject_img = self.transform_fn(subject_img_)

        if self.args.exp_num in {81, 82}:
            transform_tensor = embedding_transform(union_img_)

        # print(list_objects[subject_id], list_predicates[pred_id], list_objects[object_id])
        # cv2.imshow('test', transform_tensor.permute(1, 2, 0).numpy())
        # cv2.waitKey()

        union_mask = torch.zeros(*img.shape[:2])
        union_mask[union_bbox[0]: union_bbox[1], union_bbox[2]:union_bbox[3]] = 1

        subject_mask = torch.zeros(*img.shape[:2])
        subject_mask[subject_bbox[0]:subject_bbox[1], subject_bbox[2]:subject_bbox[3]] = 1

        object_mask = torch.zeros(*img.shape[:2])
        object_mask[object_bbox[0]:object_bbox[1], object_bbox[2]:object_bbox[3]] = 1

        mask = torch.cat([union_mask.unsqueeze(0), subject_mask.unsqueeze(0), object_mask.unsqueeze(0)], 0)
        mask = spatial_transform(mask)

        if self.args.exp_num in {46, 47, 48, 53, 54, 55, 56, 57, 58, 59}:
            object_conf = torch.FloatTensor([object_conf])
            subject_conf = torch.FloatTensor([subject_conf])
        elif self.args.exp_num in {81, 82, 36}:
            object_dist = torch.FloatTensor(to_one_hot(object_id, args.num_objects))
            subject_dist = torch.FloatTensor(to_one_hot(subject_id, args.num_objects))
            object_id = torch.LongTensor([object_id])
            subject_id = torch.LongTensor([subject_id])
            predicate_id = torch.LongTensor([pred_id])
        elif self.args.exp_num in {66, 67, 68, 69, 79, 80}:
            object_conf = torch.FloatTensor([object_conf])
            subject_conf = torch.FloatTensor([subject_conf])
            object_dist = torch.FloatTensor(to_one_hot(object_id, args.num_objects))
            subject_dist = torch.FloatTensor(to_one_hot(subject_id, args.num_objects))
        elif self.args.exp_num in {11, 12, 25, 41, 42, 43, 45, 49, 50, 51, 64, 65}:
            object_dist = torch.FloatTensor(to_one_hot(object_id, args.num_objects))
            subject_dist = torch.FloatTensor(to_one_hot(subject_id, args.num_objects))
            label = torch.LongTensor([pred_id])
        elif self.args.exp_num in {76}:
            object_dist = torch.FloatTensor(to_one_hot(object_id, args.num_objects))
            subject_dist = torch.FloatTensor(to_one_hot(subject_id, args.num_objects))
            index = torch.LongTensor([index])
        else:
            triplet_embedding = torch.FloatTensor(triplet_embedding_dict[(subject_id, pred_id, object_id)])
            object_dist = torch.FloatTensor(to_one_hot(object_id, args.num_objects))
            subject_dist = torch.FloatTensor(to_one_hot(subject_id, args.num_objects))
            object_id = torch.LongTensor([object_id])
            subject_id = torch.LongTensor([subject_id])
            predicate_id = torch.LongTensor([pred_id])

        if self.args.exp_num in {11, 12, 25, 45, 53, 54, 55, 56, 79}:
            return (subject_img, union_img, object_img, mask), (img_id, subject_bbox, object_bbox)
        elif self.args.exp_num in {81}:
            return (subject_img, union_img, object_img, mask), (subject_id, predicate_id, object_id, transform_tensor)
        elif self.args.exp_num in {82}:
            return (subject_img, union_img, object_img, mask, subject_dist, object_dist), (subject_id, predicate_id, object_id, transform_tensor)
        elif self.args.exp_num in {46, 47, 48, 57, 58, 59, 69, 79}:
            return (subject_img, union_img, object_img, mask), (
                img_id, subject_bbox, object_bbox, subject_conf, object_conf)
        elif self.args.exp_num in {66, 67, 68, 80}:
            return (subject_img, union_img, object_img, mask, subject_dist, object_dist), (
                img_id, subject_bbox, object_bbox, subject_conf, object_conf)
        elif self.args.exp_num in {64, 65}:
            return (subject_img, union_img, object_img, mask, subject_dist, object_dist), (label,)
        elif self.args.exp_num in {41, 42, 43, 49, 50, 51}:
            return (subject_img, union_img, object_img, mask, subject_dist, object_dist), (
                img_id, subject_bbox, object_bbox)
        elif self.args.exp_num in {70}:
            return (subject_img, union_img, object_img, mask), (
                subject_id, predicate_id, object_id, triplet_embedding)
        elif self.args.exp_num in {76}:
            return (subject_img, union_img, object_img, mask, subject_dist, object_dist), index
        else:
            return (subject_img, union_img, object_img, mask, subject_dist, object_dist), (
                subject_id, predicate_id, object_id, triplet_embedding)

    def __len__(self):
        return len(self.examples)


class VRDModel(nn.Module):
    def __init__(self, args):
        super(VRDModel, self).__init__()

        self.args = args
        self.activate_fn = getattr(F, args.activation)

        self.visual_feature = getattr(models, args.reg_model_name)(pretrained=True).features
        self.spatial_feature = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=(5, 5), stride=(2, 2), padding=2),
            nn.ReLU(),
            nn.Conv2d(96, 128, kernel_size=(5, 5), stride=(2, 2), padding=2),
            nn.Conv2d(128, 64, kernel_size=(8, 8)),
            nn.ReLU()
        )

        if args.exp_num in {25, 26, 29, 30, 31, 32, 34, 36, 37, 38, 39, 40, 41, 42, 43, 44,
                            45, 48, 49, 50, 51, 52, 53, 56, 57, 66, 67, 68, 69, 76, 79, 80, 81, 82}:
            self.predicate_feature = nn.Linear(2208 + 64, args.num_predicates)
        elif args.exp_num in {21, 22}:
            self.predicate_feature = nn.Linear(2208 + 64, 512)
        elif args.exp_num in {11, 12, 46, 47, 54, 55, 58, 59}:
            self.predicate_features = nn.Linear(2208 + 64, args.num_predicates)

        if args.exp_num in {21, 22, 25, 26, 27, 28, 29, 30, 31, 32, 34, 36, 37, 38, 39, 40, 44, 45,
                            48, 52, 53, 56, 57, 69, 79, 81}:
            self.object_feature = VisualModel(args)
        elif args.exp_num in {11, 12, 46, 47, 54, 55, 58, 59}:
            self.reg_net = VisualModel(args)

        if args.exp_num in {25, 26, 29, 30, 31, 32, 34, 36, 37, 38, 39, 40, 43, 44, 45, 48, 51,
                            52, 53, 56, 57, 66, 67, 68, 69, 76, 79, 80, 81, 82}:
            self.transform_matrix = nn.Linear(args.num_objects * 2 + args.num_predicates, 4096)
        elif args.exp_num in {27, 28}:
            self.transform_matrix = nn.Linear(2208 * 3 + 64, 4096)
        elif args.exp_num in {23, 24}:
            self.transform_matrix = nn.Linear(2208 + 64, 4096)
        elif args.exp_num in {21, 22, 34}:
            self.transform_matrix = nn.Linear(512 * 3, 4096)

        if args.exp_num not in {11, 12, 46, 47, 58, 59}:
            self.subject_class = nn.Linear(4096, args.num_objects)
            self.predicate_class = nn.Linear(4096, args.num_predicates)
            self.object_class = nn.Linear(4096, args.num_objects)

        if args.exp_num in {32, 37, 38, 42, 50}:
            self.b_Wrr = nn.Linear(args.num_predicates, args.num_predicates)
            self.b_Wrs = nn.Linear(args.num_objects, args.num_predicates)
            self.b_Wro = nn.Linear(args.num_objects, args.num_predicates)

        if args.exp_num in {12, 31, 32, 36, 39, 40, 43, 44, 45, 47, 48, 51, 52, 53, 55,
                            56, 57, 59, 66, 67, 68, 69, 76, 81, 82}:
            self.Wrr = nn.Linear(args.num_predicates, args.num_predicates)
            self.Wrs = nn.Linear(args.num_objects, args.num_predicates)
            self.Wro = nn.Linear(args.num_objects, args.num_predicates)

    def forward(self, x):
        if self.args.exp_num in {11, 12, 25, 45, 46, 47, 48, 53, 54, 55, 56, 57, 58, 59, 69, 79, 81}:
            subject_image, union_image, object_image, mask = x
        else:
            subject_image, union_image, object_image, mask, subject_dist, object_dist = x

        x_visual = self.visual_feature(union_image)
        x_visual = F.avg_pool2d(F.relu(x_visual, inplace=True), kernel_size=7).view(x_visual.size(0), -1)
        x_spatial = self.spatial_feature(mask).view(-1, 64)

        if self.args.exp_num in {21, 22, 25, 26, 29, 30, 31, 32, 34, 36, 37, 38, 39, 40, 41,
                                 42, 43, 44, 45, 48, 49, 50, 51, 52, 53, 56, 57, 66, 67, 68, 69, 76, 79, 80, 81, 82}:
            x_r = self.predicate_feature(torch.cat([x_visual, x_spatial], 1))
        elif self.args.exp_num in {11, 12, 46, 47, 54, 55, 58, 59}:
            x_r = self.predicate_features(torch.cat([x_visual, x_spatial], 1))

        if self.args.exp_num in {21, 22, 25, 26, 27, 28, 29, 30, 31, 32, 34, 36, 37, 38, 39, 40, 44, 45, 48, 52,
                                 53, 56, 57, 69, 79, 81}:
            x_s = self.object_feature(subject_image)
            x_o = self.object_feature(object_image)
        elif self.args.exp_num in {11, 12, 46, 47, 54, 55, 58, 59}:
            x_s = self.reg_net(subject_image)
            x_o = self.reg_net(object_image)
        elif self.args.exp_num in {41, 42, 43, 49, 50, 51, 66, 67, 68, 76, 80, 82}:
            x_s = subject_dist
            x_o = object_dist

        if self.args.exp_num in {43, 51, 66, 67, 68, 76, 80, 82}:
            x_s_ = x_s
            x_o_ = x_o

        if self.args.exp_num in {32, 37, 38, 42, 50}:
            x_rs = self.b_Wrs(x_s)
            x_ro = self.b_Wro(x_o)
            # Message Passing

            for i in range(self.args.num_iterations):
                x_r = self.activate_fn(self.b_Wrr(x_r) + x_rs + x_ro)

        # project to semantic space
        if self.args.exp_num in {21, 22, 25, 26, 29, 30, 31, 32, 34, 36, 37, 38, 39, 40, 43, 44, 45, 48, 51, 52,
                                 53, 56, 57, 66, 67, 68, 69, 76, 79, 80, 81, 82}:
            x_content = torch.cat([x_s, x_r, x_o], 1)
        elif self.args.exp_num in {27, 28}:
            x_content = torch.cat([x_s, x_visual, x_spatial, x_o], 1)
        elif self.args.exp_num in {23, 24}:
            x_content = torch.cat([x_visual, x_spatial], 1)

        if self.args.exp_num not in {11, 12}:
            x_semantic = self.transform_matrix(x_content)

        if self.args.exp_num not in {11, 12, 38, 41, 42, 46, 47, 49, 50, 58, 59}:
            x_s = self.subject_class(x_semantic)
            x_r = self.predicate_class(x_semantic)
            x_o = self.object_class(x_semantic)

        if self.args.exp_num in {12, 31, 32, 36, 39, 40, 43, 44, 45, 47, 48, 51, 52,
                                 53, 55, 56, 57, 59, 66, 67, 68, 69, 76, 81, 82}:
            x_rs = self.Wrs(x_s)
            x_ro = self.Wro(x_o)
            # Message Passing

            for i in range(self.args.num_iterations):
                x_r = self.activate_fn(self.Wrr(x_r) + x_rs + x_ro)

        if self.args.exp_num in {43, 51, 66, 67, 68, 76, 80, 82}:
            x_s = x_s_
            x_o = x_o_

        if self.args.mode == 'test':
            return x_s, x_r, x_o
        else:
            return x_s, x_r, x_o, x_semantic


def cosine_loss(input1, input2):
    output = F.cosine_similarity(input1, input2).sum() / input1.size(0)
    # print(output.data[0])
    return output


def special_loss(input, target):
    num_examples = target.size(0)
    subject, predicate, object_ = tuple(F.softmax(x) for x in input)

    gt_subject_score = subject.gather(1, target[:, 0].unsqueeze(1))
    gt_predicate_score = predicate.gather(1, target[:, 1].unsqueeze(1))
    gt_object_score = object_.gather(1, target[:, 2].unsqueeze(1))
    gt_score = gt_subject_score * gt_predicate_score * gt_object_score

    subject_score, _ = subject.max(1)
    predicate_score, _ = predicate.max(1)
    object_score, _ = object_.max(1)
    highest_score = subject_score * predicate_score * object_score

    return_loss = torch.sum(gt_score - highest_score) / num_examples

    return return_loss


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
    parser.add_argument('--mode', type=str, default='test', metavar='N',
                        help='mode: train or test (default: train)')
    parser.add_argument('--num_objects', type=int, default=100, metavar='N',
                        help='num of object class')
    parser.add_argument('--num_iterations', type=int, default=10, metavar='N',
                        help='num of iteration in message passing')
    parser.add_argument('--num_predicates', type=int, default=70, metavar='N',
                        help='num of predicate class')
    parser.add_argument('--num-workers', type=int, default=1, metavar='N',
                        help='num of workers to fetch daa')
    parser.add_argument('--exp-num', type=int, default=68, metavar='N',
                        help='Experiment number to run')
    parser.add_argument('--resume-train', type=bool, default=False, metavar='N',
                        help='Resume training')
    parser.add_argument('--use-softmax', type=bool, default=True, metavar='N',
                        help='use softmax for result or not')

    args = parser.parse_args()
    args.env_name = args.env_name.format(args.exp_num)

    if args.exp_num in [40]:
        with open('data/triplet_embedding_100000.pkl', 'rb') as f:
            triplet_embedding_dict = pickle.load(f)
    elif args.mode == 'test':
        args.batch_size = 256
    else:
        with open('data/vrd_triplet_embedding.pkl', 'rb') as f:
            triplet_embedding_dict = pickle.load(f)

    print(args)

    if args.exp_num in {60, 61, 62, 64, 65, 70}:
        # train_embedding model

        if args.exp_num in [64, 65]:
            net = PairFiltering(args)
            loss = nn.CrossEntropyLoss()
            monitor_outputs = [0]
        elif args.exp_num in [70]:
            net = RedesignedModel(args)
            loss = [nn.CrossEntropyLoss()] * 3 + [nn.SmoothL1Loss(), special_loss]
        else:
            net = RedesignedModel(args)
            loss = [nn.CrossEntropyLoss()] + [nn.SmoothL1Loss()]
            monitor_outputs = [1]

        model = Model(net, args)
        model.train()

        train_dataset = VRDDataset('train', args)
        val_dataset = VRDDataset('test', args)

        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

        dset_loaders = {'train': train_loader, 'val': val_loader}
        dset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}

        optimizer = getattr(optim, args.optimizer)(net.parameters(), args.lr)
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2, verbose=True)

        model.compile(loss, optimizer, scheduler=scheduler)

        folder_checkpoint = "model_checkpoints/" + args.env_name.replace(' ', '_') + '/'

        if not os.path.exists(folder_checkpoint):
            os.mkdir(folder_checkpoint)

        model_checkpoint = folder_checkpoint + "weights.{epoch:02d}-train_loss:{train_loss:.2f}" \
                           + "-train_acc:{train_acc:.2f}-val_loss:{val_loss:.2f}-val_acc:{val_acc:.2f}.pkl"

        monitor_outputs = [0, 1, 2]
        display_acc = 1

        loss_weights = None

        model.fit_loader(dset_loaders, dset_sizes, env=args.env_name, batch_size=args.batch_size,
                         evaluate_fn=None, num_epochs=args.epochs, model_checkpoint=model_checkpoint,
                         monitor_outputs=monitor_outputs, display_acc=display_acc, loss_weights=loss_weights)

        # end train_embedding model

    else:

        if args.mode == 'train':
            net = RedesignedModel(args) if args.exp_num in {63} else VRDModel(args)
            model = Model(net, args)

            if args.resume_train:
                model.load('model_checkpoints/Experiment_' + str(args.exp_num)
                           + '/weights.30-train_loss:0.20-train_acc:0.90-val_loss:0.50-val_acc:0.72.pkl')
            model.train()

            # Load datasets
            val_dataset = VRDDataset('test', args)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

            train_dataset = VRDDataset('train', args)
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

            dset_loaders = {'train': train_loader, 'val': val_loader}
            dset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}

            if args.exp_num in {45, 52}:
                embedding_loss = nn.SmoothL1Loss()
            elif args.exp_num in {31}:
                embedding_loss = nn.MSELoss()
            else:
                embedding_loss = cosine_loss

            loss = [nn.CrossEntropyLoss()] * 3 + [embedding_loss]

            optimizer = getattr(optim, args.optimizer)(net.parameters(), args.lr)

            scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2, verbose=True)

            model.compile(loss, optimizer, scheduler=scheduler)

            folder_checkpoint = "model_checkpoints/" + args.env_name.replace(' ', '_') + '/'

            if not os.path.exists(folder_checkpoint):
                os.mkdir(folder_checkpoint)

            if args.exp_num in [34]:
                model_checkpoint = folder_checkpoint + "weights.{epoch:02d}-train_loss:{train_loss:.2f}" \
                                   + "-valid_loss:{val_loss:.2f}.pkl"
                monitor_outputs = None
                display_acc = None
            else:
                model_checkpoint = folder_checkpoint + "weights.{epoch:02d}-train_loss:{train_loss:.2f}" \
                                   + "-train_acc:{train_acc:.2f}-val_loss:{val_loss:.2f}-val_acc:{val_acc:.2f}.pkl"

                monitor_outputs = [0, 1, 2]
                display_acc = 1

            loss_weights = None

            model.fit_loader(dset_loaders, dset_sizes, env=args.env_name, batch_size=args.batch_size, evaluate_fn=None,
                             num_epochs=args.epochs, model_checkpoint=model_checkpoint, monitor_outputs=monitor_outputs,
                             display_acc=display_acc, loss_weights=loss_weights)

        else:

            if args.exp_num in {81, 82}:
                net = RedesignedModel(args) if args.exp_num in {63} else VRDModel(args)
                model = Model(net, args)

                model.load('model_checkpoints/Experiment_' + str(args.exp_num) + '/official_weights.pkl')
                model.eval()

                dataset = VRDDataset('test', args)
                loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)

                list_embedding_output = []
                list_label_output = []
                list_tensor_output = []

                for k, example in enumerate(loader):
                    print(k, len(loader))
                    list_inputs, (subject_id, predicate_id, object_id, tensor) = example
                    # list_inputs = [Variable(x.cuda(), volatile=True) for x in list_inputs]
                    #
                    # subject, predicate, object_, embedding = model.net.forward(list_inputs)
                    #
                    # _, predict_subject_id = subject.data.max(1)
                    # _, predict_predicate_id = predicate.data.max(1)
                    # _, predict_object_id = object_.data.max(1)

                    for i in range(subject_id.size(0)):
                        # embedding_ = (100 * embedding[i, :].data).round().long()
                        # list_embedding_output.append(embedding_.tolist())
                        gt_triplet = id_to_object[int(subject_id[i, 0])] + '-' + id_to_predicate[
                            int(predicate_id[i, 0])] + '-' + id_to_object[int(object_id[i, 0])]
                        # predict_triplet = id_to_object[int(predict_subject_id[i])] + '-' + id_to_predicate[
                        #     int(predict_predicate_id[i])] + '-' + id_to_object[int(predict_object_id[i])]
                        list_label_output.append([gt_triplet])
                        # list_tensor_output.append(tensor[i:i+1, :])

                # with open('data/vrd_test_tensor_experiment_{}.tsv'.format(args.exp_num), 'w') as f:
                #     wr = csv.writer(f, delimiter='\t')
                #     wr.writerows(list_embedding_output)
                #
                with open('data/vrd_test_labels_experiment_{}.tsv'.format(args.exp_num), 'w') as f:
                    wr = csv.writer(f, delimiter='\t')
                    wr.writerows(list_label_output)

                # list_tensor_output.extend([torch.zeros(1, 3, 93, 93)] * (88 * 88 - 7638))
                #
                # tensor = torch.cat(list_tensor_output, 0)
                #
                # tensor = tensor.cpu()
                # grid = torchvision.utils.make_grid(tensor, nrow=88, padding=0)
                # ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
                # ndarr = ndarr[1:, 1:, :]
                # # cv2.imshow('test', ndarr)
                # # cv2.waitKey()
                # cv2.imwrite('data/vrd_test_sprite_experiment_{}.png'.format(args.exp_num), ndarr)
            else:
                dict_points = {}
                modes = ['test']

                for mode in modes:
                    if args.use_softmax:
                        result_file = 'result_softmax/experiment_' + str(args.exp_num) + '_{}.pkl'.format(mode)
                    else:
                        result_file = 'result/experiment_' + str(args.exp_num) + '_{}.pkl'.format(mode)

                    with open('data/list_{}_images.pkl'.format(mode), 'rb') as f:
                        list_images = pickle.load(f)

                    top_hs = [1] if args.exp_num in {81, 82} else [1, 2, 5, 8, 20]

                    last_top_h = top_hs[-1]
                    save_number = 5000
                    if args.exp_num in {68, 69, 79, 80}:
                        box_file = 'data/vrd_test_data_combine_detector_box.pkl'
                    elif args.exp_num in {12, 41, 42, 43, 45, 63}:
                        box_file = 'data/vrd_{}_gt_boxes.pkl'.format(mode)

                    with open(box_file, 'rb') as f:
                        boxes = pickle.load(f)
                        dict_boxes_to_indices = {image_id: {x: index for index, x in enumerate(boxes[image_id])}
                                                 for image_id in list_images}
                        dict_indices_to_boxes = {image_id: {index: x for index, x in enumerate(boxes[image_id])}
                                                 for image_id in list_images}

                    if os.path.isfile(result_file):
                        with open(result_file, 'rb') as f:
                            results = pickle.load(f)
                    else:

                        net = RedesignedModel(args) if args.exp_num in {63} else VRDModel(args)
                        model = Model(net, args)

                        model.load('model_checkpoints/Experiment_' + str(args.exp_num) + '/official_weights.pkl')
                        model.eval()

                        test_image_dict = {image_id: i for i, image_id in enumerate(list_images)}
                        results = {top_h: {image_id: {'list_tuples': [],
                                                      'list_tuple_confs': [],
                                                      'list_sub_bboxes': [], 'list_obj_bboxes': []}
                                           for image_id in list_images}
                                   for top_h in top_hs}

                        dataset = VRDDataset(mode, args)
                        loader = DataLoader(dataset, batch_size=args.batch_size,
                                            num_workers=args.num_workers)

                        for k, example in enumerate(loader):
                            print(k, len(loader))
                            if args.exp_num in {41, 42, 43, 11, 12, 45, 63, 25}:
                                list_inputs, (image_ids, subject_bboxes_, object_bboxes_) = example
                            else:
                                list_inputs, (
                                    image_ids, subject_bboxes_, object_bboxes_, subject_conf, object_conf) = example
                                subject_conf = subject_conf.view(-1).tolist()
                                object_conf = object_conf.view(-1).tolist()

                            subject_bboxes = torch.cat([x.unsqueeze(1) for x in subject_bboxes_], 1)
                            object_bboxes = torch.cat([x.unsqueeze(1) for x in object_bboxes_], 1)

                            list_inputs = [Variable(x.cuda(), volatile=True) for x in list_inputs]

                            subject, predicate, object_ = model.net.forward(list_inputs)

                            if args.use_softmax:
                                subject = F.softmax(subject)
                                predicate = F.softmax(predicate)
                                object_ = F.softmax(object_)

                            conf_ss, index_ss = subject.sort(1, descending=True)
                            conf_ps, index_ps = predicate.sort(1, descending=True)
                            conf_os, index_os = object_.sort(1, descending=True)

                            for i, image_id in enumerate(image_ids):
                                if args.exp_num in {41, 42, 43, 63, 66, 67, 68, 80}:
                                    data_s_ = torch.cat((conf_ss[i, :1].view(-1, 1),
                                                         index_ss[i, :1].view(-1, 1).float()), 1).data.cpu().numpy()
                                    data_p_ = torch.cat((conf_ps[i, :last_top_h].view(-1, 1),
                                                         index_ps[i, :last_top_h].view(-1, 1).float()),
                                                        1).data.cpu().numpy()
                                    data_o_ = torch.cat((conf_os[i, :1].view(-1, 1),
                                                         index_os[i, :1].view(-1, 1).float()), 1).data.cpu().numpy()
                                else:
                                    data_s_ = torch.cat((conf_ss[i, :last_top_h].view(-1, 1),
                                                         index_ss[i, :last_top_h].view(-1, 1).float()),
                                                        1).data.cpu().numpy()
                                    data_p_ = torch.cat((conf_ps[i, :last_top_h].view(-1, 1),
                                                         index_ps[i, :last_top_h].view(-1, 1).float()),
                                                        1).data.cpu().numpy()
                                    data_o_ = torch.cat((conf_os[i, :last_top_h].view(-1, 1),
                                                         index_os[i, :last_top_h].view(-1, 1).float()),
                                                        1).data.cpu().numpy()

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
                                s_p_o_scores_tuples = [(s, p, o) for s in data_s[:, 0]
                                                       for p in data_p[:, 0]
                                                       for o in data_o[:, 0]]

                                s_p_o_scores, s_p_o_indices = torch.FloatTensor(s_p_o_scores_triplets).sort(0,
                                                                                                            descending=True)

                                for top_h in top_hs:
                                    confs = s_p_o_scores[:top_h].tolist()
                                    num_triplets = len(confs)

                                    tuple_confs = []
                                    for m in s_p_o_indices[:top_h]:
                                        # if args.exp_num in {66, 67, 68, 80}:
                                        #     tuple_confs.append(
                                        #         (subject_conf[i], s_p_o_scores_tuples[m][1], object_conf[i]))
                                        # else:
                                        tuple_confs.append(s_p_o_scores_tuples[m])

                                    triplets = [dict_triplets_to_indices[s_p_o_indices_triplets[m]] for m in
                                                s_p_o_indices[:top_h]]

                                    s_boxes = [dict_boxes_to_indices[image_id][
                                                   tuple(subject_bboxes[i, :].tolist())]] * num_triplets
                                    o_boxes = [dict_boxes_to_indices[image_id][
                                                   tuple(object_bboxes[i, :].tolist())]] * num_triplets

                                    results[top_h][image_id]['list_tuple_confs'].extend(tuple_confs)
                                    results[top_h][image_id]['list_tuples'].extend(triplets)
                                    results[top_h][image_id]['list_sub_bboxes'].extend(s_boxes)
                                    results[top_h][image_id]['list_obj_bboxes'].extend(o_boxes)

                        for top_h in top_hs:
                            for i, image_id in enumerate(list_images):
                                info = results[top_h][image_id]
                                labels = info['list_tuples']
                                boxSub = info['list_sub_bboxes']
                                boxObj = info['list_obj_bboxes']
                                tuple_confs = info['list_tuple_confs']

                                # confs = [compute_score(tuple_conf) for tuple_conf in tuple_confs]
                                if len(tuple_confs) > 0:
                                    _, ind = torch.from_numpy(np.array(tuple_confs)).cuda().prod(1).sort(0,
                                                                                                         descending=True)

                                    ind = ind[:save_number]

                                    results[top_h][image_id]['list_tuple_confs'] = get_sublist(tuple_confs, ind)
                                    results[top_h][image_id]['list_tuples'] = get_sublist(labels, ind)
                                    results[top_h][image_id]['list_sub_bboxes'] = get_sublist(boxSub, ind)
                                    results[top_h][image_id]['list_obj_bboxes'] = get_sublist(boxObj, ind)

                        with open(result_file, 'wb') as f:
                            pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)

                        print('complete dumping file')

                    for top_h in top_hs:
                        print('Top {}'.format(top_h))
                        top_recall_phrase(results[top_h], dict_indices_to_triplets, dict_indices_to_boxes, mode='test',
                                          zeroshot=True)

                        top_recall_relationship(results[top_h], dict_indices_to_triplets,
                                                dict_indices_to_boxes, mode='test', zeroshot=True)

