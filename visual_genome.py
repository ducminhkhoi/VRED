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
import os
import argparse
import numpy as np
from itertools import chain
from extract_data import get_rcnn_results
from evaluate import top_recall_phrase, top_recall_relationship, compute_score, get_sublist
from collections import defaultdict
from copy import deepcopy
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from visdom import Visdom
viz = Visdom()

image_folder = '/scratch/datasets/Data1/visual_genome/VG_100K_images/{}.jpg'

with open('data/vg_meta_data.pkl', 'rb') as f:
    meta_data = pickle.load(f)
    list_predicates = meta_data['list_predicates']
    list_objects = meta_data['list_objects']
    list_train_files = meta_data['list_train_files']
    list_test_files = meta_data['list_test_files']
    gt_boxes_to_indices = meta_data['gt_boxes_to_indices']
    gt_indices_to_boxes = meta_data['gt_indices_to_boxes']
    triplet_embedding_dict = meta_data['triplet_embedding']

choose_box_file = 'data/vg_test_faster_rcnn_box.pkl'
with open(choose_box_file, 'rb') as f:
    dict_boxes = pickle.load(f)

boxes_to_indices = {key: {b: i for i, b in enumerate(value)} for key, value in dict_boxes.items()}
indices_to_boxes = {key: {i: b for i, b in enumerate(value)} for key, value in dict_boxes.items()}

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


class PairFiltering(nn.Module):
    """Classify a pair of bounding boxes is a relationship or not"""
    def __init__(self, args):
        super(PairFiltering, self).__init__()

        self.spatial_feature = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=(5, 5), stride=(2, 2), padding=2),
            nn.ReLU(),
            nn.Conv2d(96, 128, kernel_size=(5, 5), stride=(2, 2), padding=2),
            nn.Conv2d(128, 64, kernel_size=(8, 8)),
            nn.ReLU()
        )

        self.fc1 = nn.Linear(64 + 2 * args.num_objects, 256)
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


class VGDataset(data.Dataset):
    def __init__(self, mode, args=None):

        self.image_path = '/home/khoinguyen/data/visual_genome/VG_100K_images/{}.jpg'
        if mode == 'train':
            self.transform_fn = transform_reg_train
            if args.exp_num in {75}:
                data_file = 'data/vg_train_data_pair_filtering.pkl'
            else:
                data_file = 'data/vg_train_data.pkl'
        else:
            self.transform_fn = transform_reg_test

            if args.exp_num in {75}:
                if args.mode == 'train':
                    data_file = 'data/vg_test_data_pair_filtering.pkl'
                else:
                    data_file = 'data/vg_test_faster_rcnn_detector.pkl'
            elif args.exp_num in {73, 74}:
                data_file = 'data/vg_test_faster_rcnn_new_detector.pkl'
            elif args.exp_num in {77, 78}:
                data_file = 'data/vg_test_gt_data.pkl'
            else:
                data_file = 'data/vg_test_data.pkl'

        self.args = args

        with open(data_file, 'rb') as f:
            self.examples = pickle.load(f)
            print(len(self.examples))

    def __getitem__(self, index):
        if self.args.exp_num in {73, 74, 75}:
            img_id, union_bbox, object_bbox, subject_bbox, object_id, subject_id, object_conf, subject_conf = self.examples[index]
            pred_id = None
        else:
            img_id, union_bbox, object_bbox, subject_bbox, object_id, subject_id, pred_id = self.examples[index]
            object_conf = None
            subject_conf = None

            subject_bbox = gt_indices_to_boxes[img_id][subject_bbox]
            object_bbox = gt_indices_to_boxes[img_id][object_bbox]

        img = cv2.imread(self.image_path.format(img_id), cv2.IMREAD_COLOR)

        if self.args.exp_num not in {75}:
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

            union_img = img[new_u_bbox[0]:new_u_bbox[1], new_u_bbox[2]:new_u_bbox[3]]
            union_img = self.transform_fn(union_img)

            object_img = img[new_o_bbox[0]:new_o_bbox[1], new_o_bbox[2]:new_o_bbox[3]]
            object_img = self.transform_fn(object_img)

            subject_img = img[new_s_bbox[0]:new_s_bbox[1], new_s_bbox[2]:new_s_bbox[3]]
            subject_img = self.transform_fn(subject_img)

        union_mask = torch.zeros(*img.shape[:2])
        union_mask[union_bbox[0]: union_bbox[1], union_bbox[2]:union_bbox[3]] = 1

        subject_mask = torch.zeros(*img.shape[:2])
        subject_mask[subject_bbox[0]:subject_bbox[1], subject_bbox[2]:subject_bbox[3]] = 1

        object_mask = torch.zeros(*img.shape[:2])
        object_mask[object_bbox[0]:object_bbox[1], object_bbox[2]:object_bbox[3]] = 1

        mask = torch.cat([union_mask.unsqueeze(0), subject_mask.unsqueeze(0), object_mask.unsqueeze(0)], 0)
        mask = spatial_transform(mask)

        if object_conf:
            object_conf = torch.FloatTensor([object_conf])
            subject_conf = torch.FloatTensor([subject_conf])

        if self.args.exp_num in {71, 72}:
            triplet_embedding = torch.FloatTensor(triplet_embedding_dict[(subject_id, pred_id, object_id)])

        object_dist = torch.FloatTensor(to_one_hot(object_id, args.num_objects))
        subject_dist = torch.FloatTensor(to_one_hot(subject_id, args.num_objects))
        object_id = torch.LongTensor([int(object_id)])
        subject_id = torch.LongTensor([int(subject_id)])

        if pred_id is not None:
            predicate_id = torch.LongTensor([int(pred_id)])

        if self.args.exp_num in {71, 72}:
            try:
                return (subject_img, union_img, object_img, mask, subject_dist, object_dist), \
                    (subject_id, predicate_id, object_id, triplet_embedding)
            except UnboundLocalError:
                print()
        elif self.args.exp_num in {75}:
            return (mask, subject_dist, object_dist), (pred_id if self.args.mode == 'train' else index, )
        elif self.args.exp_num in {77, 78}:
            return (subject_img, union_img, object_img, mask, subject_dist, object_dist), \
                   (img_id, subject_bbox, object_bbox)
        else:
            return (subject_img, union_img, object_img, mask, subject_dist, object_dist), \
                (img_id, subject_bbox, object_bbox, subject_conf, object_conf)

    def __len__(self):
        return len(self.examples)


class VGModel(nn.Module):
    def __init__(self, args):
        super(VGModel, self).__init__()

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

        self.predicate_feature = nn.Linear(2208 + 64, args.num_predicates)

        if args.exp_num in {71, 73, 77}:
            self.object_feature = VisualModel(args)

        self.transform_matrix = nn.Linear(args.num_objects*2 + args.num_predicates, 4096)

        self.subject_class = nn.Linear(4096, args.num_objects)
        self.predicate_class = nn.Linear(4096, args.num_predicates)
        self.object_class = nn.Linear(4096, args.num_objects)

        # self.Wrr = nn.Linear(args.num_predicates, args.num_predicates)
        # self.Wrs = nn.Linear(args.num_objects, args.num_predicates)
        # self.Wro = nn.Linear(args.num_objects, args.num_predicates)

    def forward(self, x):
        subject_image, union_image, object_image, mask, subject_dist, object_dist = x

        x_visual = self.visual_feature(union_image)
        x_visual = F.avg_pool2d(F.relu(x_visual, inplace=True), kernel_size=7).view(x_visual.size(0), -1)
        x_spatial = self.spatial_feature(mask).view(-1, 64)

        x_r = self.predicate_feature(torch.cat([x_visual, x_spatial], 1))

        if self.args.exp_num in {71, 73, 77}:
            x_s = self.object_feature(subject_image)
            x_o = self.object_feature(object_image)
        elif self.args.exp_num in {72, 74, 78}:
            x_s = subject_dist
            x_o = object_dist

        if self.args.exp_num in {72, 74, 78}:
            x_s_ = x_s
            x_o_ = x_o

        # project to semantic space
        x_semantic = self.transform_matrix(torch.cat([x_s, x_r, x_o], 1))

        x_s = self.subject_class(x_semantic)
        x_r = self.predicate_class(x_semantic)
        x_o = self.object_class(x_semantic)

        # x_rs = self.Wrs(x_s)
        # x_ro = self.Wro(x_o)
        # # Message Passing
        #
        # for i in range(self.args.num_iterations):
        #     x_r = self.activate_fn(self.Wrr(x_r) + x_rs + x_ro)

        if self.args.exp_num in {72, 74, 78}:
            x_s = x_s_
            x_o = x_o_

        if self.args.exp_num in {73, 74, 77, 78}:
            return x_s, x_r, x_o
        else:
            return x_s, x_r, x_o, x_semantic


def cosine_loss(input1, input2):
    output = F.cosine_similarity(input1, input2).sum()/input1.size(0)
    # print(output.data[0])
    return output


if __name__ == '__main__':

    # Options to run
    parser = argparse.ArgumentParser(description='VG Dataset Relational Network')

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
    parser.add_argument('--num_objects', type=int, default=201, metavar='N',
                        help='num of object class')
    parser.add_argument('--num_iterations', type=int, default=10, metavar='N',
                        help='num of iteration in message passing')
    parser.add_argument('--num_predicates', type=int, default=100, metavar='N',
                        help='num of predicate class')
    parser.add_argument('--num-workers', type=int, default=8, metavar='N',
                        help='num of workers to fetch data')
    parser.add_argument('--exp-num', type=int, default=74, metavar='N',
                        help='Experiment number to run')
    parser.add_argument('--resume-train', type=bool, default=False, metavar='N',
                        help='Resume training')
    parser.add_argument('--use-softmax', type=bool, default=True, metavar='N',
                        help='use softmax for result or not')

    args = parser.parse_args()
    args.env_name = args.env_name.format(args.exp_num)

    if args.exp_num in {73, 74, 75, 77, 78}:
        args.batch_size = 256

    print(args)

    if args.exp_num in {75}:
        # train_embedding model

        net = PairFiltering(args)
        model = Model(net, args)

        if args.mode == 'train':

            model.train()

            train_dataset = VGDataset('train', args)
            val_dataset = VGDataset('test', args)

            train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

            dset_loaders = {'train': train_loader, 'val': val_loader}
            dset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}

            optimizer = getattr(optim, args.optimizer)(net.parameters(), args.lr)
            scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2, verbose=True)
            loss = nn.CrossEntropyLoss()

            model.compile(loss, optimizer, scheduler=scheduler)

            folder_checkpoint = "model_checkpoints/" + args.env_name.replace(' ', '_') + '/'

            if not os.path.exists(folder_checkpoint):
                os.mkdir(folder_checkpoint)

            model_checkpoint = folder_checkpoint + "weights.{epoch:02d}-train_loss:{train_loss:.2f}" \
                               + "-train_acc:{train_acc:.2f}-val_loss:{val_loss:.2f}-val_acc:{val_acc:.2f}.pkl"

            monitor_outputs = [0]
            display_acc = 0

            loss_weights = None

            model.fit_loader(dset_loaders, dset_sizes, env=args.env_name, batch_size=args.batch_size,
                             evaluate_fn=None, num_epochs=args.epochs, model_checkpoint=model_checkpoint,
                             monitor_outputs=monitor_outputs, display_acc=display_acc, loss_weights=loss_weights)

        else:  # Filter pairs
            model.load('model_checkpoints/Experiment_' + str(args.exp_num) + '/official_weights.pkl')
            model.eval()

            pair_filter_file = 'data/vg_filter_pair_data.pkl'

            if os.path.isfile(pair_filter_file):
                with open(pair_filter_file, 'rb') as f:
                    list_keep_indices = pickle.load(f)

            else:
                val_dataset = VGDataset('test', args)
                val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

                list_keep_indices = []

                print(len(val_loader))

                for k, example in enumerate(val_loader):
                    print(k, len(val_loader))

                    list_inputs, list_indices = example
                    list_indices = list_indices[0].tolist()

                    list_inputs = [Variable(x.cuda(), volatile=True) for x in list_inputs]

                    output = model.net.forward(list_inputs)

                    _, result = output.max(1)

                    keep_indices_ = result.data.nonzero().view(-1).tolist()
                    keep_indices = [list_indices[i] for i in keep_indices_]

                    list_keep_indices.extend(keep_indices)

                with open(pair_filter_file, 'wb') as f:
                    pickle.dump(list_keep_indices, f, pickle.HIGHEST_PROTOCOL)

            data_file = 'data/vg_test_faster_rcnn_detector.pkl'
            new_data_file = 'data/vg_test_faster_rcnn_new_detector.pkl'
            with open(data_file, 'rb') as f:
                examples = pickle.load(f)

            print(len(examples))

            examples = [examples[i] for i in list_keep_indices]

            print(len(examples))

            with open(new_data_file, 'wb') as f:
                pickle.dump(examples, f, pickle.HIGHEST_PROTOCOL)

    else:

        if args.mode == 'train':
            net = VGModel(args)
            model = Model(net, args)
            if args.resume_train:
                model.load('model_checkpoints/Experiment_' + str(args.exp_num)
                           + '/official_weights.pkl')
            model.train()

            # Load datasets
            val_dataset = VGDataset('test', args)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

            train_dataset = VGDataset('train', args)
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

            dset_loaders = {'train': train_loader, 'val': val_loader}
            dset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}

            embedding_loss = nn.SmoothL1Loss()

            loss = [nn.CrossEntropyLoss()] * 3 + [embedding_loss]

            optimizer = getattr(optim, args.optimizer)(net.parameters(), args.lr)

            scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2, verbose=True)

            model.compile(loss, optimizer, scheduler=scheduler)

            folder_checkpoint = "model_checkpoints/" + args.env_name.replace(' ', '_') + '/'

            if not os.path.exists(folder_checkpoint):
                os.mkdir(folder_checkpoint)

            model_checkpoint = folder_checkpoint + "weights.{epoch:02d}-train_loss:{train_loss:.2f}" \
                + "-train_acc:{train_acc:.2f}-val_loss:{val_loss:.2f}-val_acc:{val_acc:.2f}.pkl"

            monitor_outputs = [1]
            display_acc = 0

            loss_weights = None

            model.fit_loader(dset_loaders, dset_sizes, env=args.env_name, batch_size=args.batch_size, evaluate_fn=None,
                             num_epochs=args.epochs, model_checkpoint=model_checkpoint, monitor_outputs=monitor_outputs,
                             display_acc=display_acc, loss_weights=loss_weights)

        else:
            modes = ['test']

            for mode in modes:
                if args.use_softmax:
                    result_file = 'result_softmax/experiment_' + str(args.exp_num) + '_{}.pkl'.format(mode)
                else:
                    result_file = 'result/experiment_' + str(args.exp_num) + '_{}.pkl'.format(mode)

                if mode == 'train':
                    list_images = list_train_files
                else:
                    list_images = list_test_files

                top_hs = [1, 2, 5, 8]

                last_top_h = top_hs[-1]
                save_number = 5000

                if os.path.isfile(result_file):
                    with open(result_file, 'rb') as f:
                        results = pickle.load(f)
                else:
                    net = VGModel(args)
                    model = Model(net, args)
                    model.load('model_checkpoints/Experiment_' + str(args.exp_num) + '/official_weights.pkl')
                    model.eval()

                    test_image_dict = {image_id: i for i, image_id in enumerate(list_images)}
                    results = {top_h: {image_id: {'list_tuples': [],
                                                  'list_tuple_confs': [],
                                                  'list_sub_bboxes': [], 'list_obj_bboxes': []}
                                      for image_id in list_images}
                               for top_h in top_hs}

                    dataset = VGDataset(mode, args)
                    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)

                    for k, example in enumerate(loader):
                        print(k, len(loader))

                        if args.exp_num in {77, 78}:
                            list_inputs, (image_ids, subject_bboxes_, object_bboxes_) = example
                        else:
                            list_inputs, (image_ids, subject_bboxes_, object_bboxes_, subject_conf, object_conf) = example

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
                            if args.exp_num in {74}:
                                data_s_ = torch.cat((conf_ss[i, :1].view(-1, 1),
                                                     index_ss[i, :1].view(-1, 1).float()), 1).data.cpu().numpy()
                                data_p_ = torch.cat((conf_ps[i, :last_top_h].view(-1, 1),
                                                     index_ps[i, :last_top_h].view(-1, 1).float()), 1).data.cpu().numpy()
                                data_o_ = torch.cat((conf_os[i, :1].view(-1, 1),
                                                     index_os[i, :1].view(-1, 1).float()), 1).data.cpu().numpy()
                            else:
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
                            s_p_o_scores_tuples = [(s, p, o) for s in data_s[:, 0]
                                                   for p in data_p[:, 0]
                                                   for o in data_o[:, 0]]

                            s_p_o_scores, s_p_o_indices = torch.FloatTensor(s_p_o_scores_triplets).sort(0, descending=True)

                            for top_h in top_hs:
                                confs = s_p_o_scores[:top_h].tolist()
                                num_triplets = len(confs)

                                tuple_confs = []
                                for m in s_p_o_indices[:top_h]:
                                    if args.exp_num in {74}:
                                        tuple_confs.append(
                                            (subject_conf[i], s_p_o_scores_tuples[m][1], object_conf[i]))
                                    else:
                                        tuple_confs.append(s_p_o_scores_tuples[m])

                                triplets = [s_p_o_indices_triplets[m] for m in s_p_o_indices[:top_h]]

                                s_boxes = [tuple(subject_bboxes[i, :].tolist())] * num_triplets

                                o_boxes = [tuple(object_bboxes[i, :].tolist())] * num_triplets

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

                            if len(tuple_confs) > 0:
                                _, ind_ = torch.from_numpy(np.array(tuple_confs)).cuda().prod(1).sort(0, descending=True)
                                ind = ind_[:save_number]

                                results[top_h][image_id]['list_tuple_confs'] = get_sublist(tuple_confs, ind)
                                results[top_h][image_id]['list_tuples'] = get_sublist(labels, ind)
                                results[top_h][image_id]['list_sub_bboxes'] = get_sublist(boxSub, ind)
                                results[top_h][image_id]['list_obj_bboxes'] = get_sublist(boxObj, ind)

                    with open(result_file, 'wb') as f:
                        pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)

                    print('complete dumping file')

                for top_h in top_hs:
                    print('Top {}'.format(top_h))
                    top_recall_phrase(results[top_h], dict_indices_to_triplets, indices_to_boxes,
                                      mode=mode, dataset='vg', list_images=list_test_files)
                    #
                    top_recall_relationship(results[top_h], dict_indices_to_triplets, indices_to_boxes,
                                            mode=mode, dataset='vg', list_images=list_test_files)

