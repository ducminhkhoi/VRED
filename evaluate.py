import pickle
import torch
import numpy as np
import itertools
from extract_data import get_gt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import scipy.io as sio


with open('data/vrd_probability.pkl', 'rb') as f:
    probability = pickle.load(f)

gt_file = 'data/annotation_test.mat'
gt = sio.loadmat(gt_file)


def get_sublist(a, indices):
    return [a[k] for k in indices]


with open('data/vrd_mean_cov.pkl', 'rb') as f:
    dict_mean_cov = pickle.load(f)


# top_ks = [50, 100, 1000, 5000]
top_ks = [50, 100]


def compute_score(X, type=1):
    """type == 1: triplet product, 2: triplet + bbox conf product"""

    X = torch.from_numpy(np.array(X)).cuda()
    if type == 1:
        result = X.prod(1)
    else:
        X_1 = X.clone()
        X_2 = torch.cat([X[:, 1:2], X[:, 2:3], X[:, 0:1]], 1)
        X_2 *= X_1
        X_3 = X_1.prod(1).unsqueeze(1)
        result = torch.cat([X_1, X_2, X_3], 1).sum(1)

    return result.sort(0, descending=True)


def top_recall_relationship(data, dict_indices_to_triplets, dict_indices_to_boxes, mode='test',
                            zeroshot=False, dataset='vrd', list_images=None):
    if dataset == 'vrd':
        with open('data/list_{}_images.pkl'.format(mode), 'rb') as f:
            list_images = pickle.load(f)

    gt_threshold = 0.5
    gt_data = get_gt(mode, zeroshot, dataset)

    num_pos_tuple = 0

    tp_list = {num: [] for num in top_ks}
    count_correct = 0

    for i, image_id in enumerate(list_images):

        gt_info = gt_data[image_id]
        info = data[image_id]

        gt_tupLabel, gt_subBox, gt_objBox = gt_info['list_tuples'], gt_info['list_sub_bboxes'], gt_info[
            'list_obj_bboxes']

        gt_subBox = [tuple(x) for x in gt_subBox]
        gt_objBox = [tuple(x) for x in gt_objBox]

        if len(gt_tupLabel) == 0:
            continue

        num_gt_tuple = len(gt_tupLabel)

        num_pos_tuple += num_gt_tuple

        labels, boxSub, boxObj, tuple_confs = info['list_tuples'][:500], info['list_sub_bboxes'][:500], \
                                              info['list_obj_bboxes'][:500], info['list_tuple_confs'][:500]

        if len(labels) == 0:
            continue

        # print(len(tuple_confs))

        if dataset == 'vrd':
            labels = [dict_indices_to_triplets[index] for index in labels]
            boxSub = [tuple(dict_indices_to_boxes[image_id][index]) for index in boxSub]
            boxObj = [tuple(dict_indices_to_boxes[image_id][index]) for index in boxObj]

        confs, ind_ = compute_score(tuple_confs, type=1)

        for num in top_ks:
            ind = ind_[:num]

            labels_ = get_sublist(labels, ind)
            boxSub_ = get_sublist(boxSub, ind)
            boxObj_ = get_sublist(boxObj, ind)

            gt_detected = np.zeros(num_gt_tuple)

            num_obj = len(labels_)
            tp = np.zeros(num_obj)
            fp = np.zeros(num_obj)

            for j in range(num_obj):
                bbO = boxObj_[j]
                bbS = boxSub_[j]
                ovmax = -np.inf
                kmax = -1

                for k in range(num_gt_tuple):

                    # # already detect, don't count anymore
                    if gt_detected[k] > 0:
                        continue

                    label = labels_[j]
                    gt_label = gt_tupLabel[k]

                    if any(e1 != e2 for e1, e2 in zip(label, gt_label)):
                        continue

                    bbgtO = gt_objBox[k]
                    bbgtS = gt_subBox[k]

                    biO = max(bbO[0], bbgtO[0]), min(bbO[1], bbgtO[1]), max(bbO[2], bbgtO[2]), min(bbO[3], bbgtO[3])
                    iwO = biO[3] - biO[2] + 1
                    ihO = biO[1] - biO[0] + 1

                    biS = max(bbS[0], bbgtS[0]), min(bbS[1], bbgtS[1]), max(bbS[2], bbgtS[2]), min(bbS[3], bbgtS[3])
                    iwS = biS[3] - biS[2] + 1
                    ihS = biS[1] - biS[0] + 1

                    if iwO > 0 and ihO > 0 and iwS > 0 and ihS > 0:
                        # compute overlap as area of intersection / area of union
                        uaO = (bbO[1] - bbO[0] + 1) * (bbO[3] - bbO[2] + 1) + \
                              (bbgtO[1] - bbgtO[0] + 1) * (bbgtO[3] - bbgtO[2] + 1) - iwO * ihO
                        ovO = iwO * ihO / uaO

                        uaS = (bbS[1] - bbS[0] + 1) * (bbS[3] - bbS[2] + 1) + \
                              (bbgtS[1] - bbgtS[0] + 1) * (bbgtS[3] - bbgtS[2] + 1) - iwS * ihS
                        ovS = iwS * ihS / uaS

                        ov = min(ovO, ovS)

                        # make sure that this object is detected according to its individual threshold
                        if ov >= gt_threshold and ov > ovmax:
                            ovmax = ov
                            kmax = k
                            count_correct += 1

                if kmax >= 0:
                    tp[j] = 1
                    gt_detected[kmax] = 1
                else:
                    fp[j] = 1

            tp_list[num].append(tp)
            # fp_list.append(fp)

    print(count_correct)

    for num in top_ks:
        count_correct = sum(sum(l) for l in tp_list[num])
        result = count_correct/num_pos_tuple
        print('R@{}, relationship'.format(num), result)


def top_recall_phrase(data, dict_indices_to_triplets, dict_indices_to_boxes, mode='test',
                      zeroshot=False, dataset='vrd', list_images=None):

    if dataset == 'vrd':
        with open('data/list_{}_images.pkl'.format(mode), 'rb') as f:
            list_images = pickle.load(f)

    gt_threshold = 0.5
    gt_data = get_gt(mode, zeroshot, dataset)

    num_pos_tuple = 0

    tp_list = {num: [] for num in top_ks}
    count_correct = 0

    for i, image_id in enumerate(list_images):

        gt_info = gt_data[image_id]
        info = data[image_id]

        gt_tupLabel, gt_subBox, gt_objBox = gt_info['list_tuples'], gt_info['list_sub_bboxes'], gt_info[
            'list_obj_bboxes']

        gt_subBox = [tuple(x) for x in gt_subBox]
        gt_objBox = [tuple(x) for x in gt_objBox]

        if len(gt_tupLabel) == 0:
            continue

        num_gt_tuple = len(gt_tupLabel)

        num_pos_tuple += num_gt_tuple

        labels, boxSub, boxObj, tuple_confs = info['list_tuples'][:500], info['list_sub_bboxes'][:500], \
                                              info['list_obj_bboxes'][:500], info['list_tuple_confs'][:500]

        if len(labels) == 0:
            continue

        # print(len(tuple_confs))

        if dataset == 'vrd':
            labels = [dict_indices_to_triplets[index] for index in labels]
            boxSub = [tuple(dict_indices_to_boxes[image_id][index]) for index in boxSub]
            boxObj = [tuple(dict_indices_to_boxes[image_id][index]) for index in boxObj]

        confs, ind_ = compute_score(tuple_confs, type=1)

        for num in top_ks:
            ind = ind_[:num]

            labels_ = get_sublist(labels, ind)
            boxSub_ = get_sublist(boxSub, ind)
            boxObj_ = get_sublist(boxObj, ind)

            gt_detected = np.zeros(num_gt_tuple)

            num_obj = len(labels_)
            tp = np.zeros(num_obj)
            fp = np.zeros(num_obj)

            for j in range(num_obj):
                bbO = boxObj_[j]
                bbS = boxSub_[j]
                bbO = min(bbO[0], bbS[0]), max(bbO[1], bbS[1]), \
                      min(bbO[2], bbS[2]), max(bbO[3], bbS[3])
                ovmax = -np.inf
                kmax = -1

                for k in range(num_gt_tuple):
                    if gt_detected[k] > 0:
                        continue
                    label = labels_[j]
                    gt_label = gt_tupLabel[k]

                    if any(e1 != e2 for e1, e2 in zip(label, gt_label)):
                        continue

                    bbgtO = gt_objBox[k]
                    bbgtS = gt_subBox[k]

                    bbgtO = min(bbgtO[0], bbgtS[0]), max(bbgtO[1], bbgtS[1]), \
                            min(bbgtO[2], bbgtS[2]), max(bbgtO[3], bbgtS[3])

                    biO = max(bbO[0], bbgtO[0]), min(bbO[1], bbgtO[1]), max(bbO[2], bbgtO[2]), min(bbO[3], bbgtO[3])
                    iwO = biO[3] - biO[2] + 1
                    ihO = biO[1] - biO[0] + 1

                    if iwO > 0 and ihO > 0:
                        # compute overlap as area of intersection / area of union
                        uaO = (bbO[1] - bbO[0] + 1) * (bbO[3] - bbO[2] + 1) + \
                              (bbgtO[1] - bbgtO[0] + 1) * (bbgtO[3] - bbgtO[2] + 1) - iwO * ihO
                        ov = iwO * ihO / uaO

                        # make sure that this object is detected according to its individual threshold
                        if ov >= gt_threshold and ov > ovmax:
                            ovmax = ov
                            kmax = k
                            count_correct += 1

                if kmax >= 0:
                    tp[j] = 1
                    gt_detected[kmax] = 1
                else:
                    fp[j] = 1

            tp_list[num].append(tp)
            # fp_list.append(fp)

    print(count_correct)

    for num in top_ks:
        count_correct = sum(sum(l) for l in tp_list[num])
        result = count_correct/num_pos_tuple
        print('R@{}, phrase'.format(num), result)


def compute_recall_phrase_and_relationship(data, dict_indices_to_triplets, dict_indices_to_boxes, mode='test',
                      zeroshot=False, dataset='vrd', list_images=None):
    with open('data/list_{}_images.pkl'.format(mode), 'rb') as f:
        list_images = pickle.load(f)

    tasks = ['phrase']

    gt_threshold = 0.5
    gt_data = get_gt(mode)

    num_pos_tuple = 0
    tp_list = {task: {num: [] for num in top_ks} for task in tasks}

    for i, image_id in enumerate(list_images):
        gt_info = gt_data[image_id]
        info = data[image_id]

        gt_tupLabel, gt_subBox, gt_objBox = gt_info['list_tuples'], gt_info['list_sub_bboxes'], gt_info[
            'list_obj_bboxes']

        num_gt_tuple = len(gt_tupLabel)

        num_pos_tuple += num_gt_tuple

        labels, boxSub, boxObj, tuple_confs = info['list_tuples'], info['list_sub_bboxes'], info['list_obj_bboxes'], \
                                              info['list_tuple_confs']

        if len(gt_tupLabel) == 0 or len(labels) == 0:
            continue

        labels_ = [dict_indices_to_triplets[index] for index in labels]
        boxSub_ = [dict_indices_to_boxes[image_id][index] for index in boxSub]
        boxObj_ = [dict_indices_to_boxes[image_id][index] for index in boxObj]

        confs, ind = torch.from_numpy(np.array(tuple_confs)).cuda().prod(1).sort(0, descending=True)

        for task in tasks:

            for num in top_ks:
                ind_ = ind[:num]

                labels__ = get_sublist(labels_, ind_)
                boxSub__ = get_sublist(boxSub_, ind_)
                boxObj__ = get_sublist(boxObj_, ind_)

                num_obj = len(labels__)

                gt_detected = np.zeros(num_gt_tuple)
                tp = np.zeros(num_obj)

                for j in range(num_obj):
                    bbO = boxObj__[j]
                    bbS = boxSub__[j]
                    ovmax = -np.inf
                    kmax = -1

                    for k in range(num_gt_tuple):

                        # # already detect, don't count anymore
                        if gt_detected[k] > 0:
                            continue

                        label = labels_[j]
                        gt_label = gt_tupLabel[k]

                        if any(e1 != e2 for e1, e2 in zip(label, gt_label)):
                            continue

                        bbgtO = gt_objBox[k]
                        bbgtS = gt_subBox[k]

                        if task == 'relationship':

                            biO = max(bbO[0], bbgtO[0]), min(bbO[1], bbgtO[1]), max(bbO[2], bbgtO[2]), min(bbO[3], bbgtO[3])
                            iwO = biO[3] - biO[2] + 1
                            ihO = biO[1] - biO[0] + 1

                            biS = max(bbS[0], bbgtS[0]), min(bbS[1], bbgtS[1]), max(bbS[2], bbgtS[2]), min(bbS[3], bbgtS[3])
                            iwS = biS[3] - biS[2] + 1
                            ihS = biS[1] - biS[0] + 1

                            if iwO > 0 and ihO > 0 and iwS > 0 and ihS > 0:
                                # compute overlap as area of intersection / area of union
                                uaO = (bbO[1] - bbO[0] + 1) * (bbO[3] - bbO[2] + 1) + \
                                      (bbgtO[1] - bbgtO[0] + 1) * (bbgtO[3] - bbgtO[2] + 1) - iwO * ihO
                                ovO = iwO * ihO / uaO

                                uaS = (bbS[1] - bbS[0] + 1) * (bbS[3] - bbS[2] + 1) + \
                                      (bbgtS[1] - bbgtS[0] + 1) * (bbgtS[3] - bbgtS[2] + 1) - iwS * ihS
                                ovS = iwS * ihS / uaS

                                ov = min(ovO, ovS)

                                # make sure that this object is detected according to its individual threshold
                                if ov >= gt_threshold and ov > ovmax:
                                    ovmax = ov
                                    kmax = k

                        else:

                            bbgtO = min(bbgtO[0], bbgtS[0]), max(bbgtO[1], bbgtS[1]), \
                                    min(bbgtO[2], bbgtS[2]), max(bbgtO[3], bbgtS[3])

                            biO = max(bbO[0], bbgtO[0]), min(bbO[1], bbgtO[1]), max(bbO[2], bbgtO[2]), min(bbO[3],
                                                                                                           bbgtO[3])
                            iwO = biO[3] - biO[2] + 1
                            ihO = biO[1] - biO[0] + 1

                            if iwO > 0 and ihO > 0:
                                # compute overlap as area of intersection / area of union
                                uaO = (bbO[1] - bbO[0] + 1) * (bbO[3] - bbO[2] + 1) + \
                                      (bbgtO[1] - bbgtO[0] + 1) * (bbgtO[3] - bbgtO[2] + 1) - iwO * ihO
                                ov = iwO * ihO / uaO

                                # make sure that this object is detected according to its individual threshold
                                if ov >= gt_threshold and ov > ovmax:
                                    ovmax = ov
                                    kmax = k

                    if kmax >= 0:
                        tp[j] = 1
                        gt_detected[kmax] = 1

                tp_list[task][num].append(tp)

    for task in tasks:
        for num in top_ks:
            count_correct = sum(sum(l) for l in tp_list[task][num])
            result = count_correct / num_pos_tuple
            print('R@{}, {}'.format(num, task), result)
