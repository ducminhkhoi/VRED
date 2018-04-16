import torch
import pickle
from collections import OrderedDict, defaultdict
import json
import operator
import os
import scipy.io as sio
import random
import numpy as np
import itertools
import h5py
import cv2

with open('data/objects.json', 'r') as f:
    list_objects = json.load(f)

with open('data/predicates.json', 'r') as f:
    list_predicates = json.load(f)

with open('data/annotations_train.json', 'r') as f:
    train_relationship_data = json.load(f)

with open('data/annotations_test.json', 'r') as f:
    test_relationship_data = json.load(f)


def convert_box_to_vrd(box):
    return tuple(box[i] for i in [1, 3, 0, 2])


def create_pbtxt_file():
    with open('data/vrd_label_map_detector.pbtxt', 'w') as f:
        for i, object_ in enumerate(list_objects):
            f.write("item {\n  id: %d\n  name: '%s'\n}\n" % (i+1, object_))


id_to_object = {i: o for i, o in enumerate(list_objects)}
object_to_id = {o: i for i, o in enumerate(list_objects)}
id_to_predicate = {i: p for i, p in enumerate(list_predicates)}
predicate_to_id = {p: i for i, p in enumerate(list_predicates)}


def get_triplet_embedding(list_objects, list_predicates, list_triplets_indices):
    infersent = torch.load('InferSent/encoder/infersent.allnli.pickle')

    infersent.set_glove_path('InferSent/dataset/GloVe/glove.840B.300d.txt')

    infersent.build_vocab([' '.join(list_objects), ' '.join(list_predicates)], tokenize=True)
    # infersent.build_vocab_k_words(K=100000)

    # triplets = [(s, p, o) for s in list_objects for p in list_predicates for o in list_objects]
    # triplet_indices = [(s, p, o) for s in range(len(list_objects)) for p in range(len(list_predicates))
    #                    for o in range(len(list_objects))]

    triplets = [(list_objects[s], list_predicates[p], list_objects[o]) for s, p, o in list_triplets_indices]

    triplets_phrase = [' '.join(triplet) for triplet in triplets]
    triplets_embedding = infersent.encode(triplets_phrase)

    triplet_encoding_dict = OrderedDict([(triplet, triplets_embedding[i])
                                         for i, triplet in enumerate(list_triplets_indices)])

    print('Done Getting, start to dump')

    return triplet_encoding_dict


def print_statistics():
    object_dict = defaultdict(int)
    relation_dict = defaultdict(int)
    relation_by_image_dict = defaultdict(list)
    human_relation_dict = defaultdict(int)
    human_shirt_relation_dict = defaultdict(int)
    total_relation = 0

    triplets = set()
    triplets_id = set()

    for i, (image_id, relationships) in enumerate(test_relationship_data.items()):
        print(i, len(test_relationship_data), image_id)

        for j, x in enumerate(relationships):
            total_relation += 1
            object_id = x['object']['category']
            subject_id = x['subject']['category']
            predicate_id = x['predicate']

            subject = id_to_object[subject_id]
            object_ = id_to_object[object_id]
            predicate = id_to_predicate[predicate_id]

            triplets.add((subject, predicate, object_))
            triplets_id.add((subject_id, predicate_id, object_id))

            # object_dict[object_] += 1
            # object_dict[subject] += 1
            # relation_dict[(subject_id, object_id, predicate_id)] += 1
            # # relation_dict[subject+'|'+predicate+'|'+object_] += 1
            # relation_by_image_dict[image_id + '|' + subject + '|' + object_].append(id_to_predicate[x['predicate']])
            # if subject in ['man', 'woman', 'person', 'boy', 'girl', 'people', 'child']:
            #     human_relation_dict[object_] += 1
            #     if object_ == 'shirt':
            #         human_shirt_relation_dict[id_to_predicate[x['predicate']]] += 1

    triplets_phrase = [' '.join(triplet) for triplet in triplets]
    triplets_embedding = infersent.encode(triplets_phrase)

    triplet_encoding_dict = {triplet: triplets_embedding[i] for i, triplet in enumerate(triplets_id)}

    with open('data/test_triplet_embedding.pkl', 'wb') as f:
        pickle.dump(triplet_encoding_dict, f, pickle.HIGHEST_PROTOCOL)

    # printout some statistics
    sorted_objects = reversed(sorted(object_dict.items(), key=operator.itemgetter(1)))
    sorted_relations = reversed(sorted(relation_dict.items(), key=operator.itemgetter(1)))
    sorted_human_relations = reversed(sorted(human_relation_dict.items(), key=operator.itemgetter(1)))
    sorted_human_shirt_relations = reversed(sorted(human_shirt_relation_dict.items(), key=operator.itemgetter(1)))
    sorted_relations_by_image_id = reversed(sorted(relation_by_image_dict.items(), key=operator.itemgetter(0)))

    pass

    with open('data/relation_stats.pkl', 'wb') as f:
        pickle.dump((relation_dict, total_relation), f, pickle.HIGHEST_PROTOCOL)

        # # most frequent objects:
        # with open('data/vrd_most_frequent_objects.txt', 'w') as f:
        #     for key, value in sorted_objects:
        #         f.write('{}: {}\n'.format(key, value))
        #
        # # most frequent relations:
        # with open('data/vrd_most_frequent_relations.txt', 'w') as f:
        #     for key, value in sorted_relations:
        #         f.write('{}: {}\n'.format(key, value/total_relation))
        #
        # # most human frequent relations:
        # with open('data/vrd_most_frequent_human_relations.txt', 'w') as f:
        #     for key, value in sorted_human_relations:
        #         f.write('{}: {}\n'.format(key, value))
        #
        # # most human frequent relations:
        # with open('data/vrd_most_frequent_human_shirt_relations.txt', 'w') as f:
        #     for key, value in sorted_human_shirt_relations:
        #         f.write('{}: {}\n'.format(key, value))
        #
        # # most relations by image id:
        # with open('data/vrd_most_frequent_relations_by_image_id.txt', 'w') as f:
        #     for key, value in sorted_relations_by_image_id:
        #         f.write('{}: {}\n'.format(key, value))


def get_rcnn_results():

    mat_file = 'Visual-Relationship-Detection/data/objectDetRCNN.mat'
    image_mat_file = 'Visual-Relationship-Detection/data/imagePath.mat'

    mat = sio.loadmat(mat_file)
    image_mat = sio.loadmat(image_mat_file)
    image_paths = image_mat['imagePath'][0]
    confs = mat['detection_confs'][0]
    bboxes = mat['detection_bboxes'][0]
    labels = mat['detection_labels'][0]

    outputs = {}
    for image_id, bbox, conf, label in zip(image_paths, bboxes, confs, labels):
        image_id = str(image_id[0])
        outputs[image_id] = (bbox, conf, label)

    return outputs


def get_vg_data():
    data_path = 'cvpr17_vtranse/data/vg1_2_meta.h5'

    image_file = '/scratch/datasets/Data1/visual_genome/VG_100K_images/{}.jpg'

    file = h5py.File(data_path, 'r')

    list_train_files = list(file['gt']['train'].keys())
    list_test_files = list(file['gt']['test'].keys())

    list_predicates = list(file['meta']['pre']['name2idx'])
    list_objects = list(file['meta']['cls']['name2idx'])

    boxes_to_indices = {}
    indices_to_boxes = {}
    set_triplets_indices = set()

    def get_data(mode, list_images):
        nonlocal boxes_to_indices, indices_to_boxes, set_triplets_indices

        gt_data = {image_id: {'list_tuples': [], 'list_sub_bboxes': [], 'list_obj_bboxes': []}
                for image_id in list_images}

        pos_examples = 0
        neg_examples = 0
        neg_indices = []
        pos_indices = []

        data = []
        for k, image_id in enumerate(list_images):
            print(k, len(list_images))
            # im = cv2.imread(image_file.format(image_id))

            triplets = file['gt'][mode][image_id]['rlp_labels']
            subject_boxes = file['gt'][mode][image_id]['sub_boxes']
            object_boxes = file['gt'][mode][image_id]['obj_boxes']

            num_relationships = triplets.shape[0]
            # colors = np.random.randint(0, 255, (num_relationships, 3))

            boxes_to_indices[image_id] = {}
            indices_to_boxes[image_id] = {}

            set_objects = set()
            list_triplets = list()
            list_pairs = list()

            for i in range(num_relationships):

                subject_id = int(triplets[i][0])
                predicate_id = int(triplets[i][1])
                object_id = int(triplets[i][2])

                # set_triplets_indices.add((subject_id, predicate_id, object_id))

                list_triplets.append((subject_id, predicate_id, object_id))

                subject_bbox = subject_boxes[i].tolist()
                object_bbox = object_boxes[i].tolist()

                # color = tuple(int(x) for x in colors[i])
                #
                # cv2.rectangle(im, (subject_box[0], subject_box[1]), (subject_box[2], subject_box[3]), color, 2)
                # cv2.putText(im, '{:s} {:s}'.format(list_objects[subject_id], list_predicates[predicate_id]),
                #             (subject_box[0], int(subject_box[1] + 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                #
                # cv2.rectangle(im, (object_box[0], object_box[1]), (object_box[2], object_box[3]), color, 2)
                # cv2.putText(im, '{:s}'.format(list_objects[object_id]),
                #             (object_box[0], int(object_box[1] + 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

                subject_bbox = convert_box_to_vrd(subject_bbox)
                object_bbox = convert_box_to_vrd(object_bbox)

                gt_data[image_id]['list_sub_bboxes'].append(subject_bbox)
                gt_data[image_id]['list_obj_bboxes'].append(object_bbox)
                gt_data[image_id]['list_tuples'].append((subject_id, predicate_id, object_id))

                # if subject_bbox not in boxes_to_indices:
                #     len_boxes_to_indices = len(boxes_to_indices[image_id])
                #     indices_to_boxes[image_id][len_boxes_to_indices] = subject_bbox
                #     boxes_to_indices[image_id][subject_bbox] = len_boxes_to_indices
                #
                # if object_bbox not in boxes_to_indices:
                #     len_boxes_to_indices = len(boxes_to_indices[image_id])
                #     indices_to_boxes[image_id][len_boxes_to_indices] = object_bbox
                #     boxes_to_indices[image_id][object_bbox] = len_boxes_to_indices

                image_bbox = (min(object_bbox[0], subject_bbox[0]), max(object_bbox[1], subject_bbox[1]),
                              min(object_bbox[2], subject_bbox[2]), max(object_bbox[3], subject_bbox[3]))

                data.append((image_id, image_bbox, object_bbox,
                             subject_bbox, object_id, subject_id, predicate_id))

                # object_tuple = tuple(object_bbox) + (object_id, )
                # subject_tuple = tuple(subject_bbox) + (subject_id,)
                #
                # set_objects.add(object_tuple)
                # set_objects.add(subject_tuple)
                #
                # list_pairs.append((subject_tuple, object_tuple))

                # if (subject_tuple, object_tuple) not in real_pairs:
                #     real_pairs.add((subject_tuple, object_tuple))
                #
                # if (object_tuple, subject_tuple) not in real_pairs:
                #     real_pairs.add((object_tuple, subject_tuple))

            # set_objects = list(set_objects)

            # data[image_id] = (set_objects, list_triplets, list_pairs)
            # num_objects = len(set_objects)

            # pairs_indices = [(x, y) for x in range(num_objects) for y in range(num_objects)]
            #
            # for pair_index in pairs_indices:
            #     subject = set_objects[pair_index[0]]
            #     object_ = set_objects[pair_index[1]]
            #     pair = (subject, object_)
            #
            #     subject_bbox, subject_id = subject[:4], subject[-1]
            #     object_bbox, object_id = object_[:4], object_[-1]
            #
            #     image_bbox = (min(object_bbox[0], subject_bbox[0]), max(object_bbox[1], subject_bbox[1]),
            #                   min(object_bbox[2], subject_bbox[2]), max(object_bbox[3], subject_bbox[3]))
            #
            #     if pair in real_pairs:
            #         pos_examples += 1
            #         pos_indices.append(len(data))
            #         data.append((image_id, image_bbox, object_bbox, subject_bbox, object_id, subject_id, 1))
            #     else:
            #         neg_examples += 1
            #         neg_indices.append(len(data))
            #         data.append((image_id, image_bbox, object_bbox, subject_bbox, object_id, subject_id, 0))

            # cv2.imshow('visualize', im)
            # cv2.waitKey()

        # choose_neg = np.random.choice(neg_indices, pos_examples).tolist()
        # choose_indices = pos_indices + choose_neg
        # random.shuffle(choose_indices)
        # data = [data[i] for i in choose_indices]

        with open('data/vg_{}_gt_data.pkl'.format(mode), 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

        with open('data/vg_{}_gt.pkl'.format(mode), 'wb') as f:
            pickle.dump(gt_data, f, pickle.HIGHEST_PROTOCOL)

    # get_data('train', list_train_files)
    get_data('test', list_test_files)

    # triplet_embedding = get_triplet_embedding(list_objects, list_predicates, list(set_triplets_indices))
    #
    # with open('data/vg_meta_data.pkl', 'wb') as f:
    #     pickle.dump({
    #         'list_predicates': list_predicates,
    #         'list_objects': list_objects,
    #         'list_train_files': list_train_files,
    #         'list_test_files': list_test_files,
    #         'gt_boxes_to_indices': boxes_to_indices,
    #         'gt_indices_to_boxes': indices_to_boxes,
    #         'triplet_embedding': triplet_embedding
    #     }, f, pickle.HIGHEST_PROTOCOL)


# get_vg_data()


def get_unrel_dataset():
    annotations = sio.loadmat('data/unrel-dataset/annotations.mat')['annotations']
    triplets = sio.loadmat('data/unrel-dataset/annotated_triplets.mat')['triplets']

    set_triplets = set()
    num_triplets = triplets.shape[0]
    for i in range(num_triplets):
        triplet = str(triplets[i][0][0])
        sub, pred, obj = triplet.split('-')
        sub_id = object_to_id[sub]
        obj_id = object_to_id[obj]
        pred_id = predicate_to_id[pred]
        set_triplets.add((sub_id, pred_id, obj_id))

    dict_images_triplets = {triplet: [] for triplet in set_triplets}
    data = []
    num_annotations = annotations.shape[0]
    list_images = []
    for i in range(num_annotations):
        annotation = annotations[i][0][0][0]
        image_id = str(annotation[1][0][0])
        list_images.append(image_id)
        relationship = annotation[2][0][0][0][0]
        subject_id = object_to_id[str(relationship[0][0])]
        object_id = object_to_id[str(relationship[1][0])]
        predicate_id = predicate_to_id[str(relationship[4][0][0][0])]

        subject_bbox = convert_box_to_vrd(relationship[2][0].tolist())
        object_bbox = convert_box_to_vrd(relationship[3][0].tolist())

        image_bbox = (min(object_bbox[0], subject_bbox[0]), max(object_bbox[1], subject_bbox[1]),
                      min(object_bbox[2], subject_bbox[2]), max(object_bbox[3], subject_bbox[3]))

        data.append((image_id, image_bbox, object_bbox, subject_bbox, object_id, subject_id, predicate_id))

    with open('data/vrd_test_data.pkl', 'rb') as f:
        data.extend(pickle.load(f))

    dict_correct_examples = {triplet: [] for triplet in set_triplets}

    for i, example in enumerate(data):
        _, _, _, _, object_id, subject_id, _ = example

        for j in range(len(list_predicates)):
            triplet = (subject_id, j, object_id)
            if triplet in dict_images_triplets:
                dict_images_triplets[triplet].append(example)

    for triplet in set_triplets:
        print(len(dict_images_triplets[triplet]))
        for i, example in enumerate(dict_images_triplets[triplet]):
            _, _, _, _, object_id, subject_id, predicate_id = example

            if triplet == (subject_id, predicate_id, object_id):
                dict_correct_examples[triplet].append(i)

    with open('data/unrel_test_data.pkl', 'wb') as f:
        pickle.dump(dict_images_triplets, f, pickle.HIGHEST_PROTOCOL)

    with open('data/unrel_test_meta_data.pkl', 'wb') as f:
        pickle.dump({
            'triplets': set_triplets,
            'correct_examples': dict_correct_examples
        }, f, pickle.HIGHEST_PROTOCOL)


# get_unrel_dataset()


def get_data():
    list_outputs = get_rcnn_results()
    train_data_file = 'data/vrd_train_gt_triplets.pkl'
    test_data_file = 'data/vrd_test_gt_boxes.pkl'
    order = [1, 3, 0, 2]
    # order = [2, 0, 3, 1]

    opposite_class = {
        0: [15, 23, 29],
        9: [15, 23, 29],
        15: [0, 9],
        23: [0, 9],
        29: [0, 9],
        32: [33],
        33: [32]
    }

    same_class = {
        0: [9],
        9: [0],
        15: [23, 29],
        23: [15, 29],
        29: [15, 23],
    }

    list_images = []


    bidirectional_class = [24, 28, 30, 47]

    def get_data_set(mode, relationship_data, data_file):
        set_triplet = set()

        set_boxes = defaultdict(set)
        data = {}
        count_prob = {(s, p, o): 0 for s in range(len(list_objects)) for p in range(len(list_predicates))
                for o in range(len(list_objects))}

        pos_examples = 0
        neg_examples = 0
        neg_indices = []
        pos_indices = []

        for i, (image_id, relationships) in enumerate(relationship_data.items()):
            print(i, len(relationship_data), image_id)

            # list_images.append(image_id)

            # if image_id not in list_test_image:
            #     list_test_image.append(image_id)
            #
            # bboxes, conf, label = list_outputs[image_id]
            # num_examples = bboxes.shape[0]
            #
            # pairs = [(x, y) for x in range(num_examples) for y in range(num_examples) if x != y]
            #
            # for pair in pairs:
            #     subject = pair[0]
            #     object_ = pair[1]
            #
            #     subject_bbox = bboxes[subject, :]
            #     object_bbox = bboxes[object_, :]
            #
            #     subject_id = int(label[subject])-1
            #     object_id = int(label[object_])-1
            #
            #     subject_conf = float(conf[subject])
            #     object_conf = float(conf[object_])
            #
            #     subject_bbox = [subject_bbox[k] for k in order]
            #     object_bbox = [object_bbox[k] for k in order]
            #
            #     image_bbox = [min(object_bbox[0], subject_bbox[0]), max(object_bbox[1], subject_bbox[1]),
            #                   min(object_bbox[2], subject_bbox[2]), max(object_bbox[3], subject_bbox[3])]
            #
            #     data.append((image_id, image_bbox, object_bbox, subject_bbox,
            #                  object_id, subject_id, object_conf, subject_conf))

            #
            # if len(relationships) == 0:
            #     print(i)
            #
            # set_objects = set()
            # real_pairs = set()
            #
            for j, x in enumerate(relationships):
                object_bbox = x['object']['bbox']
                subject_bbox = x['subject']['bbox']

                set_boxes[image_id].add(tuple(object_bbox))
                set_boxes[image_id].add(tuple(subject_bbox))

                object_id = x['object']['category']
                subject_id = x['subject']['category']
                predicate_id = x['predicate']

                set_triplet.add((subject_id, predicate_id, object_id))
            #
            #     count_prob[(subject_id, predicate_id, object_id)] += 1

                # object_tuple = tuple(object_bbox) + (object_id, )
                # set_objects.add(object_tuple)
                #
                # subject_tuple = tuple(subject_bbox) + (subject_id, )
                # set_objects.add(subject_tuple)
            #
            #     if (subject_tuple, object_tuple) not in real_pairs:
            #         real_pairs.add((subject_tuple, object_tuple))
            #
            #     if (object_tuple, subject_tuple) not in real_pairs:
            #         real_pairs.add((object_tuple, subject_tuple))
            #
            # set_objects = list(set_objects)
            # num_objects = len(set_objects)
            #
            # pairs_indices = [(x, y) for x in range(num_objects) for y in range(num_objects)]
            #
            # for pair_index in pairs_indices:
            #     subject = set_objects[pair_index[0]]
            #     object_ = set_objects[pair_index[1]]
            #     pair = (subject, object_)
            #
            #     subject_bbox, subject_id = subject[:4], subject[-1]
            #     object_bbox, object_id = object_[:4], object_[-1]
            #
            #     image_bbox = [min(object_bbox[0], subject_bbox[0]), max(object_bbox[1], subject_bbox[1]),
            #                   min(object_bbox[2], subject_bbox[2]), max(object_bbox[3], subject_bbox[3])]
            #
            #     if pair in real_pairs:
            #         pos_examples += 1
            #         pos_indices.append(len(data))
            #         data.append((image_id, image_bbox, object_bbox, subject_bbox, object_id, subject_id, 1))
            #     else:
            #         neg_examples += 1
            #         neg_indices.append(len(data))
            #         data.append((image_id, image_bbox, object_bbox, subject_bbox, object_id, subject_id, 0))

                # data.append((image_id, image_bbox, object_bbox, subject_bbox, object_id, subject_id, predicate_id))

        # count_object_new = reversed(sorted(count_object.items(), key=operator.itemgetter(1)))
        #
        # print('total', sum(count_object.values()))
        # for key, value in count_object_new:
        #     print(id_to_object[key], value)
        #

        ## random choose neg example:
        # choose_neg = np.random.choice(neg_examples, pos_examples).tolist()
        # choose_indices = pos_indices + choose_neg
        # random.shuffle(choose_indices)
        # data = [data[i] for i in choose_indices]
        #
        # total_examples = sum(count_prob.values())
        #
        # for key, value in count_prob.items():
        #     data['triplet:'+str(key)] = (value + 1) / (total_examples + len(count_prob))
        #
        # for s in range(len(list_objects)):
        #     data['subject:'+str(s)] = sum(data['triplet:'+str((s, p, o))] for p in range(len(list_predicates)) for o in range(len(list_objects)))
        # for o in range(len(list_objects)):
        #     data['object:' + str(o)] = sum(
        #         data['triplet:' + str((s, p, o))] for s in range(len(list_objects)) for p in range(len(list_predicates)))
        # for p in range(len(list_predicates)):
        #     data['predicate:' + str(p)] = sum(
        #         data['triplet:' + str((s, p, o))] for s in range(len(list_objects)) for o in range(len(list_objects)))
        #

        with open(data_file, 'wb') as f:
            pickle.dump(set_triplet, f, pickle.HIGHEST_PROTOCOL)

        # with open('data/list_train_images.pkl', 'wb') as f:
        #     pickle.dump(list(list_images), f, pickle.HIGHEST_PROTOCOL)

    get_data_set('train', train_relationship_data, train_data_file)
    # get_data_set('test', test_relationship_data, test_data_file)

    print('complete!')


# get_data()


def get_gt(mode='test', zeroshot=False, dataset='vrd'):

    if dataset == 'vg':
        gt_file = 'data/vg_{}_gt.pkl'.format(mode)
    else:
        if zeroshot:
            gt_file = 'data/gt_{}.pkl'.format(mode)
        else:
            gt_file = 'result/gt_{}.pkl'.format(mode)

    if os.path.isfile(gt_file):
        with open(gt_file, 'rb') as f:
            gt_data = pickle.load(f)
    else:
        with open('data/annotations_{}.json'.format(mode), 'r') as f:
            relationship_data = json.load(f)

        with open('data/vrd_train_gt_triplets.pkl', 'rb') as f:
            set_train_triplets = pickle.load(f)

        miss_1 = set()
        miss_2 = set()

        with open('data/list_{}_images.pkl'.format(mode), 'rb') as f:
            list_test_image = pickle.load(f)

        gt_data = {image_id: {'list_tuples': [], 'list_sub_bboxes': [], 'list_obj_bboxes': []}
                for image_id in list_test_image}

        for i, (image_id, relationships) in enumerate(relationship_data.items()):
            # print(i, len(relationship_data), image_id)

            if len(relationships) == 0:
                print(image_id)
                miss_2.add(image_id)

            for j, x in enumerate(relationships):
                object_ = x['object']['category']
                subject = x['subject']['category']
                predicate = x['predicate']

                object_bbox = x['object']['bbox']
                subject_bbox = x['subject']['bbox']

                triplet = (subject, predicate, object_)

                if zeroshot:
                    if triplet not in set_train_triplets:
                        gt_data[image_id]['list_tuples'].append(triplet)
                        gt_data[image_id]['list_sub_bboxes'].append(subject_bbox)
                        gt_data[image_id]['list_obj_bboxes'].append(object_bbox)
                else:
                    gt_data[image_id]['list_tuples'].append(triplet)
                    gt_data[image_id]['list_sub_bboxes'].append(subject_bbox)
                    gt_data[image_id]['list_obj_bboxes'].append(object_bbox)

        # print(miss_1.difference(miss_2))

        with open(gt_file, 'wb') as f:
            pickle.dump(gt_data, f, pickle.HIGHEST_PROTOCOL)

    return gt_data


# get_gt()


color_list = [(255, 0, 0),  # 'ff0000'
              (255, 153, 0),  # 'ff9900'
              (255, 255, 0),  # 'ffff00'
              (0, 255, 0),  # '00ff00'
              (0, 255, 255),  # '00ffff'
              (0, 0, 255),  # '0000ff'
              (153, 0, 255),  # '9900ff'
              (255, 0, 255),  # 'ff00ff'
            ]


def qualitative_result(list_tuples):
    import cv2
    gt_tuple_labels, gt_sub_bboxes, gt_obj_bboxes, gt_set_objects, list_images = get_gt()

    dict_correct = {0: [], 1: [], 2: [], 3: [], 4: []}
    image_folder = '/scratch/datasets/Data1/sg_dataset/'

    image_path = image_folder + 'sg_test_images/{}'

    for i, (gt_tuple_label, gt_sub_bbox, gt_obj_bboxes, gt_set_object, list_tuple) in \
            enumerate(zip(gt_tuple_labels, gt_sub_bboxes, gt_obj_bboxes, gt_set_objects, list_tuples)):

        if len(gt_tuple_label) == 4 and len(gt_set_object) < 9:
            num_correct = sum(e1 == e2 for e1, e2 in zip(gt_tuple_label, list_tuple))
            dict_correct[num_correct].append(i)

            path = 'result/qualitative/{}/'.format(i)

            if not os.path.isdir(path):
                os.mkdir(path)

            with open(path+'result.txt', 'w') as f:
                for t1, t2 in zip(gt_tuple_label, list_tuple):
                    if t1 == t2:
                        f.write('{}-{}-{}\n'.format(id_to_object[t1[0]], id_to_predicate[t1[2]], id_to_object[t1[1]]))
                    else:
                        f.write('{}-{}|{}-{}\n'.format(id_to_object[t1[0]], id_to_predicate[t1[2]], id_to_predicate[t2[2]],
                                                       id_to_object[t2[1]]))

            img = cv2.imread(image_path.format(list_images[i]), cv2.IMREAD_COLOR)

            for k, object_ in enumerate(gt_set_object):
                object_label, (y1, y2, x1, x2) = object_

                cv2.rectangle(img, (x1, y1), (x2, y2), color_list[k], 5)

            cv2.imwrite(path+'result.jpg', img)

            pass

    for key, value in dict_correct.items():
        print(key, value)
