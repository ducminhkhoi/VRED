import numpy as np
import os
import six.moves.urllib as urllib
import sys
import json
import tarfile
import tensorflow as tf
import zipfile
from time import sleep
import pickle
from operator import itemgetter

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

from object_detection.utils import label_map_util

from object_detection.utils import visualization_utils as vis_util

folder_ckpt = 'train_one_class'

if folder_ckpt == 'train_two_class':
    path_to_labels = 'data/vrd_label_map_two_class.pbtxt'
    num_classes = 2
    save_file_name = 'data/vrd_two_class_bbox.pkl'
elif folder_ckpt == 'train_one_class':
    path_to_labels = 'data/vrd_label_map_one_class.pbtxt'
    num_classes = 1
    save_file_name = 'data/vrd_one_class_bbox.pkl'
elif folder_ckpt == 'train_region':
    path_to_labels = 'data/vrd_label_map_region.pbtxt'
    num_classes = 1
    save_file_name = 'data/vrd_region_bbox.pkl'
elif folder_ckpt == 'train_detector':
    path_to_labels = 'data/vrd_label_map_detector.pbtxt'
    num_classes = 100
    save_file_name = 'data/vrd_detector_bbox.pkl'

path_to_ckpt = 'model_checkpoints/faster_rcnn/{}/post-trained/frozen_inference_graph.pb'.format(folder_ckpt)

IMAGE_SIZE = (12, 8)
image_folder = '/scratch/datasets/Data1/sg_dataset/'
with open('data/annotations_test.json', 'r') as f:
    test_relationship_data = json.load(f)

with open('data/objects.json', 'r') as f:
    list_objects = json.load(f)

gt_index = {i: {'id': i, 'name': object_name} for i, object_name in enumerate(list_objects)}
vrd_to_other = [1, 3, 0, 2]
other_to_vrd = [2, 0, 3, 1]
is_visualize = False


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def calculate_iou(boxA, boxB, call_from_region=False):
    # determine the (x, y)-coordinates of the intersection rectangle

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = (xB - xA + 1) * (yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area

    try:
        iou = interArea / float(boxAArea + boxBArea - interArea)
    except ZeroDivisionError:
        print(call_from_region)
        print(boxAArea, boxBArea, interArea, boxA, boxB)
        iou = -1

    # return the intersection over union value
    return iou


def get_rcnn_results():
    import scipy.io as sio
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


def get_boxes_scores(dict_box, object_threshold):

    boxes_, scores_, class_ = dict_box
    boxes_ = boxes_.reshape((-1, 4))
    scores_ = scores_.reshape((-1, 1))
    class_ = class_.reshape((-1, 1))

    data = np.concatenate((scores_, class_, boxes_, ), 1)
    data = data[data[:, 0] > object_threshold, :]

    scores_, class_, boxes_ = data[:, 0], data[:, 1], data[:, 2:]

    return boxes_, scores_, class_


def convert_box_to_vrd(box, img_id):
    if img_id == '194654941_052c0bd67f_o.jpg':
        return [box[i] for i in [0, 2, 1, 3]]
    else:
        return [box[i] for i in vrd_to_other]


def make_pairs(boxes, region_boxes, class_, image_id):
    pair_indices = [(x, y) for x in range(boxes.shape[0]) for y in range(boxes.shape[0]) if x != y]
    pair_classes = [(int(class_[x]), int(class_[y])) for x, y in pair_indices]
    pairs = [(boxes[x, :].astype(int).tolist(), boxes[y, :].astype(int).tolist()) for x, y in pair_indices]

    union_boxes = [[min(p1[0], p2[0]), min(p1[1], p2[1]), max(p1[2], p2[2]), max(p1[3], p2[3])]
                   for p1, p2 in pairs]

    chosen_pairs = list()

    for i, union_box in enumerate(union_boxes):
        for region_box in region_boxes:
            iou = calculate_iou(region_box, union_box, True)

            if iou > 0.75:
                chosen_pairs.append(i)
                break

    pairs_to_save = [(image_id, union_boxes[i], pairs[i][0], pairs[i][1])
                      for i in chosen_pairs]

    pairs = [pairs[i] for i in chosen_pairs]
    pair_classes = [pair_classes[i] for i in chosen_pairs]

    # pairs = [pairs[i] for i in range(len(pairs))]

    return pairs, pair_classes, pairs_to_save


def calculate_recall_relationship(pairs, triplets, gt_pairs, gt_triplets, threshold=0.5):
    num_predicted_pairs = len(pairs)
    num_gt_pairs = len(gt_pairs)
    tp_pair = np.zeros(num_predicted_pairs)
    gt_pair_detected = np.zeros(num_gt_pairs)

    for j, ((s_box, o_box), triplet) in enumerate(zip(pairs, triplets)):
        iou_max = -np.inf
        k_max = -1

        for k, ((gt_s_box, gt_o_box), gt_triplet) in enumerate(zip(gt_pairs, gt_triplets)):
            if gt_pair_detected[k] > 0:
                continue

            # if not (s_id == gt_s_id and o_id == gt_o_id):
            #     continue

            iou_0 = calculate_iou(gt_s_box, s_box)
            iou_1 = calculate_iou(gt_o_box, o_box)
            iou = min(iou_0, iou_1)

            if iou > threshold and iou > iou_max:
                iou_max = iou
                k_max = k

        if k_max > -1:
            gt_pair_detected[k_max] = 1
            tp_pair[j] = 1

    num_correct = sum(tp_pair)
    print(num_predicted_pairs, num_correct, num_gt_pairs)
    return num_correct, num_gt_pairs


def calculate_recall_phrase(pairs, triplets, gt_pairs, gt_triplets, threshold=0.5):
    num_predicted_pairs = len(pairs)
    num_gt_pairs = len(gt_pairs)
    tp_pair = np.zeros(num_predicted_pairs)
    gt_pair_detected = np.zeros(num_gt_pairs)

    for j, ((s_box, o_box), triplet) in enumerate(zip(pairs, triplets)):
        iou_max = -np.inf
        k_max = -1

        union_box = min(s_box[0], o_box[0]), min(s_box[1], o_box[1]), max(s_box[2], o_box[2]), max(s_box[3], o_box[3])

        for k, ((gt_s_box, gt_o_box), (gt_triplet)) in enumerate(zip(gt_pairs, gt_triplets)):
            if gt_pair_detected[k] > 0:
                continue

            gt_union_box = min(gt_s_box[0], gt_o_box[0]), min(gt_s_box[1], gt_o_box[1]), \
                           max(gt_s_box[2], gt_o_box[2]), max(gt_s_box[3], gt_o_box[3])

            # if not (s_id == gt_s_id and o_id == gt_o_id):
            #     continue

            iou = calculate_iou(gt_union_box, union_box)

            if iou > threshold and iou > iou_max:
                iou_max = iou
                k_max = k

        if k_max > -1:
            gt_pair_detected[k_max] = 1
            tp_pair[j] = 1

    num_correct = sum(tp_pair)
    print(num_predicted_pairs, num_correct, num_gt_pairs)
    return num_correct


def filter_box():
    with open('data/vrd_pre_faster_rcnn_bbox.pkl', 'rb') as f:
        dict_pre_bbox = pickle.load(f)

    with open('data/vrd_two_class_bbox.pkl', 'rb') as f:
        dict_two_class_bbox = pickle.load(f)

    with open('data/vrd_one_class_bbox_old.pkl', 'rb') as f:
        dict_one_class_bbox = pickle.load(f)

    with open('data/vrd_region_bbox.pkl', 'rb') as f:
        dict_region_bbox = pickle.load(f)

    with open('data/vrd_detector_bbox.pkl', 'rb') as f:
        dict_detection_bbox = pickle.load(f)

    dict_rcnn_bbox = get_rcnn_results()
    count_gt_boxes = 0
    count_gt_pairs = 0
    count_predicted_bboxes = 0
    count_correct_relationship = 0
    count_correct_phrase = 0
    threshold = 0.8
    object_threshold = 0.01  # 0.01
    region_threshold = 0.01

    list_count_predicted_objects = []
    list_count_predicted_pairs = []

    data = []
    chosen_indices = {}

    data_file = 'data/vrd_test_data_faster_rcnn_one_class.pkl'
    chosen_indices_file = 'data/vrd_test_data_faster_rcnn_one_class_chosen_indices.pkl'

    # x_min, y_min, x_max, y_max
    for i, (image_id, relationships) in enumerate(test_relationship_data.items()):
        print(i, image_id)
        set_objects = set()

        gt_pairs = []
        gt_triplets = []
        for _, x in enumerate(relationships):
            object_bbox = x['object']['bbox']
            subject_bbox = x['subject']['bbox']

            object_id = x['object']['category']
            subject_id = x['subject']['category']
            predicate_id = x['predicate']

            object_tuple = tuple(object_bbox[l] for l in vrd_to_other)
            subject_tuple = tuple(subject_bbox[l] for l in vrd_to_other)

            set_objects.add(object_tuple)
            set_objects.add(subject_tuple)

            gt_pairs.append((subject_tuple, object_tuple))
            gt_triplets.append((subject_id, predicate_id, object_id))

        # gt_boxes = np.array(list(set_objects))
        # num_gt_objects = len(set_objects)

        # if num_gt_objects > 0:
        boxes, scores, class_ = get_boxes_scores(dict_one_class_bbox[image_id], object_threshold)
        region_boxes, region_scores, _ = get_boxes_scores(dict_region_bbox[image_id], region_threshold)
        pairs, triplets, pairs_to_save = make_pairs(boxes, region_boxes, class_, image_id)

        # data.extend(pairs_to_save)
        # chosen_indices[image_id] = chosen_index

        # num_predicted_objects = boxes.shape[0]

        # list_count_predicted_objects.append(num_predicted_objects)
        # list_count_predicted_pairs.append(len(pairs))

        # count_gt_boxes += num_gt_objects

        # tp = np.zeros(num_predicted_objects)
        # gt_detected = np.zeros(num_gt_objects)

        # for j, box in enumerate(boxes):
        #
        #     iou_max = -np.inf
        #     k_max = -1
        #
        #     for k, gt_box in enumerate(gt_boxes):
        #         if gt_detected[k] > 0:
        #             continue
        #
        #         gt_box, gt_class = gt_box[:4], gt_box[-1]
        #
        #         # if gt_class != class_[j]:
        #         #     continue
        #
        #         iou = calculate_iou(gt_box, box)
        #
        #         if iou > threshold and iou > iou_max:
        #             iou_max = iou
        #             k_max = k
        #
        #     if k_max > -1:
        #         gt_detected[k_max] = 1
        #         tp[j] = 1
        #
        # num_correct = sum(tp)
        # print(num_predicted_objects, num_correct, num_gt_objects)
        # count_predicted_bboxes += num_correct

        num_correct_rel, num_gt_pairs = calculate_recall_relationship(pairs, triplets, gt_pairs, gt_triplets, threshold)
        count_correct_relationship += num_correct_rel
        num_correct_phrase = calculate_recall_phrase(pairs, triplets, gt_pairs, gt_triplets, threshold)
        count_correct_phrase += num_correct_phrase
        count_gt_pairs += num_gt_pairs

    # with open(data_file, 'wb') as f:
    #     pickle.dump(list(data), f, pickle.HIGHEST_PROTOCOL)

    # with open(chosen_indices_file, 'wb') as f:
    #     pickle.dump(chosen_indices, f, pickle.HIGHEST_PROTOCOL)

    # print('average objects:', sum(list_count_predicted_objects)/1000)
    # print('average pairs:', sum(list_count_predicted_pairs) / 1000)

    # result = count_predicted_bboxes / count_gt_boxes
    result_rel = count_correct_relationship / count_gt_pairs
    result_phrase = count_correct_phrase / count_gt_pairs

    # print('object result:', result)
    print('rel result:', result_rel)
    print('phrase result:', result_phrase)


filter_box()


def extract():

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(path_to_ckpt, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    label_map = label_map_util.load_labelmap(path_to_labels)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=num_classes,
                                                                use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            dict_bbox = dict()

            for i, (image_id, relationships) in enumerate(test_relationship_data.items()):
                print(i, image_id)

                image_path = image_folder + 'sg_test_images/{}'.format(image_id)
                image = Image.open(image_path)
                # the array based representation of the image will be used later in order to prepare the
                # result image with boxes and labels on it.
                image_np = load_image_into_numpy_array(image)

                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)
                # Actual detection.
                boxes, scores, classes, num = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})

                if is_visualize:
                    # Visualization of the ground truth of a detection.
                    set_objects = set()

                    for j, x in enumerate(relationships):
                        object_bbox = x['object']['bbox']
                        subject_bbox = x['subject']['bbox']

                        object_bbox = tuple(object_bbox[l] for l in vrd_to_other)
                        subject_bbox = tuple(subject_bbox[l] for l in vrd_to_other)

                        object_id = x['object']['category']
                        subject_id = x['subject']['category']

                        object_tuple = (object_id, 1.) + object_bbox
                        set_objects.add(object_tuple)

                        subject_tuple = (subject_id, 1.) + subject_bbox
                        set_objects.add(subject_tuple)

                    if set_objects:
                        object_data = np.array(list(set_objects))
                        gt_boxes = object_data[:, 2:]
                        gt_scores = object_data[:, 1]
                        gt_classes = object_data[:, 0]

                        vis_util.visualize_boxes_and_labels_on_image_array(
                            image_np,
                            np.squeeze(gt_boxes),
                            np.squeeze(gt_classes).astype(np.int32),
                            np.squeeze(gt_scores),
                            gt_index,
                            agnostic_mode=True,
                            use_normalized_coordinates=False,
                            min_score_thresh=.5,
                            line_thickness=8)

                    # Visualization of the results of a detection.

                    vis_util.visualize_boxes_and_labels_on_image_array(
                        image_np,
                        np.squeeze(boxes),
                        np.squeeze(classes).astype(np.int32),
                        np.squeeze(scores),
                        category_index,
                        use_normalized_coordinates=True,
                        min_score_thresh=.2,
                        line_thickness=4)
                    plt.figure(figsize=IMAGE_SIZE)
                    plt.imshow(image_np)
                    plt.show()
                    plt.close()
                else:
                    height, width, _ = image_np.shape
                    scale = np.array([width, height, width, height])
                    boxes = np.round(boxes * scale).astype(int)
                    dict_bbox[image_id] = (boxes, scores, classes)

            if not is_visualize:
                with open(save_file_name, 'wb') as f:
                    pickle.dump(dict_bbox, f, pickle.HIGHEST_PROTOCOL)

# extract()
