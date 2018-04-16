import tensorflow as tf
import json
import cv2

from object_detection.utils import dataset_util


flags = tf.app.flags
FLAGS = flags.FLAGS

image_folder = '/scratch/datasets/Data1/sg_dataset/'

with open('data/objects.json', 'r') as f:
    list_objects = json.load(f)

list_bg_objects = [1, 2, 15, 17, 42, 48, 70, 97]
list_part_objects = [6, 10, 13, 19, 27, 33, 36, 39, 45, 57, 58, 62, 88, 96]
label_map = {i: object_name for i, object_name in enumerate(list_objects)}


def create_tf_example(image_id, set_objects, mode):
    # TODO(user): Populate the following variables from your example.
    image_path = image_folder + 'sg_' + mode + '_images/{}'.format(image_id)
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    height, width, _ = image.shape

    filename = image_path.encode()  # Filename of the image. Empty if image is not from file

    with tf.gfile.GFile(image_path, 'rb') as fid:
        encoded_image_data = fid.read()

    image_format = b'jpeg'  # b'jpeg' or b'png'

    xmins = []
    ymins = []
    xmaxs = []
    ymaxs = []
    classes = []
    classes_text = []

    for example in set_objects:
        object_id, y_min, y_max, x_min, x_max = example

        xmins.append(x_min/width)  # List of normalized left x coordinates in bounding box (1 per box)
        xmaxs.append(x_max/width)  # List of normalized right x coordinates in bounding box
        # (1 per box)
        ymins.append(y_min/height)  # List of normalized top y coordinates in bounding box (1 per box)
        ymaxs.append(y_max/height)  # List of normalized bottom y coordinates in bounding box
        # (1 per box)

        classes_text.append(label_map[object_id].encode())  # List of string class name of bounding box (1 per box)
        classes.append(object_id)  # List of integer class id of bounding box (1 per box)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_image_data),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main(_):

    # TODO(user): Write code to read in your dataset to examples variable
    with open('data/annotations_train.json', 'r') as f:
        train_relationship_data = json.load(f)

    with open('data/annotations_test.json', 'r') as f:
        test_relationship_data = json.load(f)

    def get_data_set(relationship_data, mode):

        writer = tf.python_io.TFRecordWriter('data/vrd_{}_detector.record'.format(mode))

        for i, (image_id, relationships) in enumerate(relationship_data.items()):

            set_objects = set()

            for j, x in enumerate(relationships):
                object_bbox = x['object']['bbox']
                subject_bbox = x['subject']['bbox']

                object_id = x['object']['category']
                subject_id = x['subject']['category']

                object_tuple = (object_id, ) + tuple(object_bbox)
                set_objects.add(object_tuple)

                subject_tuple = (subject_id, ) + tuple(subject_bbox)
                set_objects.add(subject_tuple)

            # list_region = list()
            #
            # for j, x in enumerate(relationships):
            #     object_bbox = x['object']['bbox']
            #     subject_bbox = x['subject']['bbox']
            #
            #     region_bbox = (min(object_bbox[0], subject_bbox[0]), max(object_bbox[1], subject_bbox[1]),
            #                    min(object_bbox[2], subject_bbox[2]), max(object_bbox[3], subject_bbox[3]))
            #
            #     list_region.append(region_bbox)

            tf_example = create_tf_example(image_id, set_objects, mode)
            writer.write(tf_example.SerializeToString())

        writer.close()

    get_data_set(train_relationship_data, 'train')
    get_data_set(test_relationship_data, 'test')


if __name__ == '__main__':
    tf.app.run()
