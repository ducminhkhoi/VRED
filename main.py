from globals import *
from support_functions import get_dict_from_file, transform, spatial_transform, convert_to_string

from utils import Model, SQLiteDatabase, to_categorical

from itertools import combinations
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_normal, constant
import csv
import sys
from time import time

# Global variables
index_to_emb_dict, index_to_object_dict = get_dict_from_file('data/objects.json')
# index_to_emb_dict, index_to_predicate_dict = get_dict_from_file('data/predicates.json')
object_size = len(index_to_object_dict)  # 100
predicate_size = 70  # 70

batch_size = 32
num_epochs = 32
use_model = 2
use_loader = 1
mode = 'train_two_class'
env_name = 'relation_detection_use_model_{}_num_units_{}'.format(use_model, 10)
Object = namedtuple('Object', ['category', 'ymin', 'ymax', 'xmin', 'xmax'], verbose=False)

if isGPU:
    dtype = torch.cuda.FloatTensor
else:
    dtype = torch.FloatTensor


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        original_model = models.resnet50(pretrained=True)
        self.activate_fn = F.softmax

        self.fc1_size = 128
        self.predicate_feature_size = 2048
        self.object_feature_size = 2048
        self.cnn_output = 2048
        self.num_units = 12

        self.visual_features = nn.Sequential(
            original_model.conv1,
            original_model.bn1,
            original_model.relu,
            original_model.maxpool,
            original_model.layer1,
            original_model.layer2,
            original_model.layer3,
            original_model.layer4,
            original_model.avgpool,
        )

        # for param in self.visual_features.parameters():
        #     param.requires_grad = False

        self.spatial_feature = nn.Sequential(
            nn.Conv2d(2, 96, kernel_size=(5, 5), stride=(2, 2), padding=2),
            nn.ReLU(),
            nn.Conv2d(96, 128, kernel_size=(5, 5), stride=(2, 2), padding=2),
            nn.Conv2d(128, 64, kernel_size=(8, 8)),
            nn.ReLU()
        )

        self.fc_concat_relation_feature = nn.Sequential(
            nn.Linear(self.cnn_output + 64, self.fc1_size),
            nn.ReLU(),
            nn.Linear(self.fc1_size, self.predicate_feature_size),
            nn.ReLU()
        )

        # self.final_fc = nn.Sequential(
        #     nn.Linear(object_feature_size*2+predicate_feature_size, predicate_size),
        #     nn.LogSoftmax()
        # )

        self.Wa = Parameter(torch.randn(self.object_feature_size, object_size))
        self.Wr = Parameter(torch.randn(self.predicate_feature_size, predicate_size))
        self.Wsr = Parameter(torch.randn(predicate_size, object_size))
        self.Wso = Parameter(torch.randn(object_size, object_size))
        self.Wro = Parameter(torch.randn(object_size, predicate_size))

        # parameters to update with optimizer
        self.params = [self.Wa, self.Wr, self.Wsr, self.Wso, self.Wro]

    def get_features(self, inputs):
        inputs = inputs[0]
        xs = self.visual_features(inputs[0].view(-1, 3, 224, 224)).view(-1, self.object_feature_size)
        xo = self.visual_features(inputs[1].view(-1, 3, 224, 224)).view(-1, self.object_feature_size)

        predicate_visual_feature = self.visual_features(inputs[2].view(-1, 3, 224, 224)).view(-1, self.cnn_output)
        predicate_spatial_feature = self.spatial_feature(inputs[3].view(-1, 2, 32, 32))
        predicate_feature = torch.cat([predicate_visual_feature, predicate_spatial_feature], 1).view(-1,
                                                                                                     self.cnn_output + 64)
        xr = self.fc_concat_relation_feature(predicate_feature).view(-1, self.predicate_feature_size)

        return xs, xo, xr

    def forward(self, *inputs):
        xs, xo, xr = self.get_features(inputs)

        qs = self.activate_fn(xs.mm(self.Wa))
        qo = self.activate_fn(xo.mm(self.Wa))
        qr = self.activate_fn(xr.mm(self.Wr))

        for i in range(0, self.num_units):
            qs_new = self.activate_fn(xs.mm(self.Wa) + qr.mm(self.Wsr) + qo.mm(self.Wso))
            qo_new = self.activate_fn(xo.mm(self.Wa) + qs.mm(self.Wso) + qr.mm(self.Wro.t()))
            qr_new = self.activate_fn(xr.mm(self.Wr) + qs.mm(self.Wsr.t()) + qo.mm(self.Wro))
            qs, qo, qr = qs_new, qo_new, qr_new

        return F.log_softmax(qs), F.log_softmax(qo), F.log_softmax(qr)  # must return a list or a tuple


class MyModel2(nn.Module):  # Author's model
    def __init__(self):
        super(MyModel2, self).__init__()
        original_model = models.vgg16(pretrained=True)
        self.activate_fn = F.relu

        self.last_concat_fc_size = 128
        self.cnn_output = 4096
        self.visual_feature_output = 256
        self.num_units = 10

        last_visual_fc = nn.Linear(4096, self.visual_feature_output)
        xavier_normal(last_visual_fc.weight.data)
        constant(last_visual_fc.bias.data, 0)

        self.visual_features = original_model.features
        self.visual_classifer = nn.Sequential(
            *list(original_model.classifier.children())[:-1],
            last_visual_fc,
            nn.ReLU()
        )

        self.spatial_feature = nn.Sequential(
            nn.Conv2d(2, 96, kernel_size=(5, 5), stride=(2, 2), padding=2),
            nn.ReLU(),
            nn.Conv2d(96, 128, kernel_size=(5, 5), stride=(2, 2), padding=2),
            nn.Conv2d(128, 64, kernel_size=(8, 8)),
            nn.ReLU()
        )

        last_concat_fc = nn.Linear(self.last_concat_fc_size, predicate_size)
        xavier_normal(last_concat_fc.weight.data)
        constant(last_concat_fc.bias.data, 0)

        self.fc_concat_relation_feature = nn.Sequential(
            # customize layer here
            nn.Linear(self.visual_feature_output + 64, self.last_concat_fc_size),
            nn.ReLU(),
            last_concat_fc,  # PhiR_0
            nn.ReLU()
        )

        self.phi_a = nn.Linear(object_size, predicate_size)
        self.phi_b = nn.Linear(object_size, predicate_size)
        self.phi_r_0 = nn.Linear(predicate_size, predicate_size)
        self.phi_r_1 = nn.Linear(predicate_size, predicate_size)
        self.phi_r_2 = nn.Linear(predicate_size, predicate_size)
        self.phi_r_3 = nn.Linear(predicate_size, predicate_size)
        self.phi_r_4 = nn.Linear(predicate_size, predicate_size)
        self.phi_r_5 = nn.Linear(predicate_size, predicate_size)
        self.phi_r_6 = nn.Linear(predicate_size, predicate_size)
        self.phi_r_7 = nn.Linear(predicate_size, predicate_size)
        self.phi_r_8 = nn.Linear(predicate_size, predicate_size)
        self.phi_r_9 = nn.Linear(predicate_size, predicate_size)

        # Initialize
        # xavier_normal(self.phi_a.weight.data)
        # xavier_normal(self.phi_b.weight.data)
        # xavier_normal(self.phi_r.weight.data)
        # constant(self.phi_a.bias.data, 0)
        # constant(self.phi_b.bias.data, 0)
        # constant(self.phi_r.bias.data, 0)

        # self.params = [self.phi_a, self.phi_b, self.phi_r]

    def get_features(self, inputs):
        inputs = inputs[0]
        qs = inputs[0].view(-1, object_size)
        qo = inputs[1].view(-1, object_size)

        x = self.visual_features(inputs[2].view(-1, 3, 224, 224)).view(-1, 512 * 7 * 7)
        predicate_visual_feature = self.visual_classifer(x).view(-1, self.visual_feature_output)
        predicate_spatial_feature = self.spatial_feature(inputs[3].view(-1, 2, 32, 32))
        predicate_feature = torch.cat([predicate_visual_feature, predicate_spatial_feature], 1) \
            .view(-1, self.visual_feature_output + 64)
        qr = self.fc_concat_relation_feature(predicate_feature).view(-1, predicate_size)

        return qs, qo, qr

    def forward(self, *inputs):
        qs, qo, qr = self.get_features(inputs)

        for i in range(self.num_units):
            qr = self.activate_fn(getattr(self, 'phi_r_'+str(i))(qr) + self.phi_a(qs) + self.phi_b(qo))

        # qr = F.log_softmax(self.phi_r(qr) + self.phi_a(qs) + self.phi_b(qo))

        return qr  # must return a list or a tuple


class MyModel3(nn.Module):  # Author's model
    def __init__(self):
        super(MyModel3, self).__init__()
        original_model = models.vgg16(pretrained=True)
        self.activate_fn = F.relu

        self.last_concat_fc_size = 128
        self.cnn_output = 4096
        self.visual_feature_output = 256
        self.num_units = 10
        self.word_feature_size = 300

        last_visual_fc = nn.Linear(4096, self.visual_feature_output)
        xavier_normal(last_visual_fc.weight.data)
        constant(last_visual_fc.bias.data, 0)

        self.visual_features = original_model.features
        self.visual_classifer = nn.Sequential(
            *list(original_model.classifier.children())[:-1],
            last_visual_fc,
            nn.ReLU()
        )

        self.spatial_feature = nn.Sequential(
            nn.Conv2d(2, 96, kernel_size=(5, 5), stride=(2, 2), padding=2),
            nn.ReLU(),
            nn.Conv2d(96, 128, kernel_size=(5, 5), stride=(2, 2), padding=2),
            nn.Conv2d(128, 64, kernel_size=(8, 8)),
            nn.ReLU()
        )

        last_concat_fc = nn.Linear(self.last_concat_fc_size, predicate_size)
        xavier_normal(last_concat_fc.weight.data)
        constant(last_concat_fc.bias.data, 0)

        self.fc_concat_relation_feature = nn.Sequential(
            # customize layer here
            nn.Linear(self.visual_feature_output + 64, self.last_concat_fc_size),
            nn.ReLU(),
            last_concat_fc,  # PhiR_0
            nn.ReLU()
        )

        self.phi_a = nn.Linear(self.word_feature_size, predicate_size)
        self.phi_b = nn.Linear(self.word_feature_size, predicate_size)
        self.phi_r = nn.Linear(predicate_size, predicate_size)

        # Initialize
        xavier_normal(self.phi_a.weight.data)
        xavier_normal(self.phi_b.weight.data)
        xavier_normal(self.phi_r.weight.data)
        constant(self.phi_a.bias.data, 0)
        constant(self.phi_b.bias.data, 0)
        constant(self.phi_r.bias.data, 0)

        self.params = [self.phi_a, self.phi_b, self.phi_r]

    def get_features(self, inputs):
        inputs = inputs[0]
        qs = inputs[0].view(-1, self.word_feature_size)
        qo = inputs[1].view(-1, self.word_feature_size)

        x = self.visual_features(inputs[2].view(-1, 3, 224, 224)).view(-1, 512 * 7 * 7)
        predicate_visual_feature = self.visual_classifer(x).view(-1, self.visual_feature_output)
        predicate_spatial_feature = self.spatial_feature(inputs[3].view(-1, 2, 32, 32))
        predicate_feature = torch.cat([predicate_visual_feature, predicate_spatial_feature], 1) \
            .view(-1, self.visual_feature_output + 64)
        qr = self.fc_concat_relation_feature(predicate_feature).view(-1, predicate_size)

        return qs, qo, qr

    def forward(self, *inputs):
        qs, qo, qr = self.get_features(inputs)

        for i in range(self.num_units-1):
            qr = self.activate_fn(self.phi_r(qr) + self.phi_a(qs) + self.phi_b(qo))

        qr = F.log_softmax(self.phi_r(qr) + self.phi_a(qs) + self.phi_b(qo))

        return qr  # must return a list or a tuple


class FileDataset(Dataset):
    def __init__(self, json_file, type_dataset):
        self.type_dataset = type_dataset
        with open(json_file, 'r') as file:
            contents = json.load(file)
            contents = convert_to_string(contents).items()

            self.data = [(key, rel) for key, value in contents for rel in value if key != '4392556686_44d71ff5a0_o.jpg']

        print("number of relations:", len(self.data))

    def __getitem__(self, index):
        """Return a (transformed) vrd_input and target sample from an integer index"""
        key, rel = self.data[index]
        subject_box = rel['subject']['bbox']  # [ymin, ymax, xmin, xmax]
        object_box = rel['object']['bbox']

        minbbox = [min(subject_box[0], object_box[0]), max(subject_box[1], object_box[1]),
                   min(subject_box[2], object_box[2]), max(subject_box[3], object_box[3])]

        image = imread('/scratch/datasets/sg_dataset/sg_' + self.type_dataset + '_images/' + key)
        bboxes = [subject_box, object_box, minbbox]

        list_image = [image[bbox[0]:bbox[1], bbox[2]:bbox[3]] for bbox in bboxes]

        subject_visual_input, object_visual_input, union_visual_input = tuple(transform(x) for x in list_image)

        list_binary_image = [np.zeros_like(image) for _ in range(len(bboxes))]
        for (binary_image, bbox) in zip(list_binary_image, bboxes):
            binary_image[bbox[0]:bbox[1], bbox[2]:bbox[3]] = 1

        subject_spatial_input, object_spatial_input, union_spatial_input = \
            tuple(spatial_transform(x)[0, :, :].view(1, 32, 32) for x in list_binary_image)

        predicate_spatial_feature = torch.cat([subject_spatial_input, object_spatial_input], 0)

        object_word_feature = torch.FloatTensor(index_to_emb_dict[rel['object']['category']])
        subject_word_feature = torch.FloatTensor(index_to_emb_dict[rel['subject']['category']])

        if use_model == 1:
            input_sample = union_visual_input, predicate_spatial_feature
            target_sample = rel['subject']['category'], rel['object']['category'], rel['predicate']
        elif use_model == 2:
            input_sample = torch.FloatTensor(to_categorical(rel['subject']['category'], object_size)), \
                           torch.FloatTensor(to_categorical(rel['object']['category'], object_size)), \
                           union_visual_input, predicate_spatial_feature
            target_sample = rel['predicate']
        elif use_model == 3:
            input_sample = (subject_word_feature,
                            object_word_feature,
                            union_visual_input,
                            predicate_spatial_feature)
            target_sample = rel['predicate']

        return input_sample, target_sample

    def __len__(self):
        """Number of samples"""
        return len(self.data)


class SQLiteDataset(Dataset):
    def __init__(self, json_file, type_dataset):
        self.type_dataset = type_dataset
        self.db_file = '/scratch/datasets/vrd/relations_' + type_dataset + '.db'
        self.table_name = 'relations'
        self.batch_database_size = 1000

        with open(json_file, 'r') as file:
            contents = json.load(file)
            contents = convert_to_string(contents).items()

            data = [(key, rel) for key, value in contents for rel in value]

        print("number of relations:", len(data))

        create_sql = ('create table if not exists relations(' +
                      'id integer primary key,' +
                      'image_id text,' +
                      'subject_visual_feature array,' +
                      'subject_word_feature array,' +
                      'object_visual_feature array,' +
                      'object_word_feature array,' +
                      'predicate_visual_feature array,' +
                      'predicate_spatial_feature array,' +
                      'subject_id integer,' +
                      'object_id integer,' +
                      'predicate_id integer' +
                      ')')
        self.db = SQLiteDatabase(self.db_file, self.table_name, create_sql)

        num_records = len(self.db)

        if num_records < len(data):
            list_relation = []
            count_relation = num_records

            for i, (key, rel) in enumerate(data):
                if i < num_records:
                    continue

                print(i, len(data))
                subject_box = rel['subject']['bbox']  # [ymin, ymax, xmin, xmax]
                object_box = rel['object']['bbox']

                minbbox = [min(subject_box[0], object_box[0]), max(subject_box[1], object_box[1]),
                           min(subject_box[2], object_box[2]), max(subject_box[3], object_box[3])]

                image = imread('/scratch/datasets/vrd/sg_dataset/sg_' + type_dataset + '_images/' + key)
                bboxes = [subject_box, object_box, minbbox]

                list_image = [image[bbox[0]:bbox[1], bbox[2]:bbox[3]] for bbox in bboxes]

                list_binary_image = [np.zeros_like(image) for _ in range(len(bboxes))]
                for (binary_image, bbox) in zip(list_binary_image, bboxes):
                    binary_image[bbox[0]:bbox[1], bbox[2]:bbox[3]] = 1

                subject_visual_input, object_visual_input, union_visual_input = tuple(transform(x) for x in list_image)
                subject_spatial_input, object_spatial_input, union_spatial_input = \
                    tuple(spatial_transform(x)[0, :, :].view(1, 32, 32) for x in list_binary_image)

                predicate_spatial_feature = torch.cat([subject_spatial_input, object_spatial_input], 0)

                # subject_word_feature = np.array(emb.emb(index_to_object_dict[rel['subject']['category']]), dtype=np.float32)
                # object_word_feature = np.array(emb.emb(index_to_object_dict[rel['subject']['category']]), dtype=np.float32)

                relation = {'id': count_relation, 'image_id': key,
                            'subject_visual_feature': subject_visual_input.numpy(),
                            # 'subject_word_feature': subject_word_feature,
                            'object_visual_feature': object_visual_input.numpy(),
                            # 'object_word_feature': object_word_feature,
                            'predicate_visual_feature': union_visual_input.numpy(),
                            'predicate_spatial_feature': predicate_spatial_feature.numpy(),
                            'subject_id': rel['subject']['category'],
                            'object_id': rel['object']['category'],
                            'predicate_id': rel['predicate'],
                            }
                list_relation.append(relation)

                count_relation += 1

                if len(list_relation) == self.batch_database_size:
                    print("start inserting")
                    self.db.insert_batch(list_relation)
                    list_relation.clear()

            if list_relation:
                self.db.insert_batch(list_relation)
                list_relation.clear()

    def __getitem__(self, index):
        """Return a (transformed) vrd_input and target sample from an integer index"""

        result = list(self.db.lookup(index))

        input_sample = tuple(torch.FloatTensor(result[i]) for i in [2, 4, 6, 7])
        target_sample = tuple(torch.LongTensor([x]) for x in result[-3:])

        return input_sample, target_sample

    def __len__(self):
        """Number of samples"""
        return len(self.db)


class RedisDataset(Dataset):
    def __init__(self, json_file, type_dataset):
        self.type_dataset = type_dataset
        self.db = redis.StrictRedis(host='localhost', port=6379, db=0)
        self.batch_database_size = 1000

        with open(json_file, 'r') as file:
            contents = json.load(file)
            contents = convert_to_string(contents).items()

            self.data = [(key, rel) for key, value in contents for rel in value]

        print("number of relations:", len(self.data))

        self.num_records = self.db.dbsize()
        self.db.config_set('stop-writes-on-bgsave-error', 'yes')
        self.db.config_set('dir', '/scratch/datasets/vrd')
        self.db.config_set('dbfilename', 'temp.rdb')
        if self.num_records < len(self.data):
            self.build_db()
            self.db.save()

    @staticmethod
    def save_numpy_array(nb_array):
        array_dtype = str(nb_array.dtype)
        sizes = nb_array.shape
        nb_data = nb_array.ravel().tostring()
        prefix_string = '{0}|' + '#'.join('{' + str(i) + '}' for i in range(1, len(sizes) + 1)) + '|'
        prefix = prefix_string.format(array_dtype, *sizes).encode('utf-8')
        return prefix + nb_data

    @staticmethod
    def load_numpy_array(nb_bytes):
        array_dtype, sizes = nb_bytes.split(b'|')[0], nb_bytes.split(b'|')[1]
        nb_array = nb_bytes[len(array_dtype + sizes) + 2:]
        sizes = tuple(int(x) for x in sizes.split(b'#'))
        nb_array = np.fromstring(nb_array, dtype=array_dtype)
        nb_array = nb_array.reshape(*sizes)
        return nb_array

    def build_db(self):
        count_relation = self.num_records

        for i, (key, rel) in enumerate(self.data):
            if i < self.num_records:
                continue

            print(i, len(self.data))
            subject_box = rel['subject']['bbox']  # [ymin, ymax, xmin, xmax]
            object_box = rel['object']['bbox']

            minbbox = [min(subject_box[0], object_box[0]), max(subject_box[1], object_box[1]),
                       min(subject_box[2], object_box[2]), max(subject_box[3], object_box[3])]

            image = imread('/scratch/datasets/vrd/sg_dataset/sg_' + self.type_dataset + '_images/' + key)
            bboxes = [subject_box, object_box, minbbox]

            list_image = [image[bbox[0]:bbox[1], bbox[2]:bbox[3]] for bbox in bboxes]

            list_binary_image = [np.zeros_like(image) for _ in range(len(bboxes))]
            for (binary_image, bbox) in zip(list_binary_image, bboxes):
                binary_image[bbox[0]:bbox[1], bbox[2]:bbox[3]] = 1

            subject_visual_input, object_visual_input, union_visual_input = tuple(transform(x) for x in list_image)
            subject_spatial_input, object_spatial_input, union_spatial_input = \
                tuple(spatial_transform(x)[0, :, :].view(1, 32, 32) for x in list_binary_image)

            predicate_spatial_feature = torch.cat([subject_spatial_input, object_spatial_input], 0)

            # subject_word_feature = np.array(emb.emb(index_to_object_dict[rel['subject']['category']]), dtype=np.float32)
            # object_word_feature = np.array(emb.emb(index_to_object_dict[rel['subject']['category']]), dtype=np.float32)

            relation = {'image_id': key,
                        'subject_visual_feature': self.save_numpy_array(subject_visual_input.numpy()),
                        # 'subject_word_feature': self.save_numpy_array(subject_word_feature.numpy()),
                        'object_visual_feature': self.save_numpy_array(object_visual_input.numpy()),
                        # 'object_word_feature': self.save_numpy_array(object_word_feature.numpy()),
                        'predicate_visual_feature': self.save_numpy_array(union_visual_input.numpy()),
                        'predicate_spatial_feature': self.save_numpy_array(predicate_spatial_feature.numpy()),
                        'subject_id': rel['subject']['category'],
                        'object_id': rel['object']['category'],
                        'predicate_id': rel['predicate'],
                        }

            self.db.hmset(count_relation, relation)

            count_relation += 1

    def __getitem__(self, index):
        """Return a (transformed) vrd_input and target sample from an integer index"""

        result = self.db.hgetall(str(index))

        if use_model == 1:
            input_sample = tuple([torch.from_numpy(self.load_numpy_array(result[b'subject_visual_feature'])),
                                  torch.from_numpy(self.load_numpy_array(result[b'object_visual_feature'])),
                                  torch.from_numpy(self.load_numpy_array(result[b'predicate_visual_feature'])),
                                  torch.from_numpy(self.load_numpy_array(result[b'predicate_spatial_feature']))])

            target_sample = tuple([torch.LongTensor([int(result[b'subject_id'])]),
                                   torch.LongTensor([int(result[b'object_id'])]),
                                   torch.LongTensor([int(result[b'predicate_id'])])])

        else:
            input_sample = tuple([torch.from_numpy(to_categorical(int(result[b'subject_id']), object_size)),
                                  torch.from_numpy(to_categorical(int(result[b'object_id']), object_size)),
                                  torch.from_numpy(self.load_numpy_array(result[b'predicate_visual_feature'])),
                                  torch.from_numpy(self.load_numpy_array(result[b'predicate_spatial_feature'])),
                                  ])

            target_sample = torch.LongTensor([int(result[b'predicate_id'])])

        return input_sample, target_sample

    def __len__(self):
        """Number of samples"""
        return self.db.dbsize()


class TensorDataset(Dataset):
    def __init__(self, json_file, type_dataset):
        self.type_dataset = type_dataset
        self.db_file = '/scratch/datasets/vrd/relations_' + type_dataset + '.data'

        with open(json_file, 'r') as file:
            contents = json.load(file)
            contents = convert_to_string(contents).items()

            self.data = [(key, rel) for key, value in contents for rel in value]

        print("number of relations:", len(self.data))

        self.list_relation = []

        if os.path.isfile(self.db_file):
            with open(self.db_file, 'rb') as f:
                self.list_relation = pkl.load(f)
        else:
            for i, (key, rel) in enumerate(self.data):
                print(i, len(self.data))
                subject_box = rel['subject']['bbox']  # [ymin, ymax, xmin, xmax]
                object_box = rel['object']['bbox']

                minbbox = [min(subject_box[0], object_box[0]), max(subject_box[1], object_box[1]),
                           min(subject_box[2], object_box[2]), max(subject_box[3], object_box[3])]

                image = imread('/scratch/datasets/vrd/sg_dataset/sg_' + type_dataset + '_images/' + key)
                bboxes = [subject_box, object_box, minbbox]

                list_image = [image[bbox[0]:bbox[1], bbox[2]:bbox[3]] for bbox in bboxes]

                list_binary_image = [np.zeros_like(image) for _ in range(len(bboxes))]
                for (binary_image, bbox) in zip(list_binary_image, bboxes):
                    binary_image[bbox[0]:bbox[1], bbox[2]:bbox[3]] = 1

                relation = [transform(x) for x in list_image] + \
                           [spatial_transform(x)[0, :, :].view(1, 32, 32) for x in list_binary_image] + \
                           [torch.LongTensor([rel['subject']['category'], rel['object']['category'], rel['predicate']])]

                self.list_relation.append(relation)

            with open(self.db_file, 'wb') as f:
                pkl.dump(self.list_relation, f, protocol=pkl.HIGHEST_PROTOCOL)

        print("complete_loading", type_dataset)

    def __getitem__(self, index):
        """Return a (transformed) vrd_input and target sample from an integer index"""
        result = self.list_relation[index]

        input_sample = [result[0], result[3], result[1], result[4], result[2], result[5]]
        target_sample = result[-1]

        return input_sample, target_sample

    def __len__(self):
        """Number of samples"""
        return len(self.data)


def eval_recall(dets, all_gts):  # Copy from Author's code
    top_recall = 100
    tp = []
    score = []
    total_num_gts = 0
    for gts, det in zip(all_gts, dets):
        det = det.reshape(-1, predicate_size)
        num_gts = len(gts)
        total_num_gts += num_gts
        gt_detected = np.zeros(num_gts)
        if det.shape[0] > 0:
            det_score = det
            top_dets = np.argmax(det_score, 1)
            top_scores = np.max(det_score, 1)
            if len(top_dets) > top_recall:
                top_dets = top_dets[:top_recall]
                top_scores = top_scores[:top_recall]
            for j in range(len(top_dets)):
                arg_max = -1
                for k in range(num_gts):
                    if gt_detected[k] == 0 and top_dets[j] == gts[k]:
                            arg_max = k
                if arg_max != -1:
                    gt_detected[arg_max] = 1
                    tp.append(1)
                else:
                    tp.append(0)
                score.append(top_scores[j])
    score = np.array(score)
    tp = np.array(tp)
    inds = np.argsort(score)
    inds = inds[::-1]
    tp = tp[inds]
    tp = np.cumsum(tp)
    recall = (tp + 0.0) / total_num_gts
    top_recall = recall[-1]
    print(top_recall)
    return top_recall


def run_evaluate(model, json_file, type_dataset):

    output_file = 'data/temp_output_3.pkl'

    if os.path.isfile(output_file):
        with open(output_file, 'rb') as f:
            list_relations, list_gt_relations = pickle.load(f)
    else:
        model.load('/scratch/datasets/vrd/weights.07-train_loss:0.10-train_acc:0.52-val_loss:0.12-val_acc:0.51.pkl')

        with open(json_file, 'r') as file:
            contents = json.load(file)
            contents = convert_to_string(contents).items()

        list_relations = []
        list_gt_relations = []

        for i, (key, value) in enumerate(contents):
            print(i, len(contents))
            list_relation = []
            list_gt_relation = []

            # get list of object
            list_objects = []
            for rel in value:
                list_objects.append(Object(rel['subject']['category'], *tuple(rel['subject']['bbox'])))
                list_objects.append(Object(rel['object']['category'], *tuple(rel['object']['bbox'])))
                list_gt_relation.append(rel['predicate'])

            list_objects = list(set(list_objects))

            for subject, object_ in combinations(list_objects, 2):
                subject_box = subject.ymin, subject.ymax, subject.xmin, subject.xmax  # [ymin, ymax, xmin, xmax]
                object_box = object_.ymin, object_.ymax, object_.xmin, object_.xmax

                minbbox = [min(subject_box[0], object_box[0]), max(subject_box[1], object_box[1]),
                           min(subject_box[2], object_box[2]), max(subject_box[3], object_box[3])]

                image = imread('/scratch/datasets/sg_dataset/sg_' + type_dataset + '_images/' + key)
                bboxes = [subject_box, object_box, minbbox]

                list_image = [image[bbox[0]:bbox[1], bbox[2]:bbox[3]] for bbox in bboxes]

                list_binary_image = [np.zeros_like(image) for _ in range(len(bboxes))]
                for (binary_image, bbox) in zip(list_binary_image, bboxes):
                    binary_image[bbox[0]:bbox[1], bbox[2]:bbox[3]] = 1

                subject_visual_input, object_visual_input, union_visual_input = tuple(transform(x) for x in list_image)
                subject_spatial_input, object_spatial_input, union_spatial_input = \
                    tuple(spatial_transform(x)[0, :, :].view(1, 32, 32) for x in list_binary_image)

                predicate_spatial_feature = torch.cat([subject_spatial_input, object_spatial_input], 0)

                inputs = (torch.FloatTensor(to_categorical(subject.category, object_size)),
                          torch.FloatTensor(to_categorical(object_.category, object_size)),
                          union_visual_input, predicate_spatial_feature)

                # wrap them in Variable
                if isGPU:
                    inputs = [Variable(x.cuda(), volatile=True) for x in inputs]
                else:
                    inputs = [Variable(x, volatile=True) for x in inputs]

                # forward
                outputs = model.net(inputs)

                if isGPU:
                    list_relation.append(outputs.data.cpu().numpy())
                else:
                    list_relation.append(outputs.data.numpy())

            list_relations.append(np.array(list_relation))
            list_gt_relations.append(list_gt_relation)

        with open(output_file, 'wb') as f:
            pickle.dump((list_relations, list_gt_relations), f, pickle.HIGHEST_PROTOCOL)

    score = eval_recall(list_relations, list_gt_relations)
    return score


if __name__ == '__main__':
    use_model = 2

    if use_model == 1:
        net = MyModel()
    elif use_model == 2:
        net = MyModel2()
    elif use_model == 3:
        net = MyModel3()

    model = Model(net)
    # model.summary()

    if mode == 'train_two_class':
        # if use_loader == 1:
        train_dataset = FileDataset('data/annotations_train.json',
                                    'train_two_class')  # slower but can load much larger datasets
        val_dataset = FileDataset('data/annotations_test.json', 'test')
        # elif use_loader == 2:
        #     train_dataset = SQLiteDataset('data/annotations_train.json',
        #                                   'train')  # faster but we cannot multi processing
        #     val_dataset = SQLiteDataset('data/annotations_test.json', 'test')
        # elif use_loader == 3:
        #     train_dataset = RedisDataset('data/annotations_train.json',
        #                                  'train')  # faster but loading database is very long
        #     val_dataset = RedisDataset('data/annotations_test.json', 'test')
        # else:
        #     train_dataset = TensorDataset('data/annotations_train.json',
        #                                   'train')  # very fast, but limited to size of RAM
        #     val_dataset = TensorDataset('data/annotations_test.json', 'test')

        train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=2)
        #
        # # for e in train_loader:
        # #     print(e)
        #
        dset_loaders = {'train_two_class': train_loader, 'val': val_loader}
        dset_sizes = {'train_two_class': len(train_dataset), 'val': len(val_dataset)}

        # dict_param = [{'params': x.parameters()} for x in net.phi_r]

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            net.parameters(),
            lr=1e-3,
            # momentum=0.9
        )

        model_checkpoint = "/scratch/datasets/vrd/weights.{epoch:02d}-train_loss:{train_loss:.2f}-train_acc:{train_acc:.2f}" \
                           + "-val_loss:{val_loss:.2f}-val_acc:{val_acc:.2f}.pkl"
        model.compile(loss=criterion, opt=optimizer, weight_decay=0.1, decay_step=10)
        model.fit_loader(dset_loaders, dset_sizes, batch_size=batch_size, num_epochs=num_epochs,
                         model_checkpoint=model_checkpoint, env=env_name)

    else:
        run_evaluate(model, 'data/annotations_test.json', 'test')
