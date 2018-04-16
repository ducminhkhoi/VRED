import torch
import numpy as np
from torch.autograd import Variable
import time
import copy
from visdom import Visdom
from graphviz import Digraph
from torch.autograd import Variable, backward
from tqdm import tqdm
import sys
import io
import itertools
from globals import *
import torch.nn.functional as F
import torch.nn.init as init
from torch.optim.lr_scheduler import _LRScheduler
from operator import mul
from functools import reduce
viz = Visdom()


def selu(x):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale * F.elu(x, alpha)


def make_dot(var):
    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"), format='png')
    seen = set()

    def add_nodes(var):
        if var not in seen:
            if isinstance(var, Variable):
                value = '('+(', ').join(['%d'% v for v in var.size()])+')'
                dot.node(str(id(var)), str(value), fillcolor='lightblue')
            else:
                dot.node(str(id(var)), str(type(var).__name__))
            seen.add(var)
            if hasattr(var, 'previous_functions'):
                for u in var.previous_functions:
                    dot.edge(str(id(u[0])), str(id(var)))
                    add_nodes(u[0])
    add_nodes(var.creator)
    return dot


def accuracy(outputs, targets):
    _, preds = torch.max(outputs, 1)
    acc = torch.sum(preds == targets.squeeze())
    return acc.data[0]


class Model:
    def __init__(self, net, args=None):
        self.gpu = torch.cuda.is_available()
        if self.gpu:
            self.net = net.cuda()
        else:
            self.net = net
        self.losses = None
        self.optimizer = None
        self.decay_step = None
        self.weight_decay = None
        self.lr = None
        self.accuracy_fn = None
        self.scheduler = None
        self.args = args

    def summary(self):
        print(self.net)

    def eval(self):
        self.net.eval()

    def train(self):
        self.net.train()

    def compile(self, loss, opt, scheduler=None, accuracy_fn=accuracy, decay_step=None, weight_decay=1):
        """We can use add decay_step or weight decay"""
        if isinstance(loss, (list, tuple)):
            self.losses = loss
        else:
            self.losses = [loss]

        self.scheduler = scheduler
        self.optimizer = opt
        self.lr = opt.param_groups[0]['lr']

        self.accuracy_fn = accuracy_fn

        self.weight_decay = weight_decay
        if decay_step:
            self.decay_step = decay_step

    def visualize_net(self, input_size, to_file):
        input = Variable(torch.randn(*input_size))
        make_dot(self.net.forward(input)).render(to_file, view=True)

    def fit(self, X_train, Y_train, valid_set=None, epochs=30, batch_size=32, visualize=True):
        """Use when given pair X and Y instead of dataset Loader"""

        list_phases = ['train']
        best_loss = 0.0
        best_model = self.net

        # if provide validation set
        if valid_set:
            X_val, Y_val = valid_set
            nb_val_examples = len(X_val)
            list_phases.append('val')

            if not torch.is_tensor(X_val):
                X_val = torch.from_numpy(X_val)

            if not torch.is_tensor(Y_val):
                Y_val = torch.from_numpy(Y_val)

            nb_batches = int(nb_val_examples / batch_size) + 1
            X_val = torch.chunk(X_val, nb_batches)
            Y_val = torch.chunk(Y_val, nb_batches)

        nb_examples = len(X_train)

        if not torch.is_tensor(X_train):
            X_train = torch.from_numpy(X_train)

        if not torch.is_tensor(Y_train):
            Y_train = torch.from_numpy(Y_train)

        nb_batches = int(nb_examples / batch_size) + 1
        X_train = torch.chunk(X_train, nb_batches)
        Y_train = torch.chunk(Y_train, nb_batches)

        list_train_loss = []
        list_valid_loss = []

        for epoch in range(epochs):
            print('Epoch {}/{}'.format(epoch, epochs - 1))
            print('-' * 10)

            for phase in list_phases:
                if phase == 'train':
                    X, Y = X_train, Y_train
                else:
                    X, Y = X_val, Y_val

                running_loss = 0.0

                for i, data in enumerate(zip(X, Y)):
                    x, y = data
                    if self.gpu:
                        x, y = Variable(x.cuda()), Variable(y.cuda())
                    else:
                        x, y = Variable(x), Variable(y)

                    # zero the parameter gradients, very important
                    self.optimizer.zero_grad()

                    y_predict = self.net(x)
                    loss = self.losses(y_predict, y)

                    if phase == 'train':
                        if self.decay_step:
                            """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
                            lr = self.lr * (self.weight_decay ** (epoch // self.decay_step))
                            for param_group in self.optimizer.param_groups:
                                param_group['lr'] = lr

                        loss.backward()
                        self.optimizer.step()

                    running_loss += loss.data[0]

                    if phase == 'train':
                        print('train mode', 'epoch:', epoch, 'batch', i, 'of', X.shape[0],
                              'loss:', round(running_loss * 1. / (i+1), 3))

                epoch_loss = running_loss*1.0/nb_examples

                if phase == 'train':
                    list_train_loss.append(epoch_loss)
                else:
                    list_valid_loss.append(epoch_loss)

                print('{} Loss: {:.4f}'.format(phase, epoch_loss))

                if phase == 'val' and epoch_loss > best_loss:
                    best_loss = epoch_loss
                    best_model = copy.deepcopy(self.net)

            if visualize:  # visualize to the url: http://localhost:8097
                if valid_set:
                    if epoch == 0:
                        loss_plot = viz.line(np.zeros([1, 2]),
                                             opts=dict(
                                                 legend=['train_loss', 'val_loss'],
                                                 title="Training loss"
                                             ))
                    else:
                        viz.updateTrace(X=np.column_stack((np.arange(len(list_train_loss)), np.arange(len(list_valid_loss)))),
                                        Y=np.column_stack((np.array(list_train_loss), np.array(list_valid_loss))),
                                        win=loss_plot, append=False)
                else:
                    if epoch == 0:
                        loss_plot = viz.line(np.zeros([1]),
                                             opts=dict(
                                                 legend=['train_loss'],
                                                 title="Training loss"
                                             ))
                    else:
                        viz.updateTrace(X=np.arange(len(list_train_loss)),
                                        Y=np.array(list_train_loss),
                                        win=loss_plot, append=False)

        if valid_set:
            self.net = best_model

        print("Finish Training")

    def fit_loader(self, dset_loaders, dset_sizes, env, evaluate_fn=None, batch_size=16, num_epochs=30,
                   loss_weights=None, visualize=True, model_checkpoint=None, monitor_outputs=None, display_acc=None):
        """Use for train/fit data with dataset loader instead of pair X, Y, including 2 phases Train and Validation"""

        since = time.time()
        train_only = len(dset_loaders) == 1

        best_model = self.net
        best_acc = 0.0
        list_train_loss = []
        list_valid_loss = []

        if monitor_outputs:
            list_train_acc = []
            list_valid_acc = []

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in dset_loaders.keys():

                if phase == 'train' and self.scheduler and epoch > 0:
                    if isinstance(self.scheduler, _LRScheduler):
                        self.scheduler.step()
                    else:
                        self.scheduler.step(list_train_loss[-1] if train_only else list_valid_loss[-1])

                running_loss = 0.0

                if monitor_outputs:
                    running_corrects = 0.
                    running_accs = []

                is_volatile = phase == 'val'

                # Iterate over data.
                with tqdm(total=int(dset_sizes[phase]/batch_size)) as pbar:
                    dataloader = dset_loaders[phase]

                    for i, data in enumerate(dataloader):
                        # get the inputs
                        inputs, labels = data
                        # original_labels = labels

                        if not isinstance(inputs, (list, tuple)):
                            inputs = [inputs]
                        if not isinstance(labels, (list, tuple)):
                            labels = [labels]
                        if loss_weights is None:
                            loss_weights = [1 for _ in range(len(self.losses))]

                        # wrap them in Variable
                        if self.gpu:
                            inputs = [Variable(x.cuda(), volatile=is_volatile) for x in inputs]
                            labels = [Variable(x.cuda(), volatile=is_volatile) for x in labels]
                        else:
                            inputs = [Variable(x, volatile=is_volatile) for x in inputs]
                            labels = [Variable(x, volatile=is_volatile) for x in labels]

                        # forward
                        outputs = self.net(inputs)
                        if not isinstance(outputs, (tuple, list)):
                            outputs = [outputs]

                        if isinstance(outputs[0], (tuple, list)):  # special output, object detection only
                            loss, loss_l, loss_c = loss_weights[0] * self.losses[0](outputs[0], labels)
                        else:
                            loss = loss_weights[0] * self.losses[0](outputs[0], labels[0].view(-1))

                            for k in range(1, len(outputs)):
                                if len(self.losses) == 1:
                                    loss += loss_weights[k]*self.losses[0](outputs[k], labels[k].view(-1))
                                else:
                                    if k == 4:  # special loss
                                        label = torch.cat(labels[0:3], 1)
                                    elif labels[k].size(1) > 1:
                                        label = labels[k]
                                    else:
                                        label = labels[k].view(-1)
                                    loss += loss_weights[k]*self.losses[k](outputs[k], label)

                        # computer accuracy
                        if monitor_outputs:
                            if isinstance(outputs[0], (tuple, list)):  # special output, object detection only
                                accs = [loss.data[0], loss_l.data[0], loss_c.data[0]]
                            else:
                                accs = [self.accuracy_fn(outputs[i], labels[i]) for i in monitor_outputs]

                        # print(batch_correct)
                        running_loss += loss.data[0]

                        if monitor_outputs:
                            running_accs.append(accs)
                            running_corrects += accs[display_acc]

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            # zero the parameter gradients
                            self.optimizer.zero_grad()

                            loss.backward()

                            self.optimizer.step()

                            pbar.set_description('{} of {}'.format(i*batch_size, int(dset_sizes[phase])))
                            if monitor_outputs:
                                pbar.set_postfix(loss=round(running_loss*1./((i+1)*batch_size), 3),
                                             acc=round(running_corrects*1./((i+1)*batch_size), 3))
                            else:
                                pbar.set_postfix(loss=round(running_loss * 1. / ((i + 1) * batch_size), 3))

                            pbar.update()

                epoch_loss = running_loss*1. / dset_sizes[phase]

                if monitor_outputs:
                    epoch_acc = (np.array(running_accs).sum(0) / dset_sizes[phase]).tolist()
                    epoch_correct = running_corrects * 1. / dset_sizes[phase]

                if phase == 'train':
                    list_train_loss.append(epoch_loss)
                    if monitor_outputs:
                        list_train_acc.append(epoch_acc)
                else:
                    list_valid_loss.append(epoch_loss)
                    if monitor_outputs:
                        list_valid_acc.append(epoch_acc)

                if monitor_outputs:
                    print('{} Loss: {:.4f} Acc: {}, current lr:{}'.format(
                        phase, epoch_loss, epoch_acc, self.optimizer.param_groups[-1]['lr']))
                else:
                    print('{} Loss: {:.4f}, current lr:{}'.format(
                        phase, epoch_loss, self.optimizer.param_groups[-1]['lr']))

            if evaluate_fn:
                evaluate_fn(self.net)

            if model_checkpoint:
                if train_only:
                    self.save(model_checkpoint.format(train_loss=list_train_loss[-1], train_acc=list_train_acc[-1][0],
                                                      epoch=epoch))
                elif monitor_outputs:
                    self.save(model_checkpoint.format(val_loss=list_valid_loss[-1], val_acc=list_valid_acc[-1][0],
                                                      train_loss=list_train_loss[-1], train_acc=list_train_acc[-1][0],
                                                      epoch=epoch))
                else:
                    self.save(model_checkpoint.format(val_loss=list_valid_loss[-1], train_loss=list_train_loss[-1],
                                                      epoch=epoch))

            if visualize:  # visualize to the url: http://localhost:8097
                if epoch == 0:
                    viz.close(env)
                    viz.text(str(self.args), env=env)
                    loss_plot = viz.line(np.zeros([1] if train_only else [1, 2]), env=env,
                                         opts=dict(
                                             legend=['train_loss'] if train_only else ['train_loss', 'val_loss'],
                                             title="Training loss"
                                         ))

                    if monitor_outputs:
                        acc_legend = ['train_acc({})'.format(i) for i in monitor_outputs]
                        if not train_only:
                            acc_legend += ['val_acc({})'.format(i) for i in monitor_outputs]

                        acc_plot = viz.line(np.zeros([1, len(monitor_outputs)*(1 if train_only else 2)]),
                                            env=env,
                                            opts=dict(legend=acc_legend, title="Training accuracy"))

                else:
                    viz.updateTrace(X=np.array([np.arange(len(list_train_loss))] if train_only
                                               else [np.arange(len(list_train_loss)), np.arange(len(list_valid_loss))]).T,
                                    Y=np.array([list_train_loss] if train_only else
                                               [list_train_loss, list_valid_loss]).T,
                                    win=loss_plot, env=env, append=False)

                    if monitor_outputs:
                        train_acc = np.array(list_train_acc)
                        val_acc = np.array(list_valid_acc)
                        viz.updateTrace(
                            X=np.repeat(np.arange(len(list_train_acc)).reshape(-1, 1),
                                        len(monitor_outputs)*(1 if train_only else 2), 1),
                            Y=train_acc if train_only else np.concatenate((train_acc, val_acc), axis=1),
                            win=acc_plot, env=env, append=False)
            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))
        self.net = best_model

    def predict_loader(self, dset_loaders, dset_sizes, batch_size=16):
        """Use for train/fit data with dataset loader instead of pair X, Y, including 2 phases Train and Validation"""

        phase = 'val'
        list_outputs = []
        list_labels = []

        # Iterate over data.
        with tqdm(total=int(dset_sizes[phase]/batch_size)) as pbar:
            for i, data in enumerate(dset_loaders[phase]):
                # get the inputs
                inputs, labels = data
                list_labels.append(labels)

                if not isinstance(inputs, (list, tuple)):
                    inputs = [inputs]

                # wrap them in Variable
                if self.gpu:
                    inputs = [Variable(x.cuda(), volatile=True) for x in inputs]
                else:
                    inputs = [Variable(x, volatile=True) for x in inputs]

                # forward
                outputs = self.net(inputs)

                if self.gpu:
                    list_outputs.append(outputs.data.cpu())
                else:
                    list_outputs.append(outputs.data)

        print("Finish Predicting")

        return torch.cat(list_outputs, 0), torch.cat(list_labels, 0)

    def predict(self, X, batch_size=32):

        nb_examples = len(X)
        if not torch.is_tensor(X):
            X = torch.from_numpy(X)

        nb_batches = int(nb_examples / batch_size) + 1
        X = torch.chunk(X, nb_batches)

        Y = []

        for x in X:
            if self.gpu:
                x = Variable(x.cuda())
                y = self.net(x).cpu().data.numpy()
            else:
                x = Variable(x)
                y = self.net(x).data.numpy()

            Y.extend(y)

        Y = np.array(Y)

        print("Finish Predicting")

        return Y

    def save(self, file_path):
        torch.save(self.net.state_dict(), file_path)
        print("save model to file successful")

    def load(self, file_path):
        state_dict = torch.load(file_path, map_location=lambda storage, loc: storage)
        self.net.load_state_dict(state_dict)
        print("load model from file successful")

    def to_variable(self, x, volatile=False):
        if self.gpu:
            x = x.cuda()
        x = Variable(x, volatile=volatile)
        return x

    def to_numpy(self, x):
        if self.gpu:
            x = x.data.cpu()
        x = x.numpy()
        return x


def to_one_hot(x, size):
    out = [0] * size
    out[x] = 1
    return out


def adapt_array(arr):
    """
    http://stackoverflow.com/a/31312102/190597 (SoulNibbler)
    """
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())


def convert_array(text):
    return np.frombuffer(text, dtype=np.float32)


# Converts np.array to TEXT when inserting
sqlite3.register_adapter(np.ndarray, adapt_array)

# Converts TEXT to np.array when selecting
sqlite3.register_converter("array", convert_array)


class SQLiteDatabase:

    def __init__(self, fname, database_name, create_sql=None):
        """
        Args:
            fname: location of the database.
        Returns:
            db: a SQLite3 database with an relation table.
        """

        db = sqlite3.connect(fname, detect_types=sqlite3.PARSE_DECLTYPES)

        if create_sql:
            c = db.cursor()
            c.execute(create_sql)
            db.commit()

        self.db = db
        self.database_name = database_name

    def __len__(self):
        """
        Returns:
            count: number of relations in the database.
        """
        c = self.db.cursor()
        q = c.execute('select count(*) from ' + self.database_name)
        self.db.commit()
        return q.fetchone()[0]

    def insert_batch(self, batch):
        """
        Args:
            batch: a list of relations to insert, each of which is a tuple `(word, embeddings)`.
        """
        c = self.db.cursor()
        placeholders = '?' + ', ?' * (len(batch[0]) - 1)
        try:
            c.executemany("insert or ignore into " + self.database_name + " values (" + placeholders + ")", batch)
            self.db.commit()
        except Exception as e:
            print(e)
            raise e

    def insert_row(self, row_data):
        """
        Args:
            row_sql: `(word, embeddings)`.
            row_data: .
        """
        c = self.db.cursor()
        row_sql = '?'+', ?' * (len(row_data)-1)
        c.execute("insert or ignore into " + self.database_name + ' values (' + row_sql + ')', row_data)
        self.db.commit()

    def __contains__(self, w):
        """
        Args:
            w: word to look up.
        Returns:
            whether an embedding for `w` exists.
        """
        return self.lookup(w) is not None

    def clear(self):
        """
        Deletes all embeddings from the database.
        """
        c = self.db.cursor()
        c.execute('delete from ' + self.database_name)
        self.db.commit()

    def lookup(self, id_value, id_field=None):
        """
        Args:
            w: word to look up.
        Returns:
            embeddings for `w`, if it exists.
            `None`, otherwise.
        """
        c = self.db.cursor()
        if id_field:
            field_name = id_field
        else:
            field_name = 'id'

        q = c.execute('select * from ' + self.database_name + ' where ' + field_name + '= :id',
                      {'id': id_value}).fetchone()

        return q


def to_categorical(y, num_classes=None):
    """Converts a class vector (integers) to binary class matrix.

    E.g. for use with categorical_crossentropy.

    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.

    # Returns
        A binary matrix representation of the vrd_input.
    """
    y = np.array(y, dtype='int').ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1.
    return categorical


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j].round(2),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def xavier(param):
    init.xavier_uniform(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()
