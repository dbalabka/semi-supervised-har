from math import ceil
from sklearn.pipeline import Pipeline
from matplotlib.pyplot import plot
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer
import numpy as np
import pandas as pd
from IPython.core.display import display
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from .highlight import TimeseriesHighlightTransformer


def load_ucr_dataset(name, path='data/UCR_TS_Archive_2015/', is_fft=False, combine=False, highlight_augmentation=False, reshape=True, feature_range=(-1, 1)):
    train = pd.read_csv('%s/%s/%s_TRAIN' % (path, name, name), header=None, sep=',')
    test = pd.read_csv('%s/%s/%s_TEST' % (path, name, name), header=None, sep=',')

    if combine:
        train = test = pd.concat([train, test])

    if is_fft:
        X_train = np.fft.fft(train.loc[:, 1:].values)
        X_test = np.fft.fft(test.loc[:, 1:].values)
    else:
        X_train = train.loc[:, 1:].values
        X_test = test.loc[:, 1:].values

    y_train = train.loc[:, 0].values
    y_test = test.loc[:, 0].values

    display(X_train.shape)

    scaler = MinMaxScaler(feature_range=feature_range)
    # scaler = RobustScaler()


    plt.title('Before scaling')
    plot(X_train[0])
    plt.show()

    X_train, X_test, _ = np.split(
        scaler.fit_transform(np.concatenate((X_train, X_test), axis=0).transpose()).transpose(),
        [X_train.shape[0], X_train.shape[0] + X_test.shape[0]]
    )
    plt.title('After scaling')
    plot(X_train[0])
    plt.show()

    display(X_train.shape)
    display(X_test.shape)

    if highlight_augmentation:
        pipeline = Pipeline([
            ('timeseries_segmentation', TimeseriesHighlightTransformer()),
            ('transpose', FunctionTransformer(np.transpose)),
            ('scaler', MinMaxScaler(feature_range=(-1, 1))),
            ('transpose_back', FunctionTransformer(np.transpose)),
        ])
        X_train, X_test, _ = np.split(
            pipeline.fit_transform(np.concatenate((X_train, X_test), axis=0)),
            [X_train.shape[0], X_train.shape[0] + X_test.shape[0]]
        )
        plt.title('After highlight_augmentation')
        plot(X_train[0])
        plt.show()


    display(X_train.shape)
    display(X_test.shape)

    if reshape:
        train_measurment_count, train_series_len = X_train.shape
        test_measurment_count, test_series_len = X_test.shape

        # TODO: test new shape
        # X_train = np.apply_along_axis(lambda x: [x], 1, X_train)
        # X_test = np.apply_along_axis(lambda x: [x], 1, X_test)
        X_train = np.reshape(X_train, X_train.shape + (1,))
        X_test = np.reshape(X_test, X_test.shape + (1,))


        display(y_train)
        y_train = np.reshape(np.apply_along_axis(lambda x: np.repeat(x, train_series_len), 0,
                                                 np.reshape(y_train, (1, y_train.shape[0]))).transpose(),
                             (train_measurment_count, 1, train_series_len))
        display(y_test)
        y_test = np.reshape(np.apply_along_axis(lambda x: np.repeat(x, test_series_len), 0,
                                                np.reshape(y_test, (1, y_test.shape[0]))).transpose(),
                            (test_measurment_count, 1, test_series_len))

        # for series_num in range(0, X_train.shape[0]):
        #     plt.plot(train.loc[series_num].values[1:])
        # plt.show()

        f, plots = plt.subplots(len(np.unique(y_train)), 1)
        for series_num in range(0, X_train.shape[0]):
            color_num = y_train[series_num][0][0] % 10
            plots[color_num].plot(np.transpose(X_train[series_num])[0], color='C%s' % (color_num), alpha=.5)
        plt.show()

        f, plots = plt.subplots(len(np.unique(y_test)), 1)
        for series_num in range(0, X_test.shape[0]):
            color_num = y_test[series_num][0][0] % 10
            plots[color_num].plot(np.transpose(X_test[series_num])[0], color='C%s' % (color_num), alpha=.5)
        plt.show()

    return X_train, X_test, y_train, y_test


# Taken from https://github.com/musyoku/adversarial-autoencoder/blob/6a2334fb010845f2d4e98cdb11920d73f3bb245a/aae/dataset/semi_supervised.py#L42
class Dataset():

    def __init__(self, train, test, num_classes, num_labeled_data=5, num_extra_classes=0, use_test_as_unlabeled=False, batch_size=None):
        self.images_train, self.original_labels_train = train
        self.images_test, self.original_labels_test = test

        self.original_labels_train_map = np.unique(self.original_labels_train)
        self.original_labels_test_map = np.unique(self.original_labels_test)
        if self.original_labels_train_map.shape[0] != self.original_labels_test_map.shape[0]:
            raise AssertionError('train and test datasets contains different amount of classes')
        if self.original_labels_train_map.shape[0] != num_classes:
            raise AssertionError('train dataset amount of classes (%s) is not equal to num_classes (%s)' % (self.original_labels_train_map.shape[0], num_classes,))
        self.labels_train_map = np.arange(self.original_labels_train_map.shape[0])
        self.labels_test_map = np.arange(self.original_labels_test_map.shape[0])

        self.labels_train = self.normalize_labels(self.original_labels_train, self.original_labels_train_map, self.labels_train_map)
        self.labels_test = self.normalize_labels(self.original_labels_test, self.original_labels_test_map, self.labels_test_map)


        self.num_classes = num_classes
        self.num_extra_classes = num_extra_classes
        indices_count = len(self.images_train)
        indices = np.arange(0, indices_count)
        np.random.shuffle(indices)

        indices_u = []
        indices_l = []
        counts = [0] * num_classes
        # counts = [0] * (np.max(self.labels_train) + 1)
        # TODO: implement configuration per class
        num_per_class = num_labeled_data

        for index in indices:
            label = self.labels_train[index]
            if counts[label] < num_per_class:
                counts[label] += 1
                indices_l.append(index)
                continue
            indices_u.append(index)

        if use_test_as_unlabeled:
            indices_u.extend(np.arange(indices_count, indices_count + len(self.images_test)))
            self.images_train = np.concatenate((self.images_train, self.images_test,), axis=0)
            self.labels_train = np.concatenate((self.labels_train, self.labels_test,), axis=0)

        self.indices_l = np.asarray(indices_l)
        self.indices_u = np.asarray(indices_u)
        self.shuffle()

        self.max_items = max(self.get_num_labeled_data(), self.get_num_unlabeled_data())
        self.batch_size = int(min(self.max_items/10, 16)) if batch_size is None else batch_size
        self.steps = ceil(self.max_items / self.batch_size)

    def normalize_labels(self, y, original, new):
        def normalize(label):
            return new[original == label][0]
        return np.vectorize(normalize)(y)

    def summary(self):
        print("Train data count %s\n" % len(self.original_labels_train))
        print("Test data count %s\n" % len(self.original_labels_test))
        print("Labeled data count %s\n" % len(self.indices_l))
        print("Unlabeled data count %s\n" % len(self.indices_u))
        print("Labels: %s" % self.indices_l)
        print("Batch size %s and %s steps" % (self.batch_size, self.steps))

    def get_labeled_data(self):
        return self.images_train[self.indices_l], self.labels_train[self.indices_l]

    def get_num_labeled_data(self):
        return len(self.indices_l)

    def get_num_unlabeled_data(self):
        return len(self.indices_u)

    # def get_iterator(self, batchsize, train=False, test=False, labeled=False, unlabeled=False, gpu=True):
    #     if train:
    #         if labeled:
    #             return self.get_iterator_train_labeled(batchsize, gpu)
    #         if unlabeled:
    #             return self.get_iterator_train_unlabeled(batchsize, gpu)
    #         raise NotImplementedError()
    #
    #     if test:
    #         return self.get_iterator_test(batchsize, gpu)
    #
    #     raise NotImplementedError()
    #
    # def get_iterator_train_labeled(self, batchsize, gpu=True):
    #     return Iterator(self.images_train, self.labels_train, self.indices_l, batchsize, gpu)
    #
    # def get_iterator_train_unlabeled(self, batchsize, gpu=True):
    #     return Iterator(self.images_train, self.labels_train, self.indices_u, batchsize, gpu)

    def sample_labeled_minibatch(self, batchsize, gpu=False):
        if len(self.indices_l) == 0:
            raise Exception("Labeled data is empty")
        x_batch, y_batch, y_onehot_batch = self._sample_minibatch(self.indices_l[:batchsize], batchsize, gpu)
        self.indices_l = np.roll(self.indices_l, batchsize)
        return x_batch, y_batch, y_onehot_batch

    def sample_unlabeled_minibatch(self, batchsize, gpu=False):
        x_batch, y_batch, y_onehot_batch = self._sample_minibatch(self.indices_u[:batchsize], batchsize, gpu)
        self.indices_u = np.roll(self.indices_u, batchsize)
        return x_batch

    def _sample_minibatch(self, batch_indices, batchsize, gpu):
        x_batch = self.images_train[batch_indices]
        y_batch = self.labels_train[batch_indices]
        y_onehot_batch = to_categorical(y_batch, self.num_classes + self.num_extra_classes)

        if gpu:
            raise NotImplemented
            # x_batch = cuda.to_gpu(x_batch)
            # y_batch = cuda.to_gpu(y_batch)
            # y_onehot_batch = cuda.to_gpu(y_onehot_batch)

        return x_batch, y_batch, y_onehot_batch

    def shuffle(self):
        np.random.shuffle(self.indices_l)
        np.random.shuffle(self.indices_u)



