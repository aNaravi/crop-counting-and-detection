import datetime
from scipy import stats
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from ..postprocessors import AggregateLocalCounts


class TasselNet:
    def __init__(self, load_path=None):
        if type(load_path) == str and load_path.split('.')[-1] == 'hdf5':
            self.model = load_model(load_path)
        else:
            self.model = None

    def build(self, architecture='alexnet', input_shape=(32,32,3)):
        if architecture == 'alexnet':
            self.model = Sequential()

            # Layer 1
            self.model.add(Conv2D(filters=16, kernel_size=(3,3), strides=(1,1), padding='same', input_shape=input_shape))
            self.model.add(Activation('relu'))
            self.model.add(BatchNormalization())
            self.model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

            # Layer 2
            self.model.add(Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding='same'))
            self.model.add(Activation('relu'))
            self.model.add(BatchNormalization())
            self.model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

            # Layer 3
            self.model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same'))
            self.model.add(Activation('relu'))
            self.model.add(BatchNormalization())

            # Layer 4
            self.model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same'))
            self.model.add(Activation('relu'))
            self.model.add(BatchNormalization())

            # Layer 5
            self.model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same'))
            self.model.add(Activation('relu'))
            self.model.add(BatchNormalization())
            self.model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

            # Layer 6
            self.model.add(Conv2D(filters=128, kernel_size=tuple(map(lambda x: x // 8, input_shape[:-1]))))
            self.model.add(Activation('relu'))
            self.model.add(BatchNormalization())

            # Layer 7
            self.model.add(Conv2D(filters=128, kernel_size=(1,1)))
            self.model.add(Activation('relu'))
            self.model.add(BatchNormalization())

            # Layer 8
            self.model.add(Conv2D(filters=1, kernel_size=(1,1)))
            self.model.add(Activation('relu'))
            self.model.add(BatchNormalization())

            # Layer 9
            self.model.add(Flatten())
            self.model.add(Dense(1, activation='linear'))

    def train(self, X, Y, **kwargs):
        self.model.compile(loss=kwargs.get("loss", "mean_absolute_error"), optimizer=kwargs.get("optimizer", "sgd"))
        print("[INFO]: Training Model")
        self.model.fit(X, Y,
                       batch_size=kwargs.get("batch_size", 1024),
                       validation_split=kwargs.get("validation_split", 0.1),
                       epochs=kwargs.get("epochs", 5))
        self.model.save(kwargs.get("save_folder",'') + 'model_' + datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S") + '.hdf5')

    def test(self, X, Y, **kwargs):
        agg = AggregateLocalCounts(img_shape=kwargs.get("img_dimensions", (384,1600)),
                                   sub_img_shape=kwargs.get("sub_img_dimensions", (32,32)),
                                   stride=kwargs.get("stride", 8))

        mean_absolute_errors, agg_predictions, agg_counts = [], [], []
        for sub_img_set, counts_set in zip(X, Y):
            predictions = self.model.predict(sub_img_set)
            agg_predicted_count, agg_true_count = agg.aggregate_local_counts(predictions), agg.aggregate_local_counts(counts_set)
            agg_predictions.append(agg_predicted_count)
            agg_counts.append(agg_true_count)
            mean_absolute_errors.append(abs(agg_predicted_count - agg_true_count))

        print("Summary MAE Statistics: ", stats.describe(mean_absolute_errors))

        return mean_absolute_errors, agg_predictions, agg_counts
