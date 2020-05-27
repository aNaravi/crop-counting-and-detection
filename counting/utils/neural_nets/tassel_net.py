import datetime
import csv
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

    def train(self, X, Y, processing_parameters, models_dir, training_parameters={}):
        self.model.compile(loss=training_parameters.get("loss", "mean_absolute_error"), optimizer=training_parameters.get("optimizer", "sgd"))
        print("[INFO]: Training Model")
        self.model.fit(X, Y,
                       batch_size=training_parameters.get("batch_size", 128),
                       validation_split=training_parameters.get("validation_split", 0.1),
                       epochs=training_parameters.get("epochs", 5))

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        self.model.save(models_dir + 'model_' + timestamp + '.hdf5')
        with open(models_dir + 'parameters_' + timestamp + '.csv', 'w') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerows(processing_parameters.items())

    def test(self, X, Y, aggregator):
        mean_absolute_errors, agg_predictions, agg_counts = [], [], []

        for sub_img_set, counts_set in zip(X, Y):
            predictions = self.model.predict(sub_img_set)

            agg_predicted_count = aggregator.aggregate_local_counts(predictions)
            agg_true_count = aggregator.aggregate_local_counts(counts_set)

            agg_predictions.append(agg_predicted_count)
            agg_counts.append(agg_true_count)
            mean_absolute_errors.append(abs(agg_predicted_count - agg_true_count))

        return mean_absolute_errors, agg_predictions, agg_counts

    def predict(self, X, aggregator):
        agg_predictions = []
        for sub_img_set in X:
            predictions = self.model.predict(sub_img_set)
            agg_predictions.append(aggregator.aggregate_local_counts(predictions))

        return agg_predictions
