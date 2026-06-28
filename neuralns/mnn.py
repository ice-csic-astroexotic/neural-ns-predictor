"""Multi-output neural network"""

import os
import glob
import copy

import joblib
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tensorflow.keras import Input, Model, regularizers
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from livelossplot import PlotLossesKeras


DATASET_TYPES = ["train", "valid", "test", "holdout"]
DATASET_COLORS = ["b", "g", "m", "r"]
COLOR_BY_TYPE = {typ: col for typ, col in zip(DATASET_TYPES, DATASET_COLORS)}


TEST_METRICS = ["mean_absolute_error",  "accuracy"]

LIVEPLOT_GROUPS = {'Total Loss': ['loss', 'val_loss'],
                   'Classification Loss': ['clsout_loss', 'val_clsout_loss'],
                   'Acccuracy': ['clsout_accuracy', 'val_clsout_accuracy'],
                   'Mean Absolute Error': ['regout_mean_absolute_error',
                                           'val_regout_mean_absolute_error']}


def read_datasets(folder):
    """Read datasets from a folder."""
    filepaths = glob.glob(os.path.join(folder, "**", "*.csv*"), recursive=True)
    data = {}
    for filepath in tqdm(filepaths):
        for key in DATASET_TYPES:
            if key in filepath:
                assert key not in data
                data[key] = pd.read_csv(filepath, index_col="index")
    return data


class InputOutputTransformer:
    """Transform input features and target variables."""

    def __init__(self, x_features, y_reg_vars, y_cls_var, data):
        self.x_features = x_features
        self.y_reg_vars = y_reg_vars
        self.y_cls_var = y_cls_var
        self.x_scaler = MinMaxScaler().fit(data[self.x_features])
        self.y_reg_scaler = MinMaxScaler().fit(data[self.y_reg_vars])
        self.y_cls_encoder = LabelEncoder().fit(data[self.y_cls_var])

    def _do(self, what, which, data):
        """Transform or inverse transform the data."""
        if which == "X":
            transformer = self.x_scaler
            variables = self.x_features
        elif which == "Y_reg":
            transformer = self.y_reg_scaler
            variables = self.y_reg_vars
        elif which == "Y_cls":
            transformer = self.y_cls_encoder
            variables = self.y_cls_var
        else:
            raise ValueError(f"Unrecognized set of features `{which}`")

        if what == "transform":
            func = transformer.transform
        elif what == "inverse_transform":
            func = transformer.inverse_transform
        else:
            raise ValueError(
                f"`{what}` must be either `transform` or `inverse_transform`.")

        result = func(data[variables] if what == "transform" else data)
        if what == "inverse_transform":
            colnames = [variables] if isinstance(variables, str) else variables
            result = pd.DataFrame(result, columns=colnames)
        return result

    def transform(self, which, data):
        """Transform input features and target variables."""
        return self._do("transform", which, data)

    def inverse_transform(self, which, data):
        """Inverse transform input features and target variables."""
        return self._do("inverse_transform", which, data)

    def n_inputs(self):
        """Number of input features."""
        return len(self.x_features)

    def n_reg_outputs(self):
        """Number of regression outputs."""
        return len(self.y_reg_vars)

    def n_cls_outputs(self):
        """Number of classification outputs."""
        return len(self.y_cls_encoder.classes_)


class MultioutNN:
    """Multi-output neural network for regression and classification."""

    def __init__(self, transformer, n_hidden=4, n_neurons=64,
                 learning_rate=0.01, dropout=None, reduce_lr_patience=None,
                 l1=0.0, l2=0.0, early_stopping=10, activation="relu",
                 classification_weight=0.1,
                 verbose=True):
        assert isinstance(n_hidden, int) and n_hidden > 0
        assert isinstance(n_neurons, int) and n_neurons > 1
        assert learning_rate > 0
        assert (dropout is None) or (0 <= dropout < 1)
        assert (reduce_lr_patience is None) or (reduce_lr_patience >= 0)
        assert (l1 is None) or (l1 >= 0)
        assert (l2 is None) or (l2 >= 0)
        assert early_stopping >= 5
        assert activation in ["relu", "sigmoid", "tanh"]
        assert classification_weight >= 0

        self.verbose = verbose
        self.transformer = transformer
        self.learning_rate = learning_rate
        self.history = None

        self.callbacks = []
        if early_stopping is not None:
            assert isinstance(early_stopping, int) and early_stopping > 5
            self.callbacks.append(
                EarlyStopping(patience=early_stopping,
                              monitor='val_loss',
                              restore_best_weights=True))
        if self.verbose:
            self.callbacks.append(
                PlotLossesKeras(groups=LIVEPLOT_GROUPS, figsize=(10, 4)))
        if reduce_lr_patience is not None and reduce_lr_patience > 0:
            self.callbacks.append(
                ReduceLROnPlateau(monitor="val_loss", factor=0.5, min_lr=1e-6,
                                  patience=reduce_lr_patience))

        if l1 is not None and l1 > 0:
            if l2 is not None and l2 > 0:
                assert isinstance(l2, float) and l2 > 0
                regularizer = regularizers.l1_l2(l1=l1, l2=l2)
            else:
                assert isinstance(l1, float) and l1 > 0
                regularizer = regularizers.l1(l1)
        elif l2 is not None and l2 > 0:
            assert isinstance(l2, float) and l2 > 0
            regularizer = regularizers.l2(l2)
        else:
            regularizer = None

        n_inputs = transformer.n_inputs()
        n_regout = transformer.n_reg_outputs()
        n_clsout = transformer.n_cls_outputs()

        # loss_weights = {"regout": 1 + 0 * n_inputs, "clsout": 1.0 * 0.0}
        loss_weights = [2 * n_inputs,
                        classification_weight * np.float32(1.0 / np.log(2.0))]
        # multiplying reg loss to give each feature equal wait (and by 2 for
        # dummy regressor to mean of a scaled variable) to class log, which is
        # scaled - dummy classifier gives as a baseline -log(1/n_classes)

        inputs = Input(name='input', shape=(n_inputs,))
        hidden = None
        for i_hidden in range(n_hidden):
            hidden = Dense(
                n_neurons, activation=activation,
                kernel_regularizer=regularizer, name=f"hidden{i_hidden+1}"
                )(inputs if hidden is None else hidden)
            if dropout is not None and dropout > 0:
                hidden = Dropout(dropout)(hidden)
        out_reg = Dense(n_regout, activation='linear', name='regout')(hidden)
        out_cls = Dense(n_clsout, activation='softmax', name='clsout')(hidden)

        model = Model(inputs=inputs, outputs=[out_reg, out_cls], name="MNN")
        model.compile(
            loss=['mean_absolute_error', 'sparse_categorical_crossentropy'],
            optimizer=Adam(learning_rate=self.learning_rate),
            metrics=TEST_METRICS,
            loss_weights=loss_weights)

        self.model = model
        if self.verbose:
            self.model.summary()

    def fit(self, training_data, validation_data, epochs=100):
        """Fit the model to the training data."""
        x_train = self.transformer.transform("X", training_data)
        x_valid = self.transformer.transform("X", validation_data)
        y_train = [self.transformer.transform("Y_reg", training_data),
                   self.transformer.transform("Y_cls", training_data)]
        y_valid = [self.transformer.transform("Y_reg", validation_data),
                   self.transformer.transform("Y_cls", validation_data)]
        self.history = self.model.fit(
            x=x_train, y=y_train, epochs=epochs, batch_size=64,
            validation_data=[x_valid, y_valid], callbacks=self.callbacks,
            verbose=self.verbose)
        return self.history

    def predict(self, data, verbose=0):
        """Predict using the trained model."""
        y_reg_pred, y_cls_pred = self.model.predict(self.transformer.transform(
            "X", data), verbose=verbose)
        y_cls_pred = np.argmax(y_cls_pred, axis=1)
        y_reg_pred = self.transformer.inverse_transform("Y_reg", y_reg_pred)
        y_cls_pred = self.transformer.inverse_transform("Y_cls", y_cls_pred)
        return y_reg_pred, y_cls_pred

    def save(self, *args):
        """Save the model to a file."""
        outpath = os.path.join(*args)
        obj = copy.deepcopy(self)
        obj.callbacks = None
        joblib.dump(obj, outpath)
