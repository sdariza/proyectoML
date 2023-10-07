import optuna
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import glob
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
tf.random.set_seed(123)
np.random.seed(123)

paths = glob.glob('.\\data\\*')
dict_path_classes = {f"{idx}": path.split(
    '\\')[-1] for idx, path in enumerate(paths)}
BATCH_SIZE = 12
EPOCHS = 100


def create_initial_dataset():
    """_summary_
    This function create 2 numpy arrays to generate initial dataset
    Returns:
        numpy.array: desing metrix
        numpy.array: observation vector
    """
    X = []
    y = []
    for key, value in dict_path_classes.items():
        for path in glob.glob(f'.\\data\\{value}\\*.jpg'):
            img = plt.imread(path)
            X.append(img.reshape(128,128,1))
            y.append(key)
    return np.array(X), np.array(y)


X, y = create_initial_dataset()


def create_dataset_b():
    """_summary_
    Balancear

    Returns:
        _type_: _description_
    """
    y_b = []
    for label in y:
        if label in ["0", "1", "3"]:
            y_b.append(1)
        else:
            y_b.append(0)
    return np.array(y_b)


y_b = create_dataset_b()


def create_cnn_model(trial):
    """_summary_

    Returns:
        _type_: _description_
    """
    kernel_sizes = [(2, 2), (3, 3)]
    kz_selected = trial.suggest_categorical("kernel_size", kernel_sizes)
    model = tf.keras.models.Sequential(name="Pretraining_CNN")
    model.add(tf.keras.layers.Input(shape=(128, 128,1), name="input_layer"))
    model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=kz_selected,
              padding='same', activation="relu", name="conv2D_1"))
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=kz_selected,
              padding='same', activation="relu", name="conv2D_2"))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), name="max_pool_1"))
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=kz_selected,
              padding='same', activation="relu", name="conv2D_3"))
    model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=kz_selected,
              padding='same', activation="relu", name="conv2D_4"))
    model.add(tf.keras.layers.Flatten(name="flatten"))
    model.add(tf.keras.layers.Dense(256, activation='relu', name="MLP_1"))
    model.add(tf.keras.layers.Dropout(0.2, name="drop_out_1"))
    model.add(tf.keras.layers.Dense(128, activation='relu', name="MLP_2"))
    model.add(tf.keras.layers.Dropout(0.2, name="drop_out_2"))
    model.add(tf.keras.layers.Dense(64, activation='relu', name="MLP_3"))
    model.add(tf.keras.layers.Dropout(0.2, name="drop_out_3"))
    model.add(tf.keras.layers.Dense(
        2, activation='softmax', name="output_classification"))
    model.summary()
    return model


X_train, X_test, y_train, y_test = train_test_split(
    X, y_b, random_state=123, test_size=0.1)


X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, random_state=123, test_size=0.2)


def create_data_set():
    """_summary_

    Returns:
        _type_: _description_
    """
    train_ds = tf.data.Dataset.from_tensor_slices(tensors=(X_train, y_train))
    train_ds = train_ds.shuffle(buffer_size=len(
        y_train), seed=123).batch(batch_size=BATCH_SIZE)

    valid_ds = tf.data.Dataset.from_tensor_slices(tensors=(X_val, y_val))
    valid_ds = valid_ds.shuffle(buffer_size=len(
        y_val), seed=123).batch(batch_size=BATCH_SIZE)

    test_ds = tf.data.Dataset.from_tensor_slices(tensors=(X_test, y_test))
    test_ds = test_ds.shuffle(buffer_size=len(
        y_test), seed=123).batch(batch_size=BATCH_SIZE)
    return train_ds, valid_ds, test_ds


def create_optimizer(trial):
    kwargs = {}
    kwargs["learning_rate"] = trial.suggest_float(
        "learning_rate", 1e-4, 1e-2, log=True)
    optimizer = getattr(tf.optimizers, 'Adam')(**kwargs)
    return optimizer


def plot_loss_metric(history, params, _trial_id):
    """_summary_

    Args:
        history (_type_): _description_
        params (_type_): _description_
        _trial_id (_type_): _description_
    """
    train_loss = history.history['loss']
    train_acc = history.history['accuracy']
    val_loss = history.history['val_loss']
    val_acc = history.history['val_accuracy']

    epochs = range(1, len(train_loss)+1)

    plt.figure(figsize=(25,15))
    kz, lr = params['kernel_size'], params['learning_rate']
    plt.suptitle(f'tria_id:{_trial_id} | kz:{kz} | lr:{lr}', fontsize=20, y=0.98)

    plt.subplot(1,2,1)
    plt.plot(epochs, train_loss, label='Loss on Training')
    plt.plot(epochs, val_loss, label='Loss on Validation')
    plt.xlabel('Epoch', fontsize=20)
    plt.ylabel('loss', fontsize=20)
    plt.title('Training and Validation loss', fontsize=20)
    plt.legend(fontsize=15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.subplot(1,2,2)
    plt.plot(epochs, train_acc, label='Accuracy on Training')
    plt.plot(epochs, val_acc, label='Accuracy on Validation')
    plt.xlabel('Epoch', fontsize=20)
    plt.ylabel('accuracy', fontsize=20)
    plt.title('Training and Validation loss', fontsize=20)


    plt.legend(fontsize=15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)



    plt.tight_layout()
    plt.savefig(f'loss_val_imgs/{_trial_id}_cnn.png')



def objective(trial):
    train_ds, valid_ds, test_ds = create_data_set()
    cnn_model = create_cnn_model(trial)
    optimizer = create_optimizer(trial)
    early_stop = EarlyStopping(
        monitor="val_loss", patience=7, mode="auto", verbose=1)
    cnn_model.compile(optimizer=optimizer,  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = cnn_model.fit(train_ds, validation_data=valid_ds,
                  epochs=EPOCHS, callbacks=[early_stop], verbose=2)
    plot_loss_metric(history, trial.__dict__['_cached_frozen_trial'].params, trial.__dict__['_trial_id'])
    acc = cnn_model.evaluate(test_ds)[1]
    if np.isnan(acc):
        return -np.inf
    return acc



if __name__ == "__main__":
    study = optuna.create_study(study_name="optimizing_cnn_parameters_for_pretrain",
                                storage='sqlite:///db/optimizing_cnn_parameters_for_pretrain.db', direction='maximize', load_if_exists=True)
    study.optimize(objective, n_trials=1)
