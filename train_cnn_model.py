import optuna
from sklearn.model_selection import train_test_split
from optimize_cnn import create_initial_dataset, create_dataset_b, create_data_set
import numpy as np
import matplotlib.pyplot as plt
import glob
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
tf.random.set_seed(123)
np.random.seed(123)

def plot_loss_metric(history):
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
    plt.savefig('loss_val_imgs/best_params_cnn.png')


paths = glob.glob('.\\data\\*')
dict_path_classes = {f"{idx}": path.split(
    '\\')[-1] for idx, path in enumerate(paths)}
BATCH_SIZE = 12
EPOCHS = 100


X, y = create_initial_dataset()
y_b = create_dataset_b()


study = optuna.create_study(study_name="optimizing_cnn_parameters_for_pretrain",
                                storage='sqlite:///db/optimizing_cnn_parameters_for_pretrain.db', direction='maximize', load_if_exists=True)
best_params = study.best_params 

model = tf.keras.models.Sequential(name="Pretraining_CNN")
model.add(tf.keras.layers.Input(shape=(128, 128,1), name="input_layer"))
model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=best_params['kernel_size'],
            padding='same', activation="relu", name="conv2D_1"))
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=best_params['kernel_size'],
            padding='same', activation="relu", name="conv2D_2"))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), name="max_pool_1"))
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=best_params['kernel_size'],
            padding='same', activation="relu", name="conv2D_3"))
model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=best_params['kernel_size'],
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
model.compile(optimizer=tf.optimizers.Adam(learning_rate=best_params['learning_rate']),  loss='sparse_categorical_crossentropy', metrics=['accuracy'])

X_train, X_test, y_train, y_test = train_test_split(
    X, y_b, random_state=123, test_size=0.1)


X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, random_state=123, test_size=0.2)

train_ds, valid_ds, test_ds = create_data_set()
early_stop = EarlyStopping(
    monitor="val_loss", patience=7, mode="auto", verbose=1)


history = model.fit(train_ds, validation_data=valid_ds,
                epochs=EPOCHS, callbacks=[early_stop], verbose=2)

plot_loss_metric(history)

model.save('cnn_model/best_model.h5')