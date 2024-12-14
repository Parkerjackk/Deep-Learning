from google.colab import drive
drive.mount('/content/gdrive')
%cd /content/gdrive/MyDrive/4c16-labs/code/lab-06/

# Function to save a model
def save_model_to_disk(model, filename_base):
    # save model and weights (don't change the filenames)
    model_json = model.to_json()
    with open(filename_base + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(f"{filename_base}.h5")
    print("Saved model to model.json and weights to model.h5")

# Download the dataset
!curl --create-dirs -o /home/tcd/data/medicalimaging-dataset.zip https://tcddeeplearning.blob.core.windows.net/deeplearning202324/medicalimaging-dataset.zip

!mkdir -p /home/tcd/data/medicalimaging/
!unzip -o /home/tcd/data/medicalimaging-dataset.zip -d /home/tcd/data/medicalimaging/

# Dataset is located in /home/tcd/data/medicalimaging/

import numpy as np
import matplotlib.pyplot as plt
import random

for _type in ['benign', 'malignant', 'normal']:
    X = np.load(f'/home/tcd/data/medicalimaging/dataset/{_type}/input.npy')
    y = np.load(f'/home/tcd/data/medicalimaging/dataset/{_type}/target.npy')
    randomExample = random.randint(0, X.shape[0] - 1)
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(X[randomExample])
    axs[0].title.set_text('Input')
    axs[1].imshow(y[randomExample])
    axs[1].title.set_text('Output')
    fig.suptitle(_type.upper())
    plt.subplots_adjust(top=1.1)
    plt.show()

import keras
import tensorflow as tf

X_class = []
y_class = []
X_seg = []
y_seg = []

for _type in ['benign', 'malignant', 'normal']:
    X = np.load(f'/home/tcd/data/medicalimaging/dataset/{_type}/input.npy')
    y = np.load(f'/home/tcd/data/medicalimaging/dataset/{_type}/target.npy')

    # Classification labels: 0 for benign, 1 for malignant, 2 for normal
    if _type == 'benign':
        y_class_label = np.zeros(y.shape[0])  # Benign 0
    elif _type == 'malignant':
        y_class_label = np.ones(y.shape[0])  # Malignant 1
    else:
        y_class_label = np.ones(y.shape[0]) * 2  # Normal 2

    # Appending data for class
    X_class.append(X)
    y_class.append(y_class_label)

    # Appending data for seg
    X_seg.append(X)
    y_seg.append(y)

# Combine data from all categories into single arrays
X_classification = np.concatenate(X_class, axis=0)  # Combine all image data for classification
y_classification = np.concatenate(y_class, axis=0)  # Combine all labels for classification

X_segmentation = np.concatenate(X_seg, axis=0)  # Combine all image data for segmentation
y_segmentation = np.concatenate(y_seg, axis=0)  # Combine all segmentation maps

from sklearn.model_selection import train_test_split
# Split data into training and validation sets for the classification task
X_train_class, X_val_class, y_train_class, y_val_class = train_test_split(
    X_classification, y_classification, test_size = 0.2, stratify = y_classification, random_state = 59) #stratify ensure an equal split of data

# Split data into training and validation sets for the segmentation task
X_train_seg, X_val_seg, y_train_seg, y_val_seg = train_test_split(
    X_segmentation, y_segmentation, test_size = 0.2, random_state = 59) #flatten to ensure that the seg data is in a suitavle shape for strat

from tensorflow.keras.utils import to_categorical
# Convert labels to one-hot encoding
y_train_class = to_categorical(y_train_class, num_classes=3)
y_val_class = to_categorical(y_val_class, num_classes=3)

from tensorflow.keras.preprocessing.image import ImageDataGenerator
# Data Augmentation layer, enhances the diversity of the training dataset
data_augmentation = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
train_gen_class = data_augmentation.flow(X_train_class, y_train_class, batch_size=32)
#val_gen_class = data_augmentation.flow(X_val_class, y_val_class, batch_size=32)

from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import AdamW, Adam
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers.schedules import CosineDecay
# Task 1: classification model (InceptionV3)
def class_model():
    # Loading InceptionV3 withouot the top layer
    base_model = MobileNetV2(input_shape=(128, 128, 3), include_top=False, weights='imagenet')
    base_model.trainable = False

    print(len(base_model.layers))

    # Defining the model structure
    inputs = Input(shape=(128, 128, 3))
    x = base_model(inputs)
    x = GlobalAveragePooling2D()(x) # Pool the features


    # Specific dense layers for this problem
    x = Dense(256, activation='relu', kernel_regularizer=l2(0.005))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.35)(x)
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.005))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)
    outputs = Dense(3, activation='softmax')(x)  # Output layer with 3 classes

    lr_scheduler = CosineDecay(
        initial_learning_rate=0.001,  # Lower the learning rate to reduce noise
        decay_steps=1000
)


    lr_schedule = ExponentialDecay(
        initial_learning_rate=0.001,
        decay_steps=1000,
        decay_rate=0.95,
        staircase=True
    )
    # Create and compile the model
    model = Model(inputs = inputs, outputs = outputs)
    opt = AdamW(learning_rate=lr_schedule)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()

    return modelfrom tensorflow.keras.models import Model
from tensorflow.keras.optimizers import AdamW, Adam
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers.schedules import CosineDecay
# Task 1: classification model (InceptionV3)
def class_model():
    # Loading InceptionV3 withouot the top layer
    base_model = MobileNetV2(input_shape=(128, 128, 3), include_top=False, weights='imagenet')
    base_model.trainable = False

    print(len(base_model.layers))

    # Defining the model structure
    inputs = Input(shape=(128, 128, 3))
    x = base_model(inputs)
    x = GlobalAveragePooling2D()(x) # Pool the features


    # Specific dense layers for this problem
    x = Dense(256, activation='relu', kernel_regularizer=l2(0.005))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.35)(x)
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.005))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)
    outputs = Dense(3, activation='softmax')(x)  # Output layer with 3 classes

    lr_scheduler = CosineDecay(
        initial_learning_rate=0.001,  # Lower the learning rate to reduce noise
        decay_steps=1000
)
    lr_schedule = ExponentialDecay(
        initial_learning_rate=0.001,
        decay_steps=1000,
        decay_rate=0.95,
        staircase=True
    )
    # Create and compile the model
    model = Model(inputs = inputs, outputs = outputs)
    opt = AdamW(learning_rate=lr_schedule)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()

    return model

from IPython.display import clear_output
%matplotlib inline
from matplotlib.ticker import MaxNLocator
class PlotLossAccuracy(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.acc = []
        self.losses = []
        self.val_losses = []
        self.val_acc = []
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):

        self.logs.append(logs)
        self.x.append(int(self.i))
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.acc.append(logs.get('accuracy'))
        self.val_acc.append(logs.get('val_accuracy'))

        self.i += 1

        clear_output(wait=True)
        plt.figure(figsize=(16, 6))
        plt.subplot(1,2,1)
        plt.plot(self.x, self.losses, label="train loss")
        plt.plot(self.x, self.val_losses, label="validation loss")
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.title('Model Loss')
        plt.legend()
        plt.subplot(1,2,2)
        plt.plot(self.x, self.acc, label="training accuracy")
        plt.plot(self.x, self.val_acc, label="validation accuracy")
        plt.legend()
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.title('Model Accuracy')
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.show();

num_epochs = 80

# Create an instance of our callback functions class, to plot our loss function and accuracy with each epoch.
pltCallBack = PlotLossAccuracy()

# Run the training.
classification_model.fit(X_train_class, y_train_class,
          batch_size=1024, epochs=num_epochs,
          validation_data=(X_val_class, y_val_class),
          callbacks=[pltCallBack])

classification_model = class_model()

if (classification_model.count_params()  < 5000000) :
  save_model_to_disk(classification_model, "classification_model")
else:
  print("Your model is unecessarily complex, scale down!")


from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Conv2DTranspose, LayerNormalization, Activation, Rescaling
# Task 2: segmentation model (UNet)
import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy

# Custom F1 score for segmentation
def f1_score_metric(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred > 0.5, tf.float32)

    tp = tf.reduce_sum(y_true * y_pred)  # True positives
    fp = tf.reduce_sum(y_pred * (1 - y_true))  # False positives
    fn = tf.reduce_sum((1 - y_pred) * y_true)  # False negatives

    precision = tp / (tp + fp + tf.keras.backend.epsilon())
    recall = tp / (tp + fn + tf.keras.backend.epsilon())

    f1_score = 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())
    return f1_score

from tensorflow.keras.losses import BinaryCrossentropy

def weighted_binary_crossentropy(y_true, y_pred):
    weight_map = y_true * 5 + (1 - y_true)  # Give more weight to foreground
    bce_loss = BinaryCrossentropy()(y_true, y_pred)
    return tf.reduce_mean(bce_loss * weight_map)


def dice_loss(y_true, y_pred):
    numerator = 2 * tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred) + tf.keras.backend.epsilon()
    return 1 - numerator / denominator

def seg_model():
    inputs = Input(shape=(128, 128, 3))
    x = Rescaling(1./255)(inputs)

    # Encoder
    c0 = Conv2D(8, (3, 3), padding='same')(x)
    c0 = LayerNormalization()(c0)
    c0 = Activation('relu')(c0)
    c0 = Conv2D(8, (3, 3), padding='same')(c0)
    c0 = LayerNormalization()(c0)
    c0 = Activation('relu')(c0)
    c0 = Dropout(0.3)(c0)
    p0 = MaxPooling2D((2, 2))(c0)

    c1 = Conv2D(16, (3, 3), padding='same')(p0)
    c1 = LayerNormalization()(c1)
    c1 = Activation('relu')(c1)
    c1 = Conv2D(16, (3, 3), padding='same')(c1)
    c1 = LayerNormalization()(c1)
    c1 = Activation('relu')(c1)
    c1 = Dropout(0.3)(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(32, (3, 3), padding='same')(p1)
    #c2 = BatchNormalization()(c2)
    c2 = Activation('relu')(c2)
    c2 = Conv2D(32, (3, 3), padding='same')(c2)
    #c2 = BatchNormalization()(c2)
    c2 = Activation('relu')(c2)
    c2 = Dropout(0.35)(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(64, (3, 3), padding='same')(p2)
    #c3 = BatchNormalization()(c3)
    c3 = Activation('relu')(c3)
    c3 = Conv2D(64, (3, 3), padding='same')(c3)
    #c3 = BatchNormalization()(c3)
    c3 = Activation('relu')(c3)
    c3 = Dropout(0.4)(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    # Bottleneck
    c4 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01))(p3)
    c4 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01))(c4)
    drop = Dropout(0.5)(c4)

    # Decoder
    u3 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(drop)
    u3 = Concatenate()([u3, c3])
    c5 = Conv2D(64, (3, 3), padding='same')(u3)
    #c5 = BatchNormalization()(c5)
    c5 = Activation('relu')(c5)
    c5 = Conv2D(64, (3, 3), padding='same')(c5)
    #c5 = BatchNormalization()(c5)
    c5 = Activation('relu')(c5)

    u2 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c5)
    u2 = Concatenate()([u2, c2])
    c6 = Conv2D(32, (3, 3), padding='same')(u2)
    #c6 = BatchNormalization()(c6)
    c6 = Activation('relu')(c6)
    c6 = Conv2D(32, (3, 3), padding='same')(c6)
    #c6 = BatchNormalization()(c6)
    c6 = Activation('relu')(c6)

    u1 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c6)
    u1 = Concatenate()([u1, c1])
    c7 = Conv2D(16, (3, 3), padding='same')(u1)
    c7 = LayerNormalization()(c7)
    c7 = Activation('relu')(c7)
    c7 = Conv2D(16, (3, 3), activation='relu', padding='same')(c7)
    c7 = LayerNormalization()(c7)
    c7 = Activation('relu')(c7)

    u0 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same', kernel_regularizer=l2(0.01))(c7)
    u0 = Concatenate()([u0, c0])
    c8 = Conv2D(8, (3, 3), padding='same')(u0)
    c8 = LayerNormalization()(c8)
    c8 = Activation('relu')(c8)
    c8 = Conv2D(8, (3, 3), padding='same')(c8)
    c8 = LayerNormalization()(c8)
    c8 = Activation('relu')(c8)


    outputs = Conv2D(1, (1, 1), activation='sigmoid')(u0)


    model = Model(inputs, outputs, name="segmentation_model")
    opt = AdamW(learning_rate=0.0005)
    model.compile(optimizer=opt, loss=dice_loss, metrics=['accuracy', f1_score_metric])
    model.summary()
    return model

from IPython.display import clear_output
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

class PlotSegmentationF1(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.f1_scores = []
        self.val_f1_scores = []

    def on_epoch_end(self, epoch, logs=None):
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.f1_scores.append(logs.get('f1_score_metric'))
        self.val_f1_scores.append(logs.get('val_f1_score_metric'))
        self.i += 1

        clear_output(wait=True)
        plt.figure(figsize=(16, 6))

        # Plot Loss
        plt.subplot(1, 2, 1)
        plt.plot(self.x, self.losses, label="train loss")
        plt.plot(self.x, self.val_losses, label="validation loss")
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.title('Model Loss')
        plt.legend()

        # Plot F1 Score
        plt.subplot(1, 2, 2)
        plt.plot(self.x, self.f1_scores, label="train F1 score")
        plt.plot(self.x, self.val_f1_scores, label="validation F1 score")
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.ylabel('F1 Score')
        plt.xlabel('Epoch')
        plt.title('Model F1 Score')
        plt.legend()

        plt.show()

# Create callback instance
plot_f1_callback = PlotSegmentationF1()

# Train the model
segmentation_model.fit(
    X_train_seg, y_train_seg,
    validation_data=(X_val_seg, y_val_seg),
    epochs=40,
    batch_size=124,
    callbacks=[plot_f1_callback]
)

segmentation_model = seg_model()

if (segmentation_model.count_params()  < 3000000) :
  save_model_to_disk(segmentation_model, "segmentation_model")
else:
  print("Your model is unecessarily complex, scale down!")
