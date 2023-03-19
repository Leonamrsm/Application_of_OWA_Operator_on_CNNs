
from tensorflow.keras.datasets import cifar10, cifar100
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers
from layers.randomPadding2D import RandomPadding2D


import matplotlib.pyplot as plt
import tensorflow as tf 
import math



def prepare(ds, shuffle=False, augment=False, batch_size=32, AUTOTUNE = tf.data.AUTOTUNE, IMG_SIZE=(32,32)):
  
  resize_and_rescale = tf.keras.Sequential([
  layers.Resizing(IMG_SIZE[0], IMG_SIZE[1]),
  layers.Rescaling(1./255)
  ])


  data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    RandomPadding2D(mode="REFLECT", padding = math.floor(IMG_SIZE[0]*0.125)),
    layers.RandomCrop(IMG_SIZE[0], IMG_SIZE[1])
  ])

  # Resize and rescale all datasets.
  ds = ds.map(lambda x, y: (resize_and_rescale(x), y), 
              num_parallel_calls=AUTOTUNE)

  if shuffle:
    ds = ds.shuffle(buffer_size=1000)

  # Batch all datasets.
  ds = ds.batch(batch_size)

  # Use data augmentation only on the training set.
  if augment:
    ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y), 
                num_parallel_calls=AUTOTUNE)

  # Use buffered prefetching on all datasets.
  return ds.prefetch(buffer_size=AUTOTUNE)


# load train and test dataset
def load_dataset(dataset='cifar10'):
  # load dataset
	if dataset == 'cifar10':
		(trainX, trainY), (testX, testY) = cifar10.load_data()
	else:
		(trainX, trainY), (testX, testY) = cifar100.load_data()
	# one hot encode target values
	trainY = to_categorical(trainY)
	testY = to_categorical(testY)
	return trainX, trainY, testX, testY

def prep_pixels(x_train, x_test):

	#convert integers to float; normalise and center the mean
	x_train=x_train.astype("float32")  
	x_test=x_test.astype("float32")

	train_norm = x_train / 255.0
	test_norm = x_test / 255.0

	return train_norm, test_norm

# plot diagnostic learning curves
def summarize_diagnostics(history):
    history_dict = history.history
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']

    epochs = range(1, len(loss_values) + 1)

    plt.figure(figsize=(14, 4))

    plt.subplot(1,2,1)
    plt.plot(epochs, loss_values, 'bo', label='Training Loss')
    plt.plot(epochs, val_loss_values, 'b', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    acc = history_dict['accuracy']
    val_acc = history_dict['val_accuracy']

    epochs = range(1, len(loss_values) + 1)

    plt.subplot(1,2,2)
    plt.plot(epochs, acc, 'bo', label='Training Accuracy', c='orange')
    plt.plot(epochs, val_acc, 'b', label='Validation Accuracy', c='orange')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

# learning rate schedule
def lr_scheduler(epoch):
  """
  Implementation of learning rate scheduler

  initial learning rate = 0.001
  lr_drop = 20
  """

  learning_rate = 0.001
  lr_drop = 5
  if epoch < 20:
    return learning_rate
  elif epoch < 30:
    return learning_rate * 0.5
  else:
    return learning_rate * (0.5 ** ((epoch -20) // lr_drop))


def onecycle_scheduler(epoch, num_epoch=30):
  
  end_percentage = 0.1
  scale_percentage = 0.5
  maximum_lr=0.001
  minimun_lr = maximum_lr/10

  if epoch <= (1-end_percentage) * num_epoch * 0.5:
    return minimun_lr + ((maximum_lr - minimun_lr)/ ((1-end_percentage) * num_epoch * 0.5)) * epoch
  elif epoch < (1-end_percentage) * num_epoch:
    return 2* maximum_lr - minimun_lr + ((minimun_lr - maximum_lr)/((1-end_percentage) * num_epoch * 0.5)) * epoch
  else:
    return minimun_lr*(1-scale_percentage + scale_percentage/end_percentage) - scale_percentage*minimun_lr/(num_epoch * end_percentage) * epoch