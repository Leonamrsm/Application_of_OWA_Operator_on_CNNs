from keras.models import Sequential
from keras.layers import Dropout, Dense, Conv2D, MaxPooling2D, BatchNormalization, Flatten, Activation
from keras import regularizers
from keras.initializers import he_uniform

def AlexNet(input_shape = (224,224,3), classes = 10, weight_decay=0):
    """
    Implementation of the popular AlexNet:

    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """
    
    #Instantiation
    model = Sequential()

    #1st Convolutional Layer
    model.add(Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=input_shape, kernel_initializer = he_uniform(seed=0), name='conv2d_1'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))

    #2nd Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(5, 5), strides=(1,1), activation='relu', padding='same', kernel_initializer = he_uniform(seed=10), name='conv2d_2'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))

    #3rd Convolutional Layer
    model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same', kernel_initializer = he_uniform(seed=100), name='conv2d_3'))
    model.add(BatchNormalization())

    #4th Convolutional Layer
    model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same', kernel_initializer = he_uniform(seed=1000), name='conv2d_4'))
    model.add(BatchNormalization())

    #5th Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same', kernel_initializer = he_uniform(seed=10000), name='conv2d_5'))
    model.add(BatchNormalization())
    
    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))

    # 1st Fully Connected Layer
    model.add(Flatten())
    model.add(Dense(4096, activation='relu', kernel_initializer = he_uniform(seed=5)))
    model.add(Dropout(0.5))

    #2nd Fully Connected Layer
    model.add(Dense(4096, activation='relu', kernel_initializer = he_uniform(seed=50)))
    model.add(Dropout(0.5))

    #3rd Fully Connected Layer
    model.add(Dense(classes, kernel_initializer = he_uniform(seed=500)))

    return model