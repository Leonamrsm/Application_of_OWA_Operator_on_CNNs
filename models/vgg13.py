from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Dense, Conv2D, MaxPooling2D, BatchNormalization, Flatten, Activation
from tensorflow.keras import regularizers
from tensorflow.keras.initializers import he_uniform

def VGG13(input_shape = (32, 32, 3), classes = 10, weight_decay=0, include_top=True):
    """
    Implementation of the popular VGG13:

    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """
    
    model = Sequential()

    # conv_layers 
    # part 1
    model.add(Conv2D(64,kernel_size=[3,3],padding='same', kernel_initializer = he_uniform(seed=0), kernel_regularizer=regularizers.L2(weight_decay), input_shape=input_shape, name='block1_conv1'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    # model.add(Dropout(0.3))

    model.add(Conv2D(64,kernel_size=[3,3],padding='same', kernel_initializer = he_uniform(seed=10), kernel_regularizer=regularizers.L2(weight_decay), name='block1_conv2'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    # model.add(Dropout(0.3))

    model.add(MaxPooling2D(pool_size=[2,2],strides=2,padding='same'))


    # part 2
    model.add(Conv2D(128,kernel_size=[3,3],padding='same', kernel_initializer = he_uniform(seed=100), kernel_regularizer=regularizers.L2(weight_decay), name='block2_conv1'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    # model.add(Dropout(0.3))

    model.add(Conv2D(128,kernel_size=[3,3],padding='same', kernel_initializer = he_uniform(seed=1000), kernel_regularizer=regularizers.L2(weight_decay), name='block2_conv2'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    # model.add(Dropout(0.3))

    model.add(MaxPooling2D(pool_size=[2,2],strides=2,padding='same'))
    

    # part 3
    model.add(Conv2D(256,kernel_size=[3,3],padding='same', kernel_initializer = he_uniform(seed=54), kernel_regularizer=regularizers.L2(weight_decay), name='block3_conv1'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    # model.add(Dropout(0.4))

    model.add(Conv2D(256,kernel_size=[3,3],padding='same', kernel_initializer = he_uniform(seed=5454), kernel_regularizer=regularizers.L2(weight_decay), name='block3_conv2'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    # model.add(Dropout(0.4))

    model.add(MaxPooling2D(pool_size=[2,2],strides=2,padding='same'))


    # part 4
    model.add(Conv2D(512,kernel_size=[3,3],padding='same', kernel_initializer = he_uniform(seed=45545), kernel_regularizer=regularizers.L2(weight_decay), name='block4_conv1'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    # model.add(Dropout(0.4))

    model.add(Conv2D(512,kernel_size=[3,3],padding='same', kernel_initializer = he_uniform(seed=5454), kernel_regularizer=regularizers.L2(weight_decay), name='block4_conv2'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    # model.add(Dropout(0.4))

    model.add(MaxPooling2D(pool_size=[2,2],strides=2,padding='same'))


    # part 5
    model.add(Conv2D(512,kernel_size=[3,3],padding='same', kernel_initializer = he_uniform(seed=212), kernel_regularizer=regularizers.L2(weight_decay), name='block5_conv1'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    # model.add(Dropout(0.5))

    model.add(Conv2D(512,kernel_size=[3,3],padding='same', kernel_initializer = he_uniform(seed=454), kernel_regularizer=regularizers.L2(weight_decay), name='block5_conv2'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    # model.add(Dropout(0.5))
    
    model.add(MaxPooling2D(pool_size=[2,2],strides=2,padding='same'))

    if include_top:

      # fc_layers =[
      model.add(Flatten())
      model.add(Dense(512, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=regularizers.L2(weight_decay)))
      model.add(Activation('relu'))
      model.add(Dropout(0.5))
      model.add(Dense(classes))
    

  

    return model