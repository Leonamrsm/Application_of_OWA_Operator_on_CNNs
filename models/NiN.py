from keras.models import Sequential
from keras.layers import Dropout, Dense, Conv2D, MaxPooling2D, AvgPool2D, Flatten
from keras import regularizers
from keras.initializers import he_uniform

def NiN(input_shape = (32, 32, 3), classes = 10, weight_decay=0):
    """
    Implementation of the popular VGG13:

    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """
    
    model = Sequential()


    # part 1
    model.add(Conv2D(192,kernel_size=[5,5],padding='same', activation='relu', kernel_initializer = he_uniform(seed=0), kernel_regularizer=regularizers.L2(weight_decay), input_shape=input_shape, name='block1_conv1'))
    model.add(Conv2D(160,kernel_size=[1,1],padding='valid', activation='relu', kernel_initializer = he_uniform(seed=45), kernel_regularizer=regularizers.L2(weight_decay), name='block1_conv2'))
    model.add(Conv2D(96,kernel_size=[1,1],padding='valid', activation='relu', kernel_initializer = he_uniform(seed=858), kernel_regularizer=regularizers.L2(weight_decay), name='block1_conv3'))
    model.add(MaxPooling2D(pool_size=[3,3],strides=2,padding='same', name='max_pooling2d_1'))
    model.add(Dropout(0.5))

    # part 2
    model.add(Conv2D(192,kernel_size=[5,5],padding='same', activation='relu', kernel_initializer = he_uniform(seed=88), kernel_regularizer=regularizers.L2(weight_decay), name='block2_conv1'))
    model.add(Conv2D(192,kernel_size=[1,1],padding='valid', activation='relu', kernel_initializer = he_uniform(seed=485), kernel_regularizer=regularizers.L2(weight_decay), name='block2_conv2'))
    model.add(Conv2D(192,kernel_size=[1,1],padding='valid', activation='relu', kernel_initializer = he_uniform(seed=4965), kernel_regularizer=regularizers.L2(weight_decay), name='block2_conv3'))
    model.add(AvgPool2D(pool_size=[3,3], strides=2, padding='same', name='average_pooling2d_2'))
    model.add(Dropout(0.5))


    # part 3
    model.add(Conv2D(192,kernel_size=[5,5],padding='same', activation='relu', kernel_initializer = he_uniform(seed=0), kernel_regularizer=regularizers.L2(weight_decay), name='block3_conv1'))
    model.add(Conv2D(192,kernel_size=[1,1],padding='valid', activation='relu', kernel_initializer = he_uniform(seed=4965), kernel_regularizer=regularizers.L2(weight_decay), name='block3_conv2'))
    model.add(Conv2D(classes,kernel_size=[1,1],padding='valid', activation='relu', kernel_initializer = he_uniform(seed=4965), kernel_regularizer=regularizers.L2(weight_decay), name='block3_conv3'))


    model.add(AvgPool2D(pool_size=[8,8],padding='valid', name='average_pooling2d_3'))


    model.add(Flatten())
    model.add(Dense(classes, activation='softmax'))
    

  

    return model