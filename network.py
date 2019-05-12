import h5py
import keras
from keras import backend as K
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape, Concatenate, Lambda
from keras.layers.convolutional import Conv2D, MaxPooling2D, Conv2DTranspose, ZeroPadding2D, Cropping2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.optimizers import SGD
from keras.regularizers import l2
from keras.utils import to_categorical
from keras.layers.merge import concatenate
import tensorflow as tf

from util import *


##### Loss functions #####
def cropped_categorical_crossentropy(outer_width,inner_width,weighted=False):
    """ Categorical cross-entropy with optional per-pixel weighting
        and edge trimming options.

    Arguments:
    outer_width - int
      The width of the input image.
    inner_width - int
      The width of the input image to use in the loss function

    Keyword Arguments:
    weighted - bool
      Tells whether the training y has weights as the last dimension
      to apply to the loss function.
    """

    def _cropped_cc(y_true, y_pred):
        if (outer_width != inner_width):
          buffer = rint((outer_width-inner_width) / 2)
          y_true = y_true[:, buffer:-buffer, buffer:-buffer, :]
          y_pred = y_pred[:, buffer:-buffer, buffer:-buffer, :]
        if (weighted):
          return K.categorical_crossentropy(y_true[...,:-1],y_pred) * y_true[...,-1]
        else:
          return K.categorical_crossentropy(y_true,y_pred)
    return _cropped_cc

def get_or_assign_kwarg(keyword,kwargs,default):
  """ Helper function to check and return dictionary values

  Arguments:
  keyword - str
    Key to search dictionary for.
  kwargs - dictionary
    Dictionary to search.
  default - any value
    Value to return if the dictionary does not contain keyword.

  Returns:
    Either extracted dictionary value or the default value.
  """
  if (keyword in kwargs):
    return kwargs[keyword]
  else:
    return default

##### Networks #####
def flex_unet(inshape, n_classes, kwargs):
    """ Construct a U-net style network with flexible shape

    Arguments:
    inshape - tuple/list
      Designates the input shape of an image to be passed to
      the network.
    n_classes - int
      The number of classes the network is meant to classify.
    kwargs - dict
      A dictionary of optional keyword arguments, which may contain
      extra keywords.  Values to use are:

      conv_depth - int/str
        If integer, a fixed number of convolution filters to use
        in the network.  If 'growth' tells the network to grow
        in depth to maintain a constant number of neurons.
      batch_norm - bool
        Whether or not to use batch normalization after each layer.

    Returns:
      A U-net style network keras network.
    """
    conv_depth = get_or_assign_kwarg('conv_depth',kwargs,16)
    batch_norm = get_or_assign_kwarg('batch_norm',kwargs,False)

    inlayer = keras.layers.Input(inshape)
    growth_flag = False
    if (conv_depth == 'growth'):
      growth_flag = True
      conv_depth = 8

    # get width 
    width = inshape[1]

    pool_list = []
    pre_pool_list = []
    b1 = Conv2D(conv_depth, (3, 3), activation='relu', padding='same')(inlayer)

    # encoding layers
    if (batch_norm): b1 = BatchNormalization()(b1)
    pre_pool_list.append(Conv2D(conv_depth, (3, 3), activation='relu', padding='same')(b1))
    pool_list.append(MaxPooling2D(pool_size=(2, 2))(pre_pool_list[-1]))
    if (batch_norm): pool_list.append(BatchNormalization()(pool_list[-1]))
    if(growth_flag): conv_depth=int(2*conv_depth)

    n_encode = 1
    while width > 8:
        b2 = Conv2D(conv_depth, (3, 3), activation='relu', padding='same')(pool_list[-1])
        if (batch_norm): b2 = BatchNormalization()(b2)
        pre_pool_list.append(Conv2D(conv_depth, (3, 3), activation='relu', padding='same')(b2))
        pool_list.append(MaxPooling2D(pool_size=(2, 2))(pre_pool_list[-1]))
        if (batch_norm): pool_list.append(BatchNormalization()(pool_list[-1]))
        n_encode += 1
        width = rint(width / 2.)
        if(growth_flag): conv_depth=int(2*conv_depth)

    # decoding layers
    last_layer = pool_list[-1]
    for n in range(0, n_encode):

        b2 = Conv2D(conv_depth, (3, 3), activation='relu', padding='same')(last_layer)
        if (batch_norm): b2 = BatchNormalization()(b2)
        b2 = Conv2D(conv_depth, (3, 3), activation='relu', padding='same')(b2)
        if (batch_norm): b2 = BatchNormalization()(b2)

        u1 = UpSampling2D(size=(2, 2))(b2)
        u1 = Conv2D(conv_depth, (3, 3), activation='relu', padding='same')(u1)
        if (batch_norm): u1 = BatchNormalization()(u1)
        last_layer = Concatenate()([pre_pool_list[n_encode-1-n], u1])
        if(growth_flag): conv_depth=int(conv_depth/2)

    e1 = Conv2D(conv_depth, (3, 3), activation='relu', padding='same')(last_layer)
    if (batch_norm): e1 = BatchNormalization()(e1)
    e2 = Conv2D(conv_depth, (3, 3), activation='relu', padding='same')(e1)
    if (batch_norm): e2 = BatchNormalization()(e2)

    output_layer = Conv2D(n_classes, (1, 1), activation='softmax', padding='same')(e2)
    model = keras.models.Model(input = inlayer , output=output_layer)
    return model


def get_network(net_name,inshape,n_classes,kwargs={}):
  """ Helper function to return the appropriate network.
  
  Arguments:
  net_name - str
    Name of the network to fetch.  Options are:
      flex_unet - a flexible, U-net style network.
  inshape - tuple/list
    Designates the input shape of an image to be passed to
    the network.
  n_classes - int
    The number of classes the network is meant to classify.
  kwargs - dict
    An optional dictionary of extra keywords for different networks.

  Returns:
    A Keras network in the designated style.
  """
  if (net_name == 'flex_unet'):
    return flex_unet(inshape,n_classes,kwargs)
  else:
    raise NotImplementedError('Unknown network name')

