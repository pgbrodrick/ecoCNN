import numpy as np
import numpy.matlib
from scipy.interpolate import griddata


VALUE_NO_DATA = -9999


def scale_vector(dat, flag, nodata_value=VALUE_NO_DATA):
    """ Scale a 1-d numpy array in a specified maner, ignoring nodata values.
    Arguments:
    dat - input vector to be scaled
    flag - an indicator of the chosen scaling option

    Keyword Aguments:
    ndoata_value - value to be ignored, None of no nodata_value specified

    Return:
    The offset and gain scaling factors, in a two-value list form.
    """
    if (flag is None):
        return [0, 1]
    elif (flag == 'mean'):
      if nodata_value is None:
        return [np.mean(dat), 1]
      else:
        return [np.mean(dat[dat != nodata_value]), 1]
    elif (flag == 'mean_std'):
      if nodata_value is None:
        return [np.mean(dat), np.std(dat)]
      else:
        return [np.mean(dat[dat != nodata_value]), np.std(dat[dat != nodata_value])]
    elif (flag == 'minmax'):
      if nodata_value is None:
        return [np.min(dat), np.max(dat)]
      else:
        return [np.min(dat[dat != nodata_value]), np.max(dat[dat != nodata_value])]
    else:
        return [0, 1]


def rint(num):
    """ Round a number to it's nearest integer value, and cast it as an integer. """
    return int(round(num))


def scale_image(image, flag,nodata_value=VALUE_NO_DATA):
    """ Scale an image based on preset flag.
    Arguments:
    image - 3d array with assumed dimensions y,x,band 
    flag - scaling flag to use (None if no scaling)

    Return:
    An image matching the input image dimension with scaling applied to it.
    """
    if flag is None:
        return image
    elif (flag == 'mean_std'):
        return scale_image_mean_std(image,nodata_value)
    elif (flag == 'mean'):
        return scale_image_mean(image,nodata_value)
    elif (flag == 'minmax'):
        return scale_image_minmax(image,nodata_value)
    else:
        raise NotImplementedError('Unknown scaling flag')


def scale_image_mean_std(image, nodata_value=VALUE_NO_DATA):
    """ Mean center and standard-deviation normalize an image.
    Arguments:
    image - 3d array with assumed dimensions y,x,band 

    Keyword Aguments:
    ndoata_value - value to be ignored, None of no nodata speified

    Return:
    Image with per-band mean centering and std normalization applied
    """
    nodata_mask = np.logical_not(np.all(image == nodata_value,axis=2))
    for b in range(0,image.shape[2]):
      image[nodata_mask,b] = image[nodata_mask,b] - np.mean(image[nodata_mask,b])
      std = np.std(image[nodata_mask,b])
      if (std != 0):
        image[nodata_mask,b] = image[nodata_mask,b] / std
    return image


def scale_image_mean(image, nodata_value=VALUE_NO_DATA):
    """ Mean center an image.
    Arguments:
    image - 3d array with assumed dimensions y,x,band 

    Keyword Aguments:
    ndoata_value - value to be ignored, None of no nodata speified

    Return:
    Image with per-band mean centering applied
    """
    nodata_mask = np.logical_not(np.all(image == nodata_value,axis=2))
    for b in range(0,image.shape[2]):
      image[nodata_mask,b] = image[nodata_mask,b] - np.mean(image[nodata_mask,b])
    return image

def scale_image_minmax(image, nodata_value=VALUE_NO_DATA):
    """ Scale image based on local mins and maxes.
    Arguments:
    image - 3d array with assumed dimensions y,x,band 

    Keyword Aguments:
    ndoata_value - value to be ignored, None of no nodata speified

    Return:
    Image with per-band minmax scaling applied
    """
    nodata_mask = np.logical_not(np.all(image == nodata_value,axis=2))
    for b in range(0,image.shape[2]):
      mm = scale_vector(image[...,b], 'minmax', nodata_value=nodata_value)
      image[nodata_mask,b] = (image[nodata_mask,b] - mm[0])/float(mm[1])

    return image




def fill_nearest_neighbor(image, nodata=VALUE_NO_DATA):
    """ Fill in missing values in an image using a nearest neighbor approach.
    Arguments:
    image - 3d array with assumed dimensions y,x,band 

    Keyword Aguments:
    ndoata_value - value to be ignored, None of no nodata speified

    Return:
    Image with nodata_value values filled in with their nearest neighbors.
    """
    nodata_sum = np.sum(np.all(image == VALUE_NO_DATA,axis=2))
    if (nodata_sum > 0 and nodata_sum < image.size):
        ims = image.shape
        x_arr = np.matlib.repmat(np.arange(0, ims[1]).reshape(1, ims[1]), ims[0], 1).flatten().astype(float)
        y_arr = np.matlib.repmat(np.arange(0, ims[0]).reshape(ims[0], 1), 1, ims[1]).flatten().astype(float)

        if (len(ims) == 3):
            image = image.reshape((ims[0]*ims[1], ims[2]))
            image_nodata = np.any(image == nodata, axis=-1)
        else:
            image = image.flatten()
            image_nodata = image == nodata

        image[image_nodata] = griddata(np.transpose(np.vstack([x_arr[image_nodata], y_arr[image_nodata]])),
                            image[image_nodata], (x_arr[image_nodata], y_arr[image_nodata]), method='nearest')
        return np.reshape(image, ims)
    return image
