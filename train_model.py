import numpy as np
import sys,os,shutil
import time

import network
from util import *
from keras.models import model_from_json
import keras.models


def train_semantic_segmentation(features,responses,fold_assignments,\
    savename,network_name='flex_unet',network_kwargs={},\
    save_directory='trained_models',verification_fold=None,\
    internal_window_radius=None,weighted=False,\
    nodata_value=-9999,epoch_batch_size=1,batch_size=10,\
    max_epochs=1000,n_noimprovement_repeats=5,
    save_continuously=False,verbose=True):
  """ Train a semantic segmentation network. 

    Arguments:
    features - 4d numpy array 
      Training features, ordered as sample,y,x,p, with p as 
      the feature dimension
    responses - 4d numpy array - 
      Training responses, ordered as sample,y,x,r with the r 
      dimension as a single-band categorical classification
    fold_assignments - 1d numpy array
      sample-specific n_folds assignments
    savename - str
      name to save trained model iterations and training history as
    network_name - str 
      name of the network to deploy

    Keyword Arguments:
    network_kwargs - dict
      Keyword arguments to pass on to the specific network deployed.
    save_directory - str
      Directory that the trained models should be stored in.
    verification_fold - int
      Which fold to use for model validation - if not specified, 
      no verification set is used.
    internal_window_radius - int
      The size of the internal window on which to score the model.
    weighted - bool
      Aflag to indicate whether or not to use response weighting.
    nodata_value - float
      Data value to ignore.
    epoch_batch_size - int 
      The number of epochs to train before evaluating/saving.
    batch_size - int 
      The number of samples to train on (keras batch size).
    max_epochs - int 
      The maximum number of epochs to train on.
    n_noimprovement_repeats - int 
      The amount of iterations to continue training with no
      performance imrpovement
    save_continuously - bool
      Boolean to indicate whether or not to save the model weights
      at every epoch.
    verbose - int
      An integer indication of verbosity level.  Possible values:
        0 - print nothing
        1 - print only training info
        2 - print everything

  Returns:
    A trained CNN model.
  """

  window_radius = rint(responses.shape[1]/2.)
  if (internal_window_radius is None): internal_window_radius = window_radius

  if (os.path.isdir(save_directory) == False): os.mkdir(save_directory)

  savename_base = os.path.join(save_directory,savename)

  if (verification_fold is not None):
    train_set = fold_assignments != verification_fold
    test_set = fold_assignments == verification_fold
  else:
    train_set = np.ones(len(fold_assignments)).astype(bool)


  # assign training and testing X
  train_X = features[train_set,...]
  if (verification_fold is not None):
    test_X = features[test_set,...]
  del features


  un_responses = np.unique(responses[np.logical_and(responses != nodata_value,np.isnan(responses) == False)])
  
  tY = responses[train_set,:,:].astype(int)
  if(internal_window_radius != window_radius):
     buffer = (window_radius-internal_window_radius)
     tY[:,:buffer,:,:] = nodata_value
     tY[:,-buffer:,:,:] = nodata_value
     tY[:,:,:buffer,:] = nodata_value
     tY[:,:,-buffer:,:] = nodata_value
    
  
  if (weighted):
    train_Y = np.ones((tY.shape[0],tY.shape[1],tY.shape[2],len(un_responses)+1)).astype(float)

    # assign_weights
    response_weights = np.zeros(len(un_responses))
    response_counts = np.zeros(len(un_responses))
    for n in range(0,len(un_responses)):
      response_counts[n] = np.sum(tY == un_responses[n])

    for n in range(0,len(un_responses)):
      response_weights[n] = np.sum(tY != nodata_value)/float(response_counts[n])
      train_Y[np.squeeze(tY == un_responses[n]),-1] = response_weights[n]
    train_Y[np.squeeze(tY == nodata_value),-1] = 0
  else:
    train_Y = np.ones((tY.shape[0],tY.shape[1],tY.shape[2],len(un_responses))).astype(float)

  # Assign response values as slice categoricals
  for n in range(0,len(un_responses)):
    train_Y[...,n] = np.squeeze((tY == un_responses[n]).astype(float))


  if (verification_fold is not None):
    tY = responses[test_set,:,:].astype(float)
    
    if (weighted):
      test_Y = np.ones((tY.shape[0],tY.shape[1],tY.shape[2],len(un_responses)+1)).astype(float)
      for n in range(0,len(un_responses)):
        test_Y[...,n] = np.squeeze((tY == un_responses[n]).astype(float))
      for n in range(0,len(un_responses)):
       if(tY.shape[0] != 1):
        test_Y[np.squeeze(tY == un_responses[n]),-1] = response_weights[n]
       else:
        test_Y[0,np.squeeze(tY == un_responses[n]),-1] = response_weights[n]
      if(tY.shape[0] != 1):
        test_Y[np.squeeze(tY == nodata_value),-1] = 0
      else:
        test_Y[0,np.squeeze(tY == nodata_value),-1] = 0
    else:
      test_Y = np.ones((tY.shape[0],tY.shape[1],tY.shape[2],len(un_responses))).astype(float)

    for n in range(0,len(un_responses)):
      test_Y[...,n] = np.squeeze((tY == un_responses[n]).astype(float))


  if(weighted):
    n_classes = train_Y.shape[-1]-1
  else:
    n_classes = train_Y.shape[-1]
  model = network.get_network(network_name,train_X.shape[1:],n_classes,network_kwargs)
  model.compile(loss=network.cropped_categorical_crossentropy(train_Y.shape[1],internal_window_radius*2,weighted=weighted),optimizer='adam')
  if(verbose == 2): model.summary()

  with open(savename_base + '.json',"w") as json_file:
    json_file.write(model.to_json())
  
  best_loss = 1e50
  last_best = 0
  best_ind = -1
  validation_loss_history = []
  training_loss_history = []
  training_time = []
  training_epochs = []

  start_time = time.time()
  best_model = keras.models.clone_model(model)
  for n in range(0,max_epochs):
    if (verification_fold is not None):
      output = model.fit(train_X,train_Y,validation_data=(test_X,test_Y),epochs=epoch_batch_size,batch_size=batch_size,verbose = verbose > 0,shuffle=False)

      lvl = output.history['val_loss'][0]
      validation_loss_history.append(lvl)
      training_loss_history.append(output.history['loss'][0])
      training_time.append(time.time()-start_time)
      training_epochs.append(n*epoch_batch_size)
      if (lvl < best_loss*.98):
        best_loss = lvl
        last_best = 0
        best_ind = n
        best_model.set_weights(model.get_weights())
      else:
        last_best += 1
      perm = np.random.permutation(train_X.shape[0])
      train_X = train_X[perm,...]
      train_Y = train_Y[perm,...]
    else:
      model.fit(train_X,train_Y,epochs=epoch_batch_size,batch_size=batch_size,verbose=verbose>0,shuffle=False)
   
    if (save_continuously):
      model.save_weights(savename_base + '_epoch_' + str(n*epoch_batch_size),overwrite=True)

    if (last_best >= n_noimprovement_repeats):
      break

  best_model.save_weights(savename_base + '_weights',overwrite=True)
  np.savez(savename_base + '.npz',\
           training_loss=training_loss_history,\
           validation_loss=validation_loss_history,\
           training_time=training_time,\
           training_epochs=training_epochs)
  return model


def load_trained_model(savename,window_radius,internal_window_radius=None,save_directory='trained_models',weighted=False,verbose=True):
  """ Load a pre-trained semantic segmentation network. 

    Arguments:
    savename - str
      name to save trained model iterations and training history as
    window_radius - int
      Determines the subset image size, which results as 2*window_radius.

    Keyword Arguments:
    save_directory - str
      Directory that the trained models should be stored in.
    internal_window_radius - int
      The size of the internal window on which to score the model.
    weighted - bool
      Aflag to indicate whether or not to use response weighting.
    verbose - bool
      A boolean indication of whether or not to print outputs.
      

  Returns:
    A trained CNN model.
  """

  if (internal_window_radius is None): internal_window_radius = window_radius

  savename_base = os.path.join(save_directory,savename)
  try:
    jf = open(savename_base + '.json','r')
    model = model_from_json(jf.read())
    jf.close()
  except:
    print('Could not load model file: ' + savename_base + '.json')
    return None

  try:
    model.load_weights(savename_base + '_weights')
  except:
    print('Could not load model weight file: ' + savename_base + '_weights')
    return None

  try:
    model.compile(loss=network.cropped_categorical_crossentropy(window_radius*2,internal_window_radius*2,weighted=weighted),optimizer='adam')
  except:
    print('Could not compile model with given window_radius, internal_window_radius, and weighted flag')
    return None

  if (verbose): model.summary()

  return model
    


