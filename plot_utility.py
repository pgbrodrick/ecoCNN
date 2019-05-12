
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def plot_training_data(features,responses,images_to_plot=3,feature_band=0,nodata_value=-9999):
  """ Tool to plot the training and response data data side by side.

  Arguments:
  features - 4d numpy array
    Array of data features, arranged as n,y,x,p, where n is the number of samples, y is the 
    data y dimension (2*window_size_radius), x is the data x dimension (2*window_size_radius), 
    and p is the number of features.
  responses - 4d numpy array
    Array of of data responses, arranged as n,y,x,p, where n is the number of samples, y is the 
    data y dimension (2*window_size_radius), x is the data x dimension (2*window_size_radius), 
    and p is the response dimension (always 1).
  """
  features = features.copy()
  responses = responses.copy()
  features[features == nodata_value] = np.nan
  responses[responses == nodata_value] = np.nan

  feat_nan = np.squeeze(np.isnan(features[:,:,:,0]))
  if (feature_band != 'rgb'):
    feat_min = np.nanmin(features[:,:,:,feature_band])
    feat_max = np.nanmax(features[:,:,:,feature_band])
    #features[:,:,:,feature_band] = (features[:,:,:,feature_band] - feat_min)/(feat_max-feat_min)
  else:
    for n in range(0,3):
      feat_min = np.nanmin(features[:,:,:,n])
      feat_max = np.nanmax(features[:,:,:,n])
      features = (features[:,:,:,n] - feat_min)/(feat_max-feat_min)
  features[feat_nan,:] = np.nan

  
  fig = plt.figure(figsize=(4,images_to_plot*2))
  gs1 = gridspec.GridSpec(images_to_plot, 2)
  for n in range(0,images_to_plot):
      ax = plt.subplot(gs1[n,0])
      if (feature_band == 'rgb'):
        plt.imshow(np.squeeze(features[n,:,:,:]))
      else:
        plt.imshow(features[n,:,:,feature_band],vmin=feat_min,vmax=feat_max)
      plt.xticks([])
      plt.yticks([])
      if (n == 0):
          plt.title('Feature')
  
      ax = plt.subplot(gs1[n,1])
      plt.imshow(responses[n,:,:,0])
      plt.xticks([])
      plt.yticks([])
      if (n==0):
          plt.title('Response')
