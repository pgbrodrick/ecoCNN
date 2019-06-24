# ecoCNN
A CNN for ecologists using remotely sensed imagery, with a working inpput/output pipeline to construct and apply the model.  This repo goes with the manuscript [Uncovering ecological patterns with convolutional neural networks](https://www.sciencedirect.com/science/article/pii/S0169534719300862?via%3Dihub), which we encourage you to check out (and cite if you use this in academic work).  Our intent is that the combination of the manuscript, its SI, and this code should walk a new user through all necessary steps in order to generate training data, build a CNN, and deploy that model to a series of landscapes.  We **highly recommend** that the manuscript and SI be read by inexperienced users before trying to work too much with the code.  This repository was designed for remotely sensed imagery, and particularly imagery that covers large areas, rather than individual image scenes.  However, it can also  also work with large individual images.

This code base was intentionally written in a fairly linear manner for scientists to be able to read it easily.  This naturally means that a good bit of generality was sacrificed.  If you are interested in a more complete package that is flexible and facilitates easier reconfiguration of the CNN architecture for your needs, we have one that is out in [alpha](https://pgbrodrick.github.io/rsCNN/), and is being actively improved.

We **highly recommend** that the manuscript, and critically the companion SI, be read prior to using this code, particularly for inexperienced users.  Don't know where to start?  Try the [CNN_Tutorial](https://github.com/pgbrodrick/ecoCNN/blob/master/CNN_Tutorial.ipynb) jupyter notebook, which gives a gentle walkthrough. 

Most likely, additional features will be added to our package code (see info above) base rather than to this version - but if you don't see a particular feature you're interested in, feel free to either submit a pull request or contact me, and if it makes sense I can implement it here, or point you to somewhere it might exist.

Cheers,

[Phil Brodrick](https://www.philbrodrick.com)


## Setup

We use a keras-style model with tensor flow.  You'll need some external packages to get going, including:

numpy<br />
gdal<br />
rasterio<br />
fiona<br />
tensorflow<br />
keras<br />
matplotlib<br />

To check out the tutorial, you'll also need jupyter.
