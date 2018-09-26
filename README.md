# making it easier to find asset
demonstrating some possibilities using DNN and the asset store

## Idea 1: categorize an asset by an image
what if you could have your asset automaticly tagged and categorized by an image.

Starting with a mobilenet model that was trained on imagenet I retrained the last layer of the convolutional neural net on images I web scraped from your asset store using a technique called transferred learning.  This allowed me to leverage all the heavy lifting that when into recognizing real life objects and refine it to recognize the newonses that are in the 3D assets equivalents of those objects.
The result is a very fast and light weight 3d asset recognizer, that is lightweight to scale and quick to retrain.
Easily achieve 400+ images per sec per instance or lightweight enough to auto-scale in the cloud

this demo is in the **tagging** folder

**run `python runtraining.py` to train the model**

**run `python score.py` to visual inspect a collection of test images** 
- test images can be placed in the **ft_files/test** folder

![Alt text](https://raw.githubusercontent.com/zaront/tensorflow-assetstore/master/Capture.PNG "results")

*for this demo I used only the first 2 pages from 7 diffrent asset categories.  But the technique could be used to tag many more attributes.*

## Idea 2: find by similar image
what if you could search by an image and find assets that look visualy simmilar

The work here is incomplete but so far I've extracted vectors from inception model for each asset image. (in the **image_vectors** folder)  The next step would be to cluster them using nearest neighbor.

this demo is in the **vector** folder

this one is still work in progress but many thanks to the following as I piece this solution out:
https://github.com/Mujadded/knn-with-tensorflow/blob/master/k-nearest.py
https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/2_BasicModels/nearest_neighbor.py
https://github.com/spotify/annoy

*inspired by this:*
https://blog.griddynamics.com/create-image-similarity-function-with-tensorflow-for-retail/

## Idea 3: find by similar shape
what if assets you are using could be compaired to assets in the store and similar "swap-out" assets that just came into the store could be recommended automaticly as possible improved replacements.

I havn't gotten to this idea yet but was inspired that converting the mesh to voxel and then running through a similar process as the "find by similar image" might work

*inspired by this:*
https://people.csail.mit.edu/khosla/papers/cvpr2015_wu.pdf
