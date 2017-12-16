# Building Segmentation Using Convolutional Neural Network (CNN)

Using publicly available high resolution dataset, a shallow CNN is trained to label buildings in an held out samples.

![Figure 1](/images/R1.jpg)

The picture is part of an example output of the classifier. The green parts are true positives, the red parts are false positives, the blue parts are false negatives and the rest are true negatives. On test set: 86.82% accuracy, 76.72% precision and 66.62% recall.


### Getting the data

For running the program yourself you will need some aerial images. You can download from [here](https://www.cs.toronto.edu/~vmnih/data/)

After you downloaded the images, place the aerial imagery in a directory named `Sentinel-2` under the `input` directory.
Please take a look at the config.py file to see which shapefiles belong to which satellite images.

## Data directory

I tried to follow Ali Eslami's great [blog post](http://arkitus.com/patterns-for-research-in-machine-learning/) about structuring your Machine Learning projects. Therefore I have a data directory which is split up into input, working and output directories. So my data directory looks like this:
```
/data
  /input
    /{satellite provider name}
    /Shapefiles
  /working
    /models
      /{model_id}
    /train_data
      /labels_images
      /patches
      /building_bitmaps
      /WGS84_images
  /output
    /{model_id}
    /tensorboard
```

All the metrics and hyperparameters of a model are stored in `/output/{model-id}` model weights are stored under `/working/models/{model-id}`. The model ID is a string consisting of the current timestamp, the dataset that is used and the neural net architecture. `/output/tensorboard` contains the logs for tensorboard. The last repository which might be of interest is `/working/train_data/labels_images` which contains the visualisations of the water polygons for a satellite images. These images will be created if you run the script with the `-v` tag. The remaining directories in `/working/train_data` are used as caches for the preproccesing of the data.

## Getting the source code

You can clone the repository using the command: `git clone https://github.com/kayodeolaleye/building_segmentation_count.git`

### Running it
`$ cd building_segmentation_count/src`

`$ mkdir data`

`$ python buildingNets.py --setup`

`$ python buildingNets.py -p`

`$ python buildingNets.py -a 'one_layer' -i -t -E -C -T -v -e`

## Acknowledgements
[WaterNet](https://github.com/treigerm/WaterNet) from [Treigerm](https://github.com/treigerm) helped a lot to get started on this project. 

[DeepOSM](https://github.com/trailbehind/DeepOSM) from [TrailBehind](https://github.com/trailbehind) 

Volodymyr Mnih's PhD thesis [Machine Learning for Aerial Image Labeling](https://www.cs.toronto.edu/~vmnih/docs/Mnih_Volodymyr_PhD_Thesis.pdf) 
