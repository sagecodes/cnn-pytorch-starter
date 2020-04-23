# Pytorch Convolutional Neural Network Starter Template

## Summary

After building out many Convolutional Neural Networks (CNNs) for several
research projects I decided to make a template to make getting started easy
while still having access to all the functions for per project customization
if needed.   

This template tackles most of the formats I have needed to load data and models
so far (See next sections). If it doesn't cover your use case the built functions
should provide a good starting point.

This template also contains a bare bones flask app that returns json response
with the pytorch models predicted class.

## Loading Data

Please visit the [Example Jupyter Notebook](example.ipynb) for examples of
how to use the data loading functions. 
Or read next section on how to use data with [train.py](train.py)

pre-built functions exsist for loading image data in two different ways:

See the functions in [helpers/data_loader.py](helpers/data_loader.py)

##### Directory (Images in folders seperated by class)

Directory data loader expects a root directory path containing sub folders for 
each class. 

Example:

```
root/dog/xxx.png
root/cat/xxx.png
root/panda/xxx.png
```

if this root directory exsists in the same directory as this template:

use the path: `root/`

##### CSV 

The CSV data loader expects a root directory and CSV file 
containing two columns:

1. relative path to image from root directory

2. label (class) of image ( Cat, Dog, Panda, etc..) 

Example:

images within root directory

```
root/dog_xxx.png
root/cat_xxx.png
root/panda_xxx.png
```

labels.csv: 

| FilePath  |  Label |
|---|---|
| cat_xxx.png |  cat |
| dog_xxx.png  | dog  |
| panda_xxx.png | Panda   |

if this root directory exsists in the same directory as this template:

use the path: `root/`

##### Default transforms

By default the only transforms applied to images is resize, toTensor,
and normalization based on imagenet data set 

 See the image_transforms function 
 in [helpers/data_loader.py](helpers/data_loader.py) to make changes

```
img_transforms = transforms.Compose(
        [transforms.Resize(size=(img_size,img_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225] )])
```

## Training a Model

Please visit the [Example Jupyter Notebook](example.ipynb) for examples of
how to use the training functions.

*note:* Currently train.py is set to use a pre-trained resnet50, VGG, or a
model programmed from scratch. reference the [models folder](/models) for how
a model is built. It is fairly straight foward to add more pre-built models. 
Or experiement with your own by writing it in 
[models folder](/models/scratch.py) 

[train.py](train.py) is an option to train directly from your terminal by
passing in arguments from the command line. read instructions below:

### Using train.py:

[train.py](train.py) can be run from terminal and passed commands arguments

##### Example: CSV Labels 

```
python train.py --verbose=True --device=cuda --num_classes=3 --n_epochs=30 
--learn_rate=0.001 --save_path=trained_models/test_train_tmp
--csv_labels=../datasets/animals/labels.csv --img_size=244 --batch_size=32 
--num_workers=0 --data_dir=../datasets/animals/
```

CSV Validation data 
expected validation dataset is in same root dir

```
python train.py --verbose=True --device=cuda --num_classes=3 --n_epochs=30 
--learn_rate=0.001 --save_path=trained_models/test_train_tmp
--csv_labels=../datasets/animals/labels.csv --img_size=244 --batch_size=32 
--num_workers=0 --data_dir=../datasets/animals/ --val_data=../datasets/test_animals/test_labels.csv
```

##### Example Directory Labels:

```
python train.py --verbose=True --device=cuda --num_classes=3 --n_epochs=30 
--learn_rate=0.001 --save_path=trained_models/test_train_tmp --img_size=244 
--batch_size=32 --num_workers=0 --data_dir=../datasets/animals/
```

directory validation
```
python train.py --device=cuda --num_classes=133 --n_epochs=30 
--learn_rate=0.001 --save_path=trained_models/test_train_tmp 
--img_size=244 --batch_size=32 --num_workers=0 
--data_dir=../datasets/dog_breeds/train --model_type=vgg16 
--val_data=../datasets/dog_breeds/valid
```

Arguments for python train.py 

`--verbose=True` 

`--device=cuda`

`--num_classes=3`

`--n_epochs=30`

`--learn_rate=0.001`

`--save_path=trained_models/test_train_tmp`

 `--csv_labels=../datasets/animals/labels.csv`

 `--img_size=244`

 `--batch_size=32`

 `--num_workers=0` 

 `--data_dir=../datasets/animals/`

## Using a Model to Predict Classes

Run test.py

Example:

```
python test.py --device=cuda --weights=trained_models/test_train.pt 
--data_csv=../datasets/test_animals/test_labels.csv
--data_dir=../datasets/test_animals/ --num_classes=3
```
csv
python test.py --device=cuda --weights=trained_models/test_train_tmp.pt --data_csv=../datasets/test_animals/test_labels.csv --data_dir=../datasets/test_animals/ --num_classes=3 --model_type=scratch

directory 
python test.py --device=cuda --weights=trained_models/test_train_tmp.pt --data_dir=../datasets/test_animals/ --num_classes=3 --model_type=vgg16



## Run Flask Server

This is a very bare bones fask set up to help get you started hosting your model
You can upload an image and get back a response with the prediction index.

The pytorch docs have a good flask example 
[here](https://pytorch.org/tutorials/intermediate/flask_rest_api_tutorial.html)

I'll be building a more robust example in a different repo soon, but hopefully
this will be enough to get you started!

[app.py](app.py) contains the flask app

[flask_test.py](flask_test.py) contains some example requests you can modify
to quickly check your app behavior on different images.

Windows:

`set FLASK_ENV=development`

`set FLASK_APP=app.py`

Mac/linux:

`export FLASK_ENV=development`

`export FLASK_APP=app.py`

`flask run`

## Requirments

[requirements.txt](requirements.txt)

The versions of libraries I used / tested on

- python `3.7.5`
- cudatoolkit `10.1.243`
- flask `1.1.1`
- matplotlib `3.1.2`
- numpy `1.17.2`
- pandas `0.25.2`
- pillow `6.2.1`
- pytorch `1.3.1`
- seaborn `0.9.0`
- torchvision `0.4.2`
- jupyter `1.0.0` (optional: only if you want to load in notebook)
- click `7.1.1`


## Acknowledgements

Inspiration for building this came while I was doing Udacity's 
[Deep Learning Nanodegree](https://www.udacity.com/course/deep-learning-nanodegree--nd101) 
and working on the Dog Breed classification project.
You can find their prompt [here](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/project-dog-classification). This goes beyond the scope they covered but I want to thank them for such great intro to Pytorch.

And of course the offcial [Pyorch](https://pytorch.org/) docs were handy. 
Specifically the docs on [creating a custom data loader](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html), and running a model on [flask](https://pytorch.org/tutorials/intermediate/flask_rest_api_tutorial.html) 


## Improvments
- [x] accept command line args in train.py
- [x] Add validation data option to train.py
- [x] accept command line args in test.py
- [ ] create jupyter examples
- [ ] Use Shap values or Captum for Model Interpretability
- [ ] Save model per epoch option (currently rewrites if loss decreases)
- [ ] check for run folder 
- [ ] image transform options
