# Pytorch Convolutional Neural Network Starter Template

## Summary

I had been building out Convolutional Neural Networks to solve various research 
problems or as fun personal projects. After doing many of them I found myself 
reproducing a large amount of the same code so I thought I should make a
template for myself. I wanted something that I could grab and start easily using
new datasets and architectures. I also didn't want it to live in a giant
jupyter notebook. So I broke out everything into model and data loader helper
functions. 

This repo also contains a bare bones flask app that returns json response
with the pytorch models predicted class.

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

## Loading Data

Directory (Images in folders seperated by class)

CSV 

See examples in:
- animal_load_train_csv.py


## Training a Model

All aspect of loading images and training models are broken into functions.

Please visit the [Example Jupyter Notebook]() or [train.py](train.py)

### Using train.py from terminal:

[train.py](train.py) can be run from terminal and passed commands arguments

*note:* Currently train.py is set to use a pre-trained resnet50. reference the [models folder](/models) for how a model is built

Example:

CSV labels

```
python train.py --verbose=True --device=cuda --num_classes=3 --n_epochs=30 --learn_rate=0.001 --save_path=trained_models/test_train_tmp --csv_labels=../datasets/animals/labels.csv --img_size=244 --batch_size=32 --num_workers=0 --data_dir=../datasets/animals/

```

Example directory

```
python train.py --verbose=True --device=cuda --num_classes=3 --n_epochs=30 --learn_rate=0.001 --save_path=trained_models/test_train_tmp --img_size=244 --batch_size=32 --num_workers=0 --data_dir=../datasets/animals/

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
python test.py --device=cuda --weights=trained_models/test_train.pt --data_csv=../datasets/test_animals/test_labels.csv --data_dir=../datasets/test_animals/ --num_classes=3
```


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

## Acknowledgements

Inspiration for building this came while I was doing Udacity's 
[Deep Learning Nanodegree](https://www.udacity.com/course/deep-learning-nanodegree--nd101) 
and working on the Dog Breed classification project.
You can find their prompt [here](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/project-dog-classification). This goes beyond the scope they covered but I want to thank them for such great intro to Pytorch.

And of course the offcial [Pyorch](https://pytorch.org/) docs were handy. 
Specifically the docs on [creating a custom data loader](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html), and running a model on [flask](https://pytorch.org/tutorials/intermediate/flask_rest_api_tutorial.html) 


## Improvments
- [X] Use Argparse / FLAGs for training inputs
- [ ] Use Shap values or Captum for Model Interpretability