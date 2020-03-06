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

## Acknowledgements

Inspiration for building this came while I was doing Udacity's 
[Deep Learning Nanodegree](https://www.udacity.com/course/deep-learning-nanodegree--nd101) 
and working on the Dog Breed classification project.
You can find their prompt [here](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/project-dog-classification). This goes beyond the scope they covered but I want to thank them for such great intro to Pytorch.

And of course the offcial [Pyorch](https://pytorch.org/) docs were handy. 
Specifically the docs on [creating a custom data loader](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html), and running a model on [flask](https://pytorch.org/tutorials/intermediate/flask_rest_api_tutorial.html) 


## Requirments

## Loading Data

## Training a Model

## Using a Model to Predict Classes

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

## Improvments

- [ ] Use Argparse for training inputs
- [ ] Use Shap values or Captum for Model Interpretabilitys