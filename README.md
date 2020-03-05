
https://pytorch.org/tutorials/intermediate/flask_rest_api_tutorial.html



## Run Flask server

This is a very bare bones fask set up to help get you started hosting your model
You can upload an image and get back a response with the prediction index.

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