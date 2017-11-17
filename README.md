# mnist-tensorflow-example
Just the usual mnist example in 3 variations (+ flask serving)

### Dependencies

Install via pip


tensorflow (https://www.tensorflow.org/install/)

keras (https://keras.io/#installation)

h5py

flask

### mnist.py

`python mnist.py`

Taken from the tensorflow tutorial (https://www.tensorflow.org/get_started/mnist/beginners)

### mnist_conv.py

`python mnist-conv.py`

Taken from the tensorflow tutorial (https://www.tensorflow.org/get_started/mnist/pros)

### mnist-keras.py

`python mnist-keras.py`

Basic mnist cnn using keras (https://keras.io/)

### mnist-keras-flask.py

`python mnist-keras-flask.py`

Simple web app that loads the keras model created with the `mnist-keras.py` script and lets you test it (taken from https://github.com/llSourcell/how_to_deploy_a_keras_model_to_production)

