# Smart Grid_Stability Arduino TFlite

The main goal of this project is to create an ANN Network and later deploy it on Arduino using TensorFlow Lite. The ANN can predict whether a Smart Grid is stable or unstable (Binary Classification). The code was created on Google Collab, thus making it easy to replicate. Essentially the only thing that need to be changed is the folder path (but keeping the folder structure identical to the one on Github).

There are 3 sections:

1. An end to end walkthrough of the project
2. General details about the experiments that lead to the final results
3. Problems regarding the TFLite and Arduino Deployment

## 1 - End to End walkthrough

- Go to the final folder
- Run smart_grid_stability_train.ipynb. This will generate the tensorflow model as well as the representative dataset which will be need for quantization.
- Run model_quant.ipynb and model_quant_io.ipynb. They will generate the tflite models and header files for the two quantization approaches. One is full integer quantization without affecting the input/output and the second quantizes both the input and the output of the model. The last approach leads to an unexpected behavior which will be explained in the last section. 

***IMPORTANT***
- If you wish to alter the code and create new models you will have to move them from the TFLITE_models folder to their respective arduino folder. Either smart_grid_no_io_quant(no input-output quantization) or to smart_grid_io_quant(with input-output quantization)


## 2 - Experiments

- Go to the experiment folder
- There you will find the notebooks that contain all the model and quantization approaches that were tried. Only two quantization methods were able to run on the arduno though

## 3 - Problems
