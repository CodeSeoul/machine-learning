# Instructions

## Prerequisites
1. Conda or virtual environment with, for example:  
`conda create -n tfseg python=3.8`
2. Install requirements:  
`pip install -r requirements.txt`  
**Note**: This tutorial on gpu-supported laptop (Windows WSL 2). If you do not have GPU, code might be adjusted.

## Dataset
1. Download [Flood Area Dataset](https://www.kaggle.com/datasets/faizalkarim/flood-area-segmentation) from [here](https://www.kaggle.com/datasets/faizalkarim/flood-area-segmentation?resource=download)

## Code
### A. Model training
1. Run `train.py` to get original trained model, named `tf_full.h5`

### B. Quantization
1. Run `quantization.py` to get quantized model, `unet.tflite`, `unet_fp16.tflite` and `unet_int8.tflite` for fp32, fp16 and uint8 
datatype-supported quantized models respectively.

### C. Pruning
1. Run `pruning.py` to get pruned model `tf_pruning.h5`
2. Run `inference.ipynb` to check result of inference from `tf_pruning.h5` pruned model

## References

- [Segmentation notebook on Kaggle](https://www.kaggle.com/code/faizalkarim/flood-segmentation-unet)  
- [Quantization with Tensorflow Lite](https://www.tensorflow.org/lite/performance/post_training_quantization)
- [Pruning with Tensorflow Model Optimization Toolkit](https://www.tensorflow.org/model_optimization/guide/pruning/pruning_with_keras)
