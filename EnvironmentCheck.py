import torch
import tensorflow as tf
import cv2
import json
import pandas as pd
import numpy as np
import os
import ultralytics
from pycocotools.coco import COCO

# Check PyTorch and CUDA
def check_pytorch():
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA version:", torch.version.cuda)
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        print("CUDA is not available. Check your GPU drivers and CUDA installation.")

# Check TensorFlow and GPU support
def check_tensorflow():
    print("TensorFlow version:", tf.__version__)
    print("TF GPU available:", tf.test.is_gpu_available())
    print("TF devices:", tf.config.list_physical_devices('GPU'))

# Check OpenCV version
def check_opencv():
    print("OpenCV version:", cv2.__version__)

# Check ultralytics (YOLO)
def check_ultralytics():
    print("Ultralytics (YOLO) version:", ultralytics.__version__)

# Check COCO API
def check_coco():
    try:
        COCO()
        print("COCO API is installed and working.")
    except Exception as e:
        print("COCO API check failed:", e)

# Check Pandas & NumPy
def check_pandas_numpy():
    print("Pandas version:", pd.__version__)
    print("NumPy version:", np.__version__)

# Run all checks
if __name__ == "__main__":
    check_pytorch()
    print("\n------------------------\n")
    check_tensorflow()
    print("\n------------------------\n")
    check_opencv()
    print("\n------------------------\n")
    check_ultralytics()
    print("\n------------------------\n")
    check_coco()
    print("\n------------------------\n")
    check_pandas_numpy()
