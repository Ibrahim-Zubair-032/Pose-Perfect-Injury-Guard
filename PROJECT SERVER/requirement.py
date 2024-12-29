import flask
import mediapipe as mp
import cv2
import numpy as np
import pandas as pd
import pickle
import sklearn
import os
import sys

# Check versions of the imported libraries
print("Flask version:", flask.__version__)
print("Mediapipe version:", mp.__version__)
print("OpenCV version:", cv2.__version__)
print("NumPy version:", np.__version__)
print("Pandas version:", pd.__version__)
print("Pickle version: (built-in, no version)")
print("Scikit-learn version:", sklearn.__version__)
print("OS: (built-in, no version)")
print("Python version : ",sys.version)