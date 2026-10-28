"""Download the Kaggle 'Student Performance Factors' dataset into ./data

Setup (once):
  1. pip install kagglehub
  2. Create a Kaggle API token (kaggle.com -> Settings -> API) and either run
     `kaggle auth login`-style setup or set KAGGLE_USERNAME / KAGGLE_KEY.
Run:
  python download_dataset.py

Or skip all this: download the CSV manually from
https://www.kaggle.com/datasets/lainguyn123/student-performance-factors
and place StudentPerformanceFactors.csv inside the data/ folder.
"""
import glob
import os
import shutil

import kagglehub

dest = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(dest, exist_ok=True)

path = kagglehub.dataset_download("lainguyn123/student-performance-factors")
for f in glob.glob(os.path.join(path, "*.csv")):
    shutil.copy(f, dest)
    print("Saved", os.path.join(dest, os.path.basename(f)))
