import os

# generate random floating point values
from random import random, seed
# seed random number generator
seed(1)

DATA_DIR = "/mnt/c/Users/Cooma/Documents/counter-training/dataset/MIO-TCD/data"

dir = "train"
TRAIN_IMAGES_PATH = f"{DATA_DIR}/images/{dir}"
TRAIN_LABELS_PATH = f"{DATA_DIR}/labels/{dir}"

dir = "ce"
CE_IMAGES_PATH = f"{DATA_DIR}/images/{dir}"
CE_LABELS_PATH = f"{DATA_DIR}/labels/{dir}"

dir = "test"
TEST_IMAGES_PATH = f"{DATA_DIR}/images/{dir}"
TEST_LABELS_PATH = f"{DATA_DIR}/labels/{dir}"

# Get all files in the directory
files = os.listdir(f"{DATA_DIR}/images")
count = 0

for fileName in files:
    if fileName[0] != '.': # exclude hidden files
        fileType = fileName[-3:] # Get file extension

        if fileType == 'jpg': # specify file type
            label_filename = fileName[:-3] + "txt" # Generate label filename

            count+=1

            try:
                choose = random()

                if choose <= 0.05:
                    os.rename(f"{DATA_DIR}/images/{fileName}", f"{TRAIN_IMAGES_PATH}/{fileName}") # image
                    os.rename(f"{DATA_DIR}/labels/{label_filename}", f"{TRAIN_LABELS_PATH}/{label_filename}") # label
                elif choose <= 0.45:
                    os.rename(f"{DATA_DIR}/images/{fileName}", f"{CE_IMAGES_PATH}/{fileName}") # image
                    os.rename(f"{DATA_DIR}/labels/{label_filename}", f"{CE_LABELS_PATH}/{label_filename}") # label
                else:
                    os.rename(f"{DATA_DIR}/images/{fileName}", f"{TEST_IMAGES_PATH}/{fileName}") # image
                    os.rename(f"{DATA_DIR}/labels/{label_filename}", f"{TEST_LABELS_PATH}/{label_filename}") # label
            except Exception as e:
                print(f"Fail to rename {label_filename}")
                print(e)
                exit() 

