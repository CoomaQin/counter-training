## data preparation
1. download MIO-TCD dataset
   `cd /dataset/MIO-TCD/`
   `.download.sh`
2. Unzip files, run **parser.py** to gather annotation in YOLO1.1 format. 
3. Create three folders **train**, **ce**, **test** under images and labels separately, run **split_data.py** to separate the data. 

## TODO
- modify YOLO dataloading script to adapt to one image foler vs two label folders.
- get batches from false localizations and mislcassification alteratively, feed them to the corresponding loss funtion. 
