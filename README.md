## data preparation
1. download MIO-TCD dataset
   `cd /dataset/MIO-TCD/`
   `.download.sh`
2. Unzip files, run **parser.py** to gather annotation in YOLO1.1 format. 
   
3. Create three folders **train**, **ce**, **test** under images and labels separately, run **split_data.py** to separate the data. 

## TODO
- [x] Modify YOLO dataloading script to adapt to one image foler vs two label folders.
- [x] Get batches from false localizations and mislcassification alteratively, feed them to the corresponding loss funtion. 
- [ ] AUC-ROC  
- [x] Merge pedestrian, other people, rider
- [ ] Imporve False localization gathering (if a yolo detection is a subarea of the ground-truth, it is not a false localization)
- [ ] Use the previous classification loss of each class to weight their misclassificaiton loss
- [ ] adapt prototype net sampler to data which doesn't has the same amount of sample in each class
- [ ] counter training grid search
- [ ] compare counter training with regular training in converged performance
- [ ] prototype net grid search 
- [ ] dock prototype net with counter training
- [ ] adaptive adversarial training (when yolo is accurate, do more error-driven learning otherwise regualr trainging, when prototype net is accurate, use more yolo detection as the support element otherwise ground-truth annotations.)
