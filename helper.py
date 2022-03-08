import pandas as pd
import os
import numpy as np
import cv2
from tqdm import tqdm
import torch


def get_IoU(box1, box2):
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)

class Helper:
    def __init__(self, data_path, label_path) -> None:
        self.data_path = data_path
        self.label_path = label_path

    def get_files(self, path) -> list:
        files = []
        for (_, _, filenames) in os.walk(path):
            files.extend(filenames)
            break
        return files

    def generate_false_prediction(self, weight, output_path, force_reload=False) -> None:
        yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', weight, force_reload=force_reload)
        images = self.get_files(self.data_path)
        for image_name in images:
            fl_str = ""
            mc_str = ""
            lbl_name = image_name.split(".")[0] + ".txt"
            labels = []
            img = cv2.imread(os.path.join(self.data_path, image_name))
            height, width, _ = img.shape
            with open(os.path.join(self.label_path, lbl_name)) as f:
                labels = f.readlines()
            yolo_results = yolo_model(img)
            # yolo_results.print()
            preds = yolo_results.xyxy[0].cpu()
            gts = []
            for lbl in labels:
                cls, x, y, w, h = lbl[:-3].split(" ")
                x1, y1, x2, y2 = (float(x) - float(w) / 2) * width, (float(y) - float(h) / 2) * height, (float(x) + float(w) / 2) * width, (float(y) + float(h) / 2) * height
                gts.append([x1, y1, x2, y2])
            gts = torch.FloatTensor(gts)
            # print(gts, preds[:, :4])
            IoUs = get_IoU(preds[:, :4], gts).numpy
            for k in range(len(IoUs)):
                res = IoUs[k, :]
                match = np.where(res > 0.8, 1, 0)
                if np.sum(match) == 0: # false localization
                    pass
                else:
                    for idx, elem in enumerate(match):
                        if int(elem) == 1 and int(labels[k][0]) != preds[idx][5]: # mis-classification
                            pass

            break
if __name__ == "__main__":
    helpler = Helper("./dataset/MIO-TCD/data/images/ce", "./dataset/MIO-TCD/data/labels/ce")
    helpler.generate_false_prediction("./weights/yolov5s.pt", "./dataset/MIO-TCD/data/")