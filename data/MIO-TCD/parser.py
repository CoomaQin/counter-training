import pandas as pd
import os
import numpy as np
import cv2
from tqdm import tqdm


class Parser():
    def __init__(self, data_path='./MIO-TCD-Localization/', output_path='./MOT_TCD_refine/labels/'):
        self.data_path = data_path
        self.output_path = output_path
        self.cls_mapping = {
            "articulated_truck": "6",
            "bicycle": "1",
            "bus": "4",
            "car": "2",
            "motorcycle": "3",
            "motorized_vehicle": "2",
            "non-motorized_vehicle": "2",
            "pedestrian": "0",
            "pickup_truck": "2",
            "single_unit_truck": "6",
            "work_van": "2",
        }

    def read(self):
        if os.path.exists(self.data_path):
            self.df = pd.read_csv(self.data_path + "gt_train.csv", dtype={
                         "img": str, "cls": str, "x1": np.int32, "y1": np.int32, "x2": np.int32, "y2": np.int32})
        else:
            print("data_path not exist")

    def extract_object(self, target="bus"):
        obj_idx = 0
        for _, row in self.df.iterrows():
            img_name, cls, x1, y1, x2, y2 = row.values
            if cls == target:
                image_path = f"{self.data_path}train/{img_name}.jpg"
                if not os.path.exists(image_path):
                    image_path = f"{self.data_path}test/{img_name}.jpg"
                print(image_path)
                img = cv2.imread(image_path, cv2.IMREAD_COLOR)
                cv2.imwrite(f"{self.output_path}{obj_idx}.jpg", img[y1:y2:, x1:x2, :])
                obj_idx += 1
                if obj_idx >= 2000:
                    break
    
    
    def yolo_label(self):
        for idx, row in tqdm(self.df.iterrows()):
            img_name, cls, x1, y1, x2, y2 = row.values
            image_path = f"{self.data_path}train/{img_name}.jpg"
            is_train = True
            if not os.path.exists(image_path):
                image_path = f"{self.data_path}test/{img_name}.jpg"
                is_train = False
            img = cv2.imread(image_path, cv2.IMREAD_COLOR)
            height, width, _ = img.shape
            label_path = f"{self.output_path}{img_name}.txt"
            # print("label_path", label_path, os.path.isfile(label_path))
            if os.path.isfile(label_path):
                with open(label_path, "a+") as f:
                    f.write(f"{self.cls_mapping[cls]} {((x1 + x2) / 2 / width):.6f} {((y2 + y1) / 2 / height):.6f} {((x2 - x1) / width):.6f} {((y2 - y1) / height):.6f} \n")
            else:
                with open(label_path, "w+") as f:
                    f.write(f"{self.cls_mapping[cls]} {((x1 + x2) / 2 / width):.6f} {((y2 + y1) / 2 / height):.6f} {((x2 - x1) / width):.6f} {((y2 - y1) / height):.6f} \n")


    def test(self):
        df = pd.read_csv(self.data_path + "gt_train.csv", dtype={
                         "img": str, "cls": str, "x1": np.int32, "y1": np.int32, "x2": np.int32, "y2": np.int32})
        print(df.values[0].tolist())
        img_name, cls, x1, y1, x2, y2 = df.values[1].tolist()
        image_path = f"{self.data_path}train/{img_name}.jpg"
        if not os.path.exists(image_path):
            image_path = f"{self.data_path}test/{img_name}.jpg"
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        # img = cv2.putText(img, cls, (x1, y1 - 2), 1, 0.5, [0, 255, 255], thickness=1, lineType=cv2.LINE_AA)
        # img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imwrite("./1.jpg", img[y1:y2:, x1:x2, :])
        

if __name__ == "__main__":
    parser = Parser()
    parser.read()
    parser.yolo_label()
