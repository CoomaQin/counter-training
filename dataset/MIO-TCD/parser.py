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
            "pedestrian": "0",
            "bicycle": "1",
            "car": "2",
            "motorcycle": "3",
            "bus": "4",
            "motorized_vehicle": "5",
            "single_unit_truck": "6",
            "articulated_truck": "7",
            "non-motorized_vehicle": "8",
            "pickup_truck": "9",
            "work_van": "10",
        }
        self.lbl_mapping = {
            "0": "pedestrian",
            "1": "bicycle",
            "2": "car",
            "3": "motorcycle",
            "4": "bus",
            "5": "motorized_vehicle",
            "6": "single_unit_truck",
            "7": "articulated_truck",
            "8": "non-motorized_vehicle",
            "9": "pickup_truck",
            "10": "work_van",
            "11": "FL"
        }


    def read(self):
        if os.path.exists(self.data_path):
            self.df = pd.read_csv(self.data_path + "gt_train.csv", dtype={
                         "img": str, "cls": str, "x1": np.int32, "y1": np.int32, "x2": np.int32, "y2": np.int32})
        else:
            print("data_path not exist")


    def extract_object(self, target="bus"):
        """
        Extract instance-wise images.
        """
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
        """
        Convert the original MIO-TCD annotation into YOLO format.
        """
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
            if os.path.isfile(label_path):
                with open(label_path, "a+") as f:
                    f.write(f"{self.cls_mapping[cls]} {((x1 + x2) / 2 / width):.6f} {((y2 + y1) / 2 / height):.6f} {((x2 - x1) / width):.6f} {((y2 - y1) / height):.6f} \n")
            else:
                with open(label_path, "w+") as f:
                    f.write(f"{self.cls_mapping[cls]} {((x1 + x2) / 2 / width):.6f} {((y2 + y1) / 2 / height):.6f} {((x2 - x1) / width):.6f} {((y2 - y1) / height):.6f} \n")


    def test_ori(self):
        """
        Plot bounding boxes in images for manually verifying whether annotations are correct 
        """
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

    
    def test_yolo(self, data_path="./data"):
        """
        Plot bounding boxes in images for manually verifying whether annotations are correct 
        """
        image_path = os.path.join(data_path, "images", "ce")
        label_path = os.path.join(data_path, "labels", "ce_fpa")
        images = []
        for (dirpath, dirnames, filenames) in os.walk(image_path):
            images.extend(filenames)
            break
        print(len(images))
        for idx, img_name in enumerate(images):
            img = cv2.imread(os.path.join(image_path, img_name), cv2.IMREAD_COLOR)
            label_name = img_name.split(".")[0] + ".txt"
            label = []
            with open(os.path.join(label_path, label_name)) as f:
                label = f.readlines()
            height, width, _ = img.shape
            # print(label)
            for line in label:
                cls, x, y, w, h, _ = line.split(" ")
                x1, y1, x2, y2 = int((float(x) - float(w) / 2) * width), int((float(y) - float(h) / 2) * height), int((float(x) + float(w) / 2) * width), int((float(y) + float(h) / 2) * height)
                img = cv2.putText(img, self.lbl_mapping[cls], (x1, y1 - 3), 1, 1, [0, 255, 255], thickness=1, lineType=cv2.LINE_AA)
                img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1, cv2.LINE_AA)
                cv2.imwrite("./raw_fasle_predictions/" + img_name, img)
            if idx > 10000:
                break

        

if __name__ == "__main__":
    parser = Parser()
    # parser.read()
    parser.test_yolo()
    # parser.test_yolo()
