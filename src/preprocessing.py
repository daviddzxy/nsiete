import pandas as pd
import os
import numpy as np
import argparse
import cv2
import re

parser = argparse.ArgumentParser("Preprocessing of images in /data/raw/. Cuts out dogs from images and resizes them. Preprocessed images are stored in /data/processed.")


def main(args):
    if re.match(".*/src$", os.getcwd()) or re.match(".*/notebooks$", os.getcwd()):
        os.chdir("../")  # change directory to root directory
    annotation_path = "./data/annotations.csv"
    raw_data_path = "./data/raw/" 
    processed_data_path = "./data/processed/"
    df = pd.read_csv(annotation_path)
    for index, row in df.iterrows():
        img = cv2.imread(raw_data_path + row["filename"] + ".jpg")
        img = img[row["ymin"]:row["ymax"], row["xmin"]: row["xmax"]]  # crop image
        img = cv2.resize(img, (args.r, args.r))
        cv2.imwrite(processed_data_path + str(row["id"]) + ".png", img)


if __name__ == '__main__':
    parser.add_argument("-r", "--resize", default="128", type=int, help="Width and height of resized image.")

    parsed_args = parser.parse_args()
    main(parsed_args)

