import pandas as pd
import os
import numpy as np
import argparse
import cv2
import re

parser = argparse.ArgumentParser("Preprocessing of images in /data/raw/.")


def main(args):
    if re.match(".*/src/preprocessing$", os.getcwd()):
        os.chdir("../../")  # change directory to root directory
    annotation_path = args.a
    raw_data_path = args.s
    processed_data_path = args.d
    df = pd.read_csv(annotation_path)
    for index, row in df.iterrows():
        img = cv2.imread(raw_data_path + row["filename"] + ".jpg")
        img = img[row["ymin"]:row["ymax"], row["xmin"]: row["xmax"]]  # crop image
        img = cv2.resize(img, (args.r, args.r))
        img = img / 255
        np.save(os.path.join(processed_data_path + str(row["id"])), img)


if __name__ == '__main__':
    parser.add_argument("-r", default="128", type=int, help="Width and height of resized image")
    parser.add_argument("-s", default="./data/raw/", type=str, help="Source directory path of images")
    parser.add_argument("-d", default="./data/processed/", type=str, help="Destination directory path of images")
    parser.add_argument("-a", default="./data/annotations.csv", type=str, help="Path to csv annotation")

    parsed_args = parser.parse_args()
    main(parsed_args)

