import pandas as pd
import os
import cv2
import argparse


def main(args):
    os.chdir("../../")  # change directory to root directory
    annotation_path = r"./data/annotations.csv"
    raw_data_path = r"./data/raw/"
    processed_data_path = r"./data/processed/"
    df = pd.read_csv(annotation_path)
    for index, row in df.iterrows():
        img = cv2.imread(raw_data_path + row["filename"] + ".jpg")
        img = img[row["ymin"]:row["ymax"], row["xmin"]: row["xmax"]]  # crop image
        img = cv2.resize(img, (args.r, args.r))
        cv2.imwrite(processed_data_path + row["filename"] + ".png", img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Preprocessing of images in /data/raw/.")
    parser.add_argument("-r", default="128", type=int, help="Width and height of resized image")
    parsed_args = parser.parse_args()
    main(parsed_args)

