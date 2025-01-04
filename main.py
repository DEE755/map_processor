import argparse
import cv2
import numpy as np


parser = argparse.ArgumentParser(description="path of the map")
parser.add_argument(
    "file_path",
    type=str,
    nargs="?",
    help="absolute path of the map"
)

args = parser.parse_args()

if args.time_in_min is not None:
    amap = args.file_path

gray = cv2.cvtColor(amap, cv2.COLOR_BGR2GRAY)

# _, binary=cv2.threshold(gray,128,255,cv2.THRESH_BINARY)
binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

edges = cv2.Canny(binary, 100, 200)

blurred = cv2.GaussianBlur(edges, (5, 5), 0)

cv2.imshow("map", blurred)
cv2.waitKey(0)

cv2.destroyAllWindows()









