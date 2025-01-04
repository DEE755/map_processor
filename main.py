import argparse
import itertools

import cv2
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from skimage.morphology import skeletonize
import PyQt5

import matplotlib
matplotlib.use('Qt5agg')
import matplotlib.pyplot
fig=matplotlib.pyplot.figure()
fig.canvas.draw()
fig.canvas.tostring_argb()

def show_image(image):


 cv2.imshow("map", image)

 cv2.waitKey(0)

 cv2.destroyAllWindows()


parser = argparse.ArgumentParser(description="path of the map")
parser.add_argument(
    "file_path",
    type=str,
    nargs="?",
    help="absolute path of the map"
)

args = parser.parse_args()

if args.file_path is not None:
    map_image_path = args.file_path

else :
    map_image_path = "Tampa,_FL.png"

map_image = cv2.imread(map_image_path)

gray_map = cv2.cvtColor(map_image, cv2.COLOR_BGR2GRAY)



blurred_map = cv2.GaussianBlur(gray_map, (5, 5), 0)

# Apply adaptive thresholding for better road isolation
#binary_map = cv2.adaptiveThreshold(blurred_map, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

binary_map=cv2.Canny(gray_map,100,200)

#show_image(binary_map)
# Skeletonize the binary image
skeleton = skeletonize(binary_map // 255).astype(np.uint8) * 255

#show_image(skeleton)

lines = cv2.HoughLinesP(skeleton, rho=1, theta=np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)

# Visualize detected lines
line_image = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR)
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green lines for visualization

plt.figure(figsize=(12, 8))
plt.title("Hough Line Transform - Detected Lines")
plt.imshow(line_image)
plt.axis("off")
#plt.show()

# Calculate intersections of the detected lines






#kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]], dtype=np.uint8)
#filtered = cv2.filter2D(skeleton, -1, kernel)
#intersections = np.where(filtered > 255)

# Extract nodes as intersections
#nodes = list(zip(intersections[1], intersections[0]))



#G = nx.Graph()
#for i, (x, y) in enumerate(nodes):
    #G.add_node(i, pos=(x, y))





#plt.figure(figsize=(10, 10))
#plt.imshow(cv2.cvtColor(map_image, cv2.COLOR_BGR2RGB))
#plt.axis("off")

#pos = nx.get_node_attributes(G, 'pos')
#nx.draw(G, pos, with_labels=False, node_size=25, edge_color="blue", node_color="red", alpha=0.6)
#plt.title("Graph of Road Intersections")

#print(f"Number of nodes: {len(nodes)}")
#print(f"Number of edges: {len(G_intersections.edges)}")

#plt.show()



