import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt
import networkx as nx
from skimage.morphology import skeletonize
import matplotlib
matplotlib.use('Qt5agg')
import matplotlib.pyplot
fig = matplotlib.pyplot.figure()
fig.canvas.draw()
fig.canvas.tostring_argb()


# Function to calculate Euclidean distance
def euclidean_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


# Function to merge nearby nodes and rebuild the graph
def simplify_graph(G, distance_threshold):
    merged_nodes = {}
    for node in list(G.nodes):
        merged = False
        for key in merged_nodes.keys():
            if euclidean_distance(node, key) < distance_threshold:
                merged_nodes[key].append(node)
                merged = True
                break
        if not merged:
            merged_nodes[node] = [node]

    # Rebuild the graph with merged nodes
    new_G = nx.Graph()
    node_mapping = {}
    for new_node, old_nodes in merged_nodes.items():
        new_G.add_node(new_node)
        for old_node in old_nodes:
            node_mapping[old_node] = new_node

    for edge in G.edges:
        new_G.add_edge(node_mapping[edge[0]], node_mapping[edge[1]])

    return new_G


# Initialize argument parser
parser = argparse.ArgumentParser(description="path of the map")
parser.add_argument(
    "file_path",
    type=str,
    nargs="?",
    help="absolute path of the map"
)

parser.add_argument(
    "--threshold",
    type=str,
    nargs="?",
    help="threshold for simplification"
)



args = parser.parse_args()

# Map image path
if args.file_path is not None:
    map_image_path = args.file_path
else:
    map_image_path = "Tampa,_FL.png"

# Load and preprocess the map
map_image = cv2.imread(map_image_path)
gray_map = cv2.cvtColor(map_image, cv2.COLOR_BGR2GRAY)
blurred_map = cv2.GaussianBlur(gray_map, (5, 5), 0)
binary_map = cv2.Canny(gray_map, 100, 200)

# Skeletonize the binary map
skeleton = skeletonize(binary_map // 255).astype(np.uint8) * 255

# Detect lines using Hough Line Transform
lines = cv2.HoughLinesP(skeleton, rho=1, theta=np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)

# Create a graph
G = nx.Graph()

# Add nodes and edges from detected lines
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]

        # Add nodes (start and end points of the line)
        G.add_node((x1, y1))
        G.add_node((x2, y2))

        # Add edge between the start and end points
        G.add_edge((x1, y1), (x2, y2))

# Simplify the graph by merging nearby nodes
if args.threshold is not None:
    distance_threshold = int(args.threshold)
else:
    distance_threshold = 50
# Increase this value to reduce the number of nodes
new_G = simplify_graph(G, distance_threshold)
new_G.remove_edges_from(nx.selfloop_edges(new_G))


# Visualize the simplified graph
plt.figure(figsize=(12, 8))
plt.imshow(cv2.cvtColor(map_image, cv2.COLOR_BGR2RGB))
pos = {node: node for node in new_G.nodes}
nx.draw(new_G, pos, node_size=10, edge_color="blue", node_color="red", with_labels=False)
plt.title(f"Road Graph (Simplified, Threshold={distance_threshold})")
plt.axis("off")
plt.show()

# Output the number of nodes and edges
print(f"Number of nodes: {len(new_G.nodes)}")
print(f"Number of edges: {len(new_G.edges)}")
