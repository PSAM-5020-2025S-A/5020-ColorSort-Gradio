import gradio as gr
import json
import numpy as np
import tarfile

import PIL.Image as PImage

from os import listdir, path, remove
from sklearn.cluster import KMeans
from urllib import request

def download_extract():
  url = "https://github.com/PSAM-5020-2025S-A/5020-utils/releases/latest/download/flowers.tar.gz"
  target_path = "flowers.tar.gz"

  with request.urlopen(request.Request(url), timeout=15.0) as response:
    if response.status == 200:
      with open(target_path, "wb") as f:
        f.write(response.read())
  
  tar = tarfile.open(target_path, "r:gz")
  tar.extractall()
  tar.close()
  remove("flowers.tar.gz")

# Posterize image and get representative colors
def top_colors(fpath, n_clusters=8, n_colors=4):
  pimg = PImage.open(fpath).convert("RGB")
  pimg_pxs = list(pimg.getdata())

  posterizer = KMeans(n_clusters=n_clusters)
  px_clusters = posterizer.fit_predict(pimg_pxs)
  cluster_colors = posterizer.cluster_centers_

  _, ccounts = np.unique(px_clusters, return_counts=True)
  ccounts_order = np.argsort(-ccounts)
  ccolors_sorted = [[round(rgb) for rgb in cluster_colors[idx]] for idx in ccounts_order]

  return ccolors_sorted[:n_colors]

# Cluster all images
def get_top_colors(flower_image_dir):
  flower_files = sorted([f for f in listdir(flower_image_dir) if f.endswith(".png")])

  file_colors = []
  for fname in flower_files:
    file_colors.append({
      "filename": fname,
      "colors": top_colors(f"{flower_image_dir}/{fname}", n_clusters=8, n_colors=4)
    })
  return file_colors

# Euclidean distance between 2 RGB color tuples
def color_distance(c0, c1):
  return ((c0[0] - c1[0])**2 + (c0[1] - c1[1])**2 + (c0[2] - c1[2])**2) ** 0.5

# Function that returns minimum distance between a reference color and colors from a list
def min_color_distance(ref_color, color_list):
  c_dists = [color_distance(ref_color, c) for c in color_list]
  return min(c_dists)

# Turns a color hex string in the form `#12AB56`
#   into an RGB tuple (18, 171, 87)
def hex_string_to_rgb(hex_str):
  return (
    int(hex_str[1:3], 16),
    int(hex_str[3:5], 16),
    int(hex_str[5:7], 16),
  )

def order_by_color(center_color_str):
  center_color = hex_string_to_rgb(center_color_str)

  # Function that returns how close an image is to a given color
  def by_color_dist(A):
    return min_color_distance(center_color, A["colors"])

  file_colors_sorted = sorted(FILE_COLORS, key=by_color_dist)
  files_sorted = [A["filename"] for A in file_colors_sorted]

  file_order = {
    "color": center_color,
    "files": files_sorted
  }

  return json.dumps(file_order)

my_inputs = [
  gr.ColorPicker(value="#ffdf00", label="center_color", interactive=True)
]

my_outputs = [
  gr.JSON(show_label=False, show_indices=False, height=200, container=False)
]

my_examples = [
  ["#FFFFFF"],
  ["#FFD700"],
  ["#7814BE"]
]

def setup():
  global FILE_COLORS
  FLOWER_IMG_DIR = "./data/image/flowers"
  if not path.isdir(FLOWER_IMG_DIR):
    download_extract()
  FILE_COLORS = get_top_colors(FLOWER_IMG_DIR)

setup()

with gr.Blocks() as demo:
  gr.Interface(
    fn=order_by_color,
    inputs=my_inputs,
    outputs=my_outputs,
    cache_examples=True,
    examples=my_examples,
    allow_flagging="never",
    fill_width=True
  )

if __name__ == "__main__":
  demo.launch()
