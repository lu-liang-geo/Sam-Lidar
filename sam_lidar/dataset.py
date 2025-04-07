import os
import laspy
import torch
import pickle
import rasterio
import numpy as np
import geopandas as gpd
import xml.etree.ElementTree as ET
from supervision.detection.utils import polygon_to_mask

class NEONTreeDataset(torch.utils.data.Dataset):
  def __init__(self, rgb_folder, lidar_folder, ann_folder):
    '''
    params
        rgb_folder (str): Path to directory containing RGB images
        lidar_folder (str): Path to directory containing LiDAR point clouds
        ann_folder (str): Path to directory containing bounding box and polygon annotations
    '''
    self.rgb_folder = rgb_folder
    self.lidar_folder = lidar_folder
    self.ann_folder = ann_folder
    file_names = os.listdir(rgb_folder)
    self.basenames = [name.split('.')[0] for name in file_names]

  def __len__(self):
    return len(self.basenames)

  def __getitem__(self, idx):
    '''
    returns
        annotated_image (dict) with keys:
            rgb: HxWxC ndarray of RGB channels
            lidar: HxWxC ndarray of multi channels
            prompt (if prompt_path specified): Nx4 ndarray of prompt bounding boxes in XYXY format
            annotation: Nx4 ndarray of ground truth bounding boxes in XYXY format
    '''
    basename = self.basenames[idx]

    rgb_path = os.path.join(self.rgb_folder, f'{basename}.tif')
    with rasterio.open(rgb_path) as rast_img:
        rgb_img = rast_img.read().transpose(1,2,0)
        height = rgb_img.shape[0]
        width = rgb_img.shape[1]

    if self.lidar_folder: 
        lidar_path = os.path.join(self.lidar_folder, f'{basename}.laz')
        las = laspy.read(lidar_path)
    else:
        las = None
    
    # Extract bounding boxes from annotations
    box_path = os.path.join(self.ann_folder, "Boxes", f"{basename}.xml")
    xyxy = []
    tree = ET.parse(box_path)
    root = tree.getroot()
    for obj in root.findall('object'):
        name = obj.find('name').text
        if name == 'Tree':
            bbox = obj.find('bndbox')
            xyxy.append([int(bbox[i].text)-1 for i in range(4)])
    boxes = np.array(xyxy)

    # Extract polygons from annotations
    poly_path = os.path.join(self.ann_folder, "Polygons", basename, f"{basename}.shp")
    polygons = []
    with rasterio.open(rgb_path) as rast_img:   
        # Open corresponding shapefile, convert to image's CRS, clip to image bounds
        shapes = gpd.read_file(poly_path)
        shapes = shapes.to_crs(rast_img.crs)
        shapes = gpd.clip(shapes, rast_img.bounds)
        # Convert shape coordinates to pixel indices, eliminate neighbors that share the same indices
        for i in range(len(shapes)):
            shape = shapes.loc[i, 'geometry']
            coords = list(shape.exterior.coords)
            pixels = [rast_img.index(x,y) for x,y in coords]
            pixels = [pixels[0]] + [pixels[i] for i in range(1,len(pixels)) if pixels[i] != pixels[i-1]]
            pixels = [(p[1],p[0]) for p in pixels]
            polygons.append(np.array(pixels, dtype='int'))
    polygons = np.array([polygon_to_mask(poly, (width,height)) for poly in polygons], dtype=bool)

    annotated_image = {'basename': basename, 'rgb':rgb_img, 'lidar':las, 'box':boxes, 'poly':polygons}
    
    return annotated_image

  def get_image(self, basename, return_index=False):
    index = self.basenames.index(basename)
    if return_index:
      return index
    else:
      return self.__getitem__(index)