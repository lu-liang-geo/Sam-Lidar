# Sam-Lidar
This code accompanies the paper **Leveraging SAM 2 and LiDAR for Automated Individual Tree Crown Delineation: A Comparative Evaluation of Prompting Methods**, currently under review in the journal _Information Geography_.

In the paper and code, we test the efficacy of the pre-trained, promptable Segment Anything Model 2 (SAM 2) for individual tree crown delineation (ITC delineation) using realistic prompting methods. These prompting methods consist of manual and automatically-drawn bounding boxes, with the optional inclusion of point prompts extracted from LiDAR point clouds. We evaluate these methods on a diverse subset of 592 trees from the NeonTreeEvaluation dataset using a suite of object-based and pixel-based metrics. We find that the selection of bounding box prompts significantly impacts the number of trees SAM 2 correctly segments, with the best performing prompts resulting in 92% F1 for object-based metrics and the worst performing prompts resulting in 35% F1. We also find that LiDAR data slightly improves SAM 2’s delineations, increasing object-based F1 by 1-4%. Pixel-based metrics, which are calculated only for trees with accurate bounding box prompts, are consistently between 78-87% F1, showing that, given a correct prompt, SAM 2 can accurately delineate tree crowns.

The following ReadMe walks you through how to install our code and the relevant libraries for SAM 2 and outside tree detection / segmentation models DeepForest and Detectree2. It then explains how to run the code to recreate our experiments and visualizations on the data used in the paper, as well as how to run the models on your own data.

## Installation
To run the code locally, you will need to install code from this repository, along with code and weights for the outside models SAM 2, DeepForest, and Detectree2. Alternatively, if you run the attached Jupyter notebooks on Google Colab, there is an option to install the relevant libraries directly into Colab at the beginning of your session, though this may take up to 10 minutes per session.

DeepForest can be installed using the Python Package Index (pip), but this code, SAM 2, and Detectree2 must be downloaded from Github.

To install our code, navigate to the desired folder, clone this repository, and install required libraries using the following:

`git clone https://github.com/lu-liang-geo/Sam-Lidar.git`
`pip install -r Sam-Lidar/requirements.txt`

For SAM 2, clone the repository and then download the model weights into the desired folder using the following:

`git clone https://github.com/facebookresearch/segment-anything-2.git`
`wget -q -P /PATH/TO/SAVE/WEIGHTS https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt`

For Detectree2, clone the repository and download the model weights using the following:

`git clone https://github.com/PatBall1/detectree2.git`
`wget -q -P /PATH/TO/SAVE/WEIGHTS https://zenodo.org/records/12773341/files/230103_randresize_full.pth`

