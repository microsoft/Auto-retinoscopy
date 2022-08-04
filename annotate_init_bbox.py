import cv2
import os
from utils import *
import csv
import argparse
import yaml

## This code allows the user to selec the initial bounding box where the frame is present.

# Initialize parser
parser = argparse.ArgumentParser()	
parser.add_argument("-v", "--video", help = "Input Video Name")
parser.add_argument("-i", "--input_dir", help = "Path to input dir", default = "./input")
parser.add_argument("-ft", "--frame_type", help = "Enter frame type as current", default = "current")
parser.add_argument("-fs", "--frame_size", help = "Enter frame size specified on the frame nosepin (2,3)", default = "2")

# Read arguments from command line
args = parser.parse_args()
input_dir = args.input_dir
video_name = args.video
frame_type = args.frame_type
frame_size = args.frame_size
video_path = os.path.join(input_dir, video_name)

scaling_factor = 4
yaml_directory = './'
csv_file_name = 'init_bbox.csv'
input_yaml_file = 'input_params.yaml'
with open(os.path.join(yaml_directory, input_yaml_file), 'r') as f:
	params = yaml.safe_load(f)

video = cv2.VideoCapture(video_path)
if not video.isOpened():
	print("Could not open video at the specified path: " + video_path)
else:
	frame_idx = 0
	while (True):
		ok, frame = video.read()
		if not ok:
			break
		else:
			rescaled_image = rescale_image(frame, percent = int(100/scaling_factor))
			bbox, tracker = initialize_bbox_tracker(params, rescaled_image, 'beam', 'CSRT')
			## Rescaling the bbox back according to the scaling factor
			bbox = tuple([x * scaling_factor for x in bbox])
			break

## Adding bounding box details to the init_bbox.csv file
file_exists = os.path.exists(csv_file_name)
if (file_exists):
	print ("Appending data to the existing file")
	with open(csv_file_name, "a", newline='') as f:
		writer = csv.writer(f)
		writer.writerow([video_name, bbox, frame_size, frame_type])
else:
	print ("CSV file does not exists. Created new file")
	with open(csv_file_name, "w", newline='') as f:
		writer = csv.writer(f)
		writer.writerow(['Video_name', 'Init_bbox', 'Frame_size', 'Frame_type'])
		writer.writerow([video_name, bbox, frame_size, frame_type])
