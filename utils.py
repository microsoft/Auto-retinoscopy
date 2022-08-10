import sys, os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv
import math
from glasses import *

## The function creates output directory structure for storing the intermediate and final outputs
def create_output_directory_structure(params, output_dir):
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	for folder in params['output_path']:
		dir_path = os.path.join(output_dir, params['output_path'][folder])
		if not os.path.exists(dir_path):
			os.makedirs(dir_path)
			
	print ('Created output directory path')
	return

## The function returns the init bbox, frame size, and type from the init_bbox file.
def read_init_bbox_csv(params):
	## Reading initial bbox from csv
	file = open(params['input']['init_bbox_file'])
	csvreader = csv.reader(file)
	header = next(csvreader)
	for row in csvreader:
		if (row[0] == params['input']['video']):
			print ("Bbox initialization found")
			return row[1], row[2].strip(), row[3].strip()

	print ("Init bbox not found in the input file.")
	sys.exit()

# The function initializes the super resolution model
def initialize_super_resolution_model(model_name, model_path, model_scale):
	sr = cv2.dnn_superres.DnnSuperResImpl_create()
	sr.readModel(model_path)
	sr.setModel(model_name, model_scale)
	return sr

# The function rescales the given image
def rescale_image(image, size=None, percent=50, super_res = False, sr_model = None):
	if (super_res):
		return sr_model.upsample(image)
	if (percent == 100):
		return image
	if size == None:
		width = int(image.shape[1] * percent/ 100)
		height = int(image.shape[0] * percent/ 100)
		dim = (width, height)
		return cv2.resize(image, dim, interpolation=cv2.INTER_CUBIC)
	else:
		return cv2.resize(image, size, interpolation=cv2.INTER_CUBIC)

## The function saves the original frames of the video without any processing
def save_original_frames(params, video_path, frames_dir):
	video = cv2.VideoCapture(video_path)
	if not video.isOpened():
		print("Could not open video at the path: " + video_path)
		return

	frame_idx = 0
	while (True):
		ok, frame = video.read()
		if not ok:
			break
		cv2.imwrite(frames_dir + '/' + '{0:05}'.format(frame_idx) + '.png', frame)
		frame_idx += 1

		if frame_idx == params['end_frame_index']:
			break
	return

# The function tracks user defined initial bbox in the video
def track_bbox_in_video(params, video_path, tracking_object, tracker_type = 'CSRT'):
	video = cv2.VideoCapture(video_path)
	if not video.isOpened():
		print("Could not open video at path: " + video_path)
		sys.exit()

	original_frames = []
	bboxes = []
	frame_indices = []

	frame_idx = 0
	initialization = False
	scaling_factor = params['scaling_factor'] * 100

	start_frame_index = params['start_frame_index']
	end_frame_index = params['end_frame_index']

	while (True):
		ok, frame = video.read()
		if not ok:
			break

		if frame_idx < start_frame_index:
			print ("Tracking: Not tracking current frame: " + str(frame_idx) + " Start Frame: " + str(start_frame_index))

		else:
			print ("Tracking: Current Frame: " + str(frame_idx))
			frame_original = frame.copy()
			frame = rescale_image(frame, percent = scaling_factor)
			
			if not initialization:
				bbox, tracker = initialize_bbox_tracker(params, frame, tracking_object, tracker_type)
				initialization = True
			else:
				bbox, tracker = update_tracker(params, frame, tracker, tracker_type, tracking_object)

			frame_indices.append(frame_idx)
			original_frames.append(frame_original)
			bboxes.append(bbox)

		frame_idx +=1
		if frame_idx == end_frame_index:
			break

	return frame_indices, original_frames, bboxes

# The function ask user to draw the bounding box and initializes the tracker for subsequent frames
def initialize_bbox_tracker(params, frame, tracking_object, tracker_type):
	if 'init_bbox' in params['input']:
		initial_bbox = params['input']['init_bbox']
	else:
		fromCenter = False
		initial_bbox = cv2.selectROI("Draw bounding box across " + tracking_object + " and press Enter", frame, fromCenter)
		image_patch = frame[initial_bbox[1]:initial_bbox[1]+initial_bbox[3], initial_bbox[0]:initial_bbox[0]+initial_bbox[2],:]
		cv2.imshow('Initialization', image_patch)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

	tracker = initialize_tracker(frame, initial_bbox, tracker_type)
	return initial_bbox, tracker

# The function takes the tracker and updates it for the given frame.
def update_tracker(params, frame, tracker, tracker_type, tracking_object):
	ok, bbox = tracker.update(frame)
	if ok:
		return bbox, tracker
	return initialize_bbox_tracker(params, frame, tracking_object, tracker_type)

# The function returns a tracker based on the trackerType
def create_tracker_by_name(tracker_type):
	(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
	tracker_types = ['KCF', 'MEDIANFLOW', 'CSRT']
	if int(minor_ver) < 3:
		tracker = cv2.Tracker_create()
	else:
		if tracker_type == 'KCF':
			tracker = cv2.TrackerKCF_create()
		elif tracker_type == 'MEDIANFLOW':
			tracker = cv2.TrackerMedianFlow_create()
		elif tracker_type == 'CSRT':
			tracker = cv2.TrackerCSRT_create()
		else:
			tracker = None
			print('Incorrect tracker name. Available trackers are:')
			for t in tracker_types:
				print(t)
	return tracker

# This function initializes the trackers given the image and the bounding boxes
def initialize_tracker(image, bbox, tracker_type):
	tracker = create_tracker_by_name(tracker_type)
	ok = tracker.init(image, tuple(bbox))
	return tracker

## This function returns the canny output by using median of image as lower and upper params
def auto_canny(image, upper_thresh= False, sigma=0.33):
	# Compute the median of the single channel pixel intensities
	v = np.median(image)
	# Apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)

	if (upper_thresh == True):
		return edged, upper
	else:
		return edged

## This function returns canny output using otsu thresholds
def otsu_canny(image):
	blurred = cv2.GaussianBlur(image[:,:,0], (7, 7), 0)
	(otsu_thresh_val, thresh_img) = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
	upper  = otsu_thresh_val
	lower = otsu_thresh_val * 0.5
	edged = cv2.Canny(image, lower, upper)
	return edged, thresh_img, otsu_thresh_val

## Auto canny over RGB images
def accum_canny(image):
	accumEdged = np.zeros(image.shape[:2], dtype="uint8")
	# loop over the blue, green, and red channels, respectively
	for chan in cv2.split(image):
		# blur the channel, extract edges from it, and accumulate the set of edges for the image
		chan = cv2.medianBlur(chan, 11)
		edged = auto_canny(chan)
		accumEdged = cv2.bitwise_or(accumEdged, edged)

	return accumEdged

## The function returns the working distance given the fiducial corners
def estimate_working_distance_from_fiducials_corners(params, fiducial_corners, fiducial_type_list, image_size):
	estimated_working_distance = []
	mean_fid_px_size = []
	for idx in range(0, len(fiducial_type_list)):
		if (fiducial_type_list[idx] == 1 and len(fiducial_corners[idx][0]) == 4):
			mean_fid_size = get_mean_fid_sz_from_corners(params, fiducial_corners[idx][0])
			mean_fid_px_size.append(mean_fid_size)
			object_sensor_ht = (mean_fid_size * params['device']['camera_sensor_height']) / image_size[1]
			working_dst = (params['glasses']['fiducial_real_size_cm'] * params['device']['camera_focal_length']) / object_sensor_ht
			estimated_working_distance.append(working_dst)
		else:
			estimated_working_distance.append(np.nan)
	return np.nanmean(mean_fid_px_size), estimated_working_distance

def get_mean_fid_sz_from_corners(params, fid_corners):
	dist_list = []
	for i in range(len(fid_corners)):
		pt1_x, pt1_y = fid_corners[i][0], fid_corners[i][1]
		for j in range(i+1,len(fid_corners)):
			pt2_x, pt2_y = fid_corners[j][0], fid_corners[j][1]
			dist_list.append(math.sqrt((pt1_x - pt2_x)**2 + (pt1_y - pt2_y)**2))

	dist_list = np.sort(np.array(dist_list))
	dist_list[-1] = dist_list[-1]/math.sqrt(2)
	dist_list[-2] = dist_list[-2]/math.sqrt(2)
	return np.mean(dist_list)

## The function refines the corners for the fid markers
def detect_corners(image, n_corners, maxDist=10, quality=0.01):
	if len(image.shape) == 3:
		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	corners = cv2.goodFeaturesToTrack(image, n_corners, quality, maxDist)
	return np.int0(corners)

## This function returns the corners of the fiducial squares.
def get_all_corners(params, original_frames, crop_bboxes, frame_indices, fiducials_bbox_list, fid_type_list, output_dir, pad=10):
	all_fiducial_corner_list = []
	for idx, frame_idx in enumerate(frame_indices):
		if fid_type_list[idx] == 0:
			all_fiducial_corner_list.append([None, None, None, None, None])
			continue

		curr_frame = cv2.cvtColor(original_frames[idx], cv2.COLOR_BGR2GRAY)
		curr_frame_corner_list = []

		# detect the 4 corners for each fiducial
		for jdx, (x,y,w,h) in enumerate(fiducials_bbox_list[idx]):
			curr_fiducial_corner_list = []
			cropped_bbox = curr_frame[y-pad:y+h+pad, x-pad:x+w+pad]
			print (idx, jdx, fid_type_list[idx], cropped_bbox.shape, x,y,w,h,pad, fiducials_bbox_list[idx])
			try:
				bil_image = cv2.bilateralFilter(cropped_bbox,15,10,10)
				corners = detect_corners(bil_image, 4, maxDist=(w+h)/5)
			
				print (idx, jdx, len(corners))
				# draw red color circles on all corners
				for kdx, i in enumerate(corners):
					xx, yy = i.ravel()
					curr_frame = cv2.circle(curr_frame, (xx+x-pad, yy+y-pad), 1, (255, 0, 0), -1)
					curr_fiducial_corner_list.append([xx+x-pad, yy+y-pad]) # get corners w.r.t. original frame

				curr_frame_corner_list.append(curr_fiducial_corner_list)

			except:
				print ("BILATERAL FAILED | FIDUCIAL SIZE: ", fiducials_bbox_list[idx])
				curr_frame_corner_list = [None, None, None, None, None]

		x,y,w,h = [int(x) for x in crop_bboxes[idx]]
		crop_frame = curr_frame[y:y+h, x:x+w]

		all_fiducial_corner_list.append(curr_frame_corner_list)

		if (output_dir is not None):
			glasses_results_directory = os.path.join(output_dir, params['output_path']['glasses_frames'])
			cv2.imwrite(glasses_results_directory + '/corners_' + '{0:05}'.format(frame_idx) + '.png', crop_frame)

	all_fiducial_corner_list = np.array(all_fiducial_corner_list, dtype=object)
	return all_fiducial_corner_list

def get_scaled_reference_data(params, fid_size, reference_frame, refernce_pts):
	scaling_factor = (fid_size / params['glasses']['fid_bbox_size'])
	scaled_image = rescale_image(reference_frame, percent = scaling_factor * 100)
	scaled_pts = []
	for pt in refernce_pts:
		scaled_pts.append([int(pt[0] * scaling_factor), int(pt[1]* scaling_factor)])
	return scaled_image, np.array(scaled_pts)

## This functions corrects for the perspective changes in the image based on fiducial centers
def fix_perspective_via_homography(params, fid_bbox_mean_size, fid_bbox, fid_type_list, original_frames, frame_indices, output_dir):
	# curr_pts -> Points in current frame, e.g., [list of fiducial centers]
	# reference_pts -> corresponding points in the reference frame
	reference_pts, reference_frame = get_reference_data(params)
	scaled_template_image = False

	warped_images = []
	warped_fid_centers = []
	for idx, frame_idx in enumerate(frame_indices):
		if fid_type_list[idx] == 0:
			warped_images.append(None)
			warped_fid_centers.append([[np.nan, np.nan]] * params['glasses']['number_of_fids'])
			continue

		if (scaled_template_image == False):
			scaled_image, scaled_refernce_pts = get_scaled_reference_data(params, fid_bbox_mean_size, reference_frame, reference_pts)
			scaled_template_image = True

		curr_pts = []
		curr_frame = original_frames[idx]
		for fid in fid_bbox[idx]:
			x,y,w,h = [x for x in fid]
			curr_pts.append([int(x + (w/2)), int(y + h/2)])

		# Calculate Homography
		h, status = cv2.findHomography(np.array(curr_pts), np.array(scaled_refernce_pts))

		## Warp source image to destination based on homography
		fixed_frame = cv2.warpPerspective(curr_frame, h, (scaled_image.shape[1], scaled_image.shape[0]))
		warped_images.append(fixed_frame.copy())
		if (output_dir is not None):
			glasses_results_directory = os.path.join(output_dir, params['output_path']['glasses_frames'])
			for pt in scaled_refernce_pts:
				fixed_frame = cv2.circle(fixed_frame, (int(pt[0]),int(pt[1])), 2, (255, 0, 0), -1)
			cv2.imwrite(glasses_results_directory + '/warped' + '{0:05}'.format(frame_idx) + '.png', fixed_frame)
		
		warped_fid_centers.append(scaled_refernce_pts)

	return warped_images, warped_fid_centers

# The function returns the cropped part of the warped image
def crop_warped_image(params, warped_images, idx, fiducials_center):
	bbox_reverse_scale = 1 / params['scaling_factor']
	fid0_x, fid0_y = [int(x) for x in fiducials_center[0]]
	fid1_x, fid1_y = [int(x) for x in fiducials_center[1]]
	fid4_x, fid4_y = [int(x) for x in fiducials_center[4]]

	eyes_bbox = [fid0_x, fid0_y, abs(fid1_x - fid0_x), abs(fid4_y - fid0_y)]
	eyes_bbox = [int(x*bbox_reverse_scale) for x in eyes_bbox]

	warped_frame = warped_images[idx].copy()
	cropped_eye = warped_frame[eyes_bbox[1]:eyes_bbox[1]+eyes_bbox[3],eyes_bbox[0]:eyes_bbox[0]+eyes_bbox[2],:]

	return eyes_bbox, cropped_eye

## Returns the 3 channel image by stacking single channel 3 times.
def F(I):
	return np.dstack((I, I, I))
