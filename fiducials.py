from contextlib import ContextDecorator
import numpy as np
import os
import cv2
from scipy.signal.signaltools import order_filter
from utils import *
import math
from scipy import interpolate
from glasses import *

resolution_model = initialize_super_resolution_model('fsrcnn', './FSRCNN_x4.pb', 4)

## This function returns the ordered list of fiducial bbox for the given frame in the given bbox
def get_fiducial_bbox_from_image(params, frame, bbox, frame_idx = None, output_dir = None, default_kernel = (3,3)):
	bbox = [int(x) for x in bbox]
	cropped_frame = frame[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]]
	gray_frame = cv2.cvtColor(cropped_frame.copy(), cv2.COLOR_BGR2GRAY)

	## Image preprocessing
	blurred_frame = cv2.medianBlur(gray_frame, 5) 
	clahe = cv2.createCLAHE(clipLimit=1, tileGridSize=(13,13))
	equalized = clahe.apply(blurred_frame)
	final_image = gray_frame.copy()
	sharpened_frame = cv2.GaussianBlur(equalized, (5,5), 0)
	
	final_image = np.hstack((final_image, equalized, sharpened_frame))

	## Auto canny for edge detection
	edges = auto_canny(equalized.copy())

	## Dilating small gaps in edge map
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, default_kernel)
	edges = cv2.dilate(edges, kernel)

	## Contour detection for finding squares
	contours = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	contours = contours[0] if len(contours) == 2 else contours[1]

	fid_area = []
	fid_contour = []

	temp_image = cropped_frame.copy()
	for c in contours:	
		# Approximate the contour
		peri = cv2.arcLength(c, True)
		area = cv2.contourArea(c)
		approx = cv2.approxPolyDP(c, 0.05* peri, True)
		_,_,w,h = cv2.boundingRect(c)
		extent = float(area)/(w*h)

		# Find all square contours
		if len(approx) == params['glasses']['fiducials_side']  and cv2.isContourConvex(approx) and extent > params['glasses']['fiducials_extent']:
			if area > params['glasses']['fiducials_min_area'] and 1 - params['glasses']['square_tolerance'] < math.fabs((peri / 4) ** 2) / area < 1 + params['glasses']['square_tolerance']:
				fid_area.append(area)
				fid_contour.append(c)
	
	## Suppressing detected squares not representing fiducials based on their area
	suppressed_fiducial_markers, suppressed_area = suppress_markers_area(params, fid_contour, fid_area, "min")
	mean_fid_size_px = math.sqrt(np.mean(suppressed_area))
	fiducial_bbox = []

	for fid_mar in suppressed_fiducial_markers:
		x,y,w,h = cv2.boundingRect(fid_mar)
		fiducial_bbox.append([x,y,w,h])
		rect = cv2.minAreaRect(fid_mar)
		box = cv2.boxPoints(rect)
		box = np.int0(box)
		temp_image = cv2.drawContours(temp_image,[box],0,(255,0,0), 2)
	
	# Finding the ordering among the detected fiducials
	final_fid_markers = []
	reordered_markers, detections = find_best_reordering(params, fiducial_bbox, mean_fid_size_px, [bbox[3], bbox[2]])
	print ("Total detections in the image: " + str(detections))
	for idx, fid_mar in enumerate(reordered_markers):
		if fid_mar[0] is not np.nan:
			x,y,w,h = fid_mar[0], fid_mar[1], fid_mar[2], fid_mar[3]
			temp_image = cv2.rectangle(temp_image, (x, y), (x + w, y + h), (0,0,255), 2)
			temp_image = cv2.putText(temp_image, str(idx), (int(x + w/2), int(y + h/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1, cv2.LINE_AA)
			## Updating to global frame of reference
			final_fid_markers.append([x + bbox[0], y + bbox[1], w, h])
		else:
			final_fid_markers.append([np.nan, np.nan, np.nan, np.nan])

	final_image = np.hstack((final_image, edges))

	## Saving intermediate frames and detected fiducials
	if (output_dir is not None):
		glasses_results_directory = os.path.join(output_dir, params['output_path']['glasses_frames'])
		cv2.imwrite(glasses_results_directory + '/' + '{0:05}'.format(frame_idx) + '.png', temp_image)		
		cv2.imwrite(glasses_results_directory + '/edges_' + '{0:05}'.format(frame_idx) + '.png', final_image)		
		
	return final_fid_markers, detections

## Suppress fiducial markers representing the same physical marker
## Find the center of markers and check if it lies between another marker. If so return the marker based on type
def suppress_markers_area(params, contours, area, type = "max"):
	suppressed_cont = []
	suppressed_area = []
	
	for idx, c in enumerate(contours):
		x1,y1,w1,h1 = cv2.boundingRect(c)
		x_mid = x1 + (w1 / 2)
		y_mid = y1 + (h1 / 2)
		flag = 0
	
		if (len(suppressed_cont) == 0):
			suppressed_area.append(area[idx])
			suppressed_cont.append(c)
		else:
			for selected_idx, selected_mar in enumerate(suppressed_cont):
				x2, y2, w2, h2 = cv2.boundingRect(selected_mar)
				if (x_mid >= x2 and x_mid < x2 + w2) and (y_mid >= y2 and y_mid < y2 + h2):
					if (type == "max") and area[idx] > suppressed_area[selected_idx]:
						suppressed_cont[selected_idx] = selected_mar
						suppressed_area[selected_idx] = area
					flag = 1
					break

			if flag == 0:
				suppressed_cont.append(c)
				suppressed_area.append(area[idx])
	
	## Using median area of detected fiducials for removing erroneous detections
	max_area_med = np.median(suppressed_area)*20
	min_area_med = 0

	# remove spurious detections after suppression
	suppressed_cont_clean, suppressed_area_clean = [], []
	for idx, c in enumerate(suppressed_cont):
		print(idx, suppressed_area[idx])
		if suppressed_area[idx] < min_area_med or suppressed_area[idx] > max_area_med:
			continue
		suppressed_cont_clean.append(suppressed_cont[idx]); suppressed_area_clean.append(suppressed_area[idx]);

	return suppressed_cont_clean, suppressed_area_clean

## Returns the distance of fid center from the given point
def fid_center_from_point(fid_mar, point):
	x, y, w, h = fid_mar[0], fid_mar[1], fid_mar[2], fid_mar[3]
	x_mid, y_mid = x + (w / 2), y + (h / 2)
	hypot = math.sqrt((x_mid - point[0])**2 + (y_mid - point[1])**2)
	return hypot

## Finding the ordering and number of correctly detected fiducials given the Upper Left Fiducial
## Uses the known structure of the frame
def reorder_detected_fiducials_coords_wrt_lu(params, fid_markers, mean_fid_size, current_fid, fid_pattern):
	ordering = [[np.nan, np.nan, np.nan, np.nan]] * params['glasses']['number_of_fids']
	ordering[0] = current_fid
	detected = 1

	x, y, w, h = current_fid[0], current_fid[1], current_fid[2], current_fid[3]
	x_mid, y_mid = x + (w / 2), y + (h / 2)

	## Ordering other fiducials wrt upper left
	for fid_mar in fid_markers:
		x2, y2, w2, h2 = fid_mar[0], fid_mar[1], fid_mar[2], fid_mar[3]
		x_mid2, y_mid2 = x2 + (w2 / 2), y2 + (h2 / 2)

		## Finding #1
		x_cond = abs(abs(x_mid2-x_mid) - abs(fid_pattern[0][1][0]-fid_pattern[0][0][0])*mean_fid_size) <= mean_fid_size
		y_cond = abs(abs(y_mid2-y_mid) - abs(fid_pattern[0][1][1]-fid_pattern[0][0][1])*mean_fid_size) <= mean_fid_size
		if x_cond and y_cond:
			ordering[1] = fid_mar
			detected = detected + 1

		## Finding #2
		x_cond = abs(abs(x_mid2-x_mid) - abs(fid_pattern[0][2][0]-fid_pattern[0][0][0])*mean_fid_size) <= mean_fid_size
		y_cond = abs(abs(y_mid2-y_mid) - abs(fid_pattern[0][2][1]-fid_pattern[0][0][1])*mean_fid_size) <= mean_fid_size
		if x_cond and y_cond:
			ordering[2] = fid_mar
			detected = detected + 1

		## Finding #3
		x_cond = abs(abs(x_mid2-x_mid) - abs(fid_pattern[0][3][0]-fid_pattern[0][0][0])*mean_fid_size) <= mean_fid_size
		y_cond = abs(abs(y_mid2-y_mid) - abs(fid_pattern[0][3][1]-fid_pattern[0][0][1])*mean_fid_size) <= mean_fid_size
		if x_cond and y_cond:
			ordering[3] = fid_mar
			detected = detected + 1

		## Finding #4
		x_cond = abs(abs(x_mid2-x_mid) - abs(fid_pattern[0][4][0]-fid_pattern[0][0][0])*mean_fid_size) <= mean_fid_size
		y_cond = abs(abs(y_mid2-y_mid) - abs(fid_pattern[0][4][1]-fid_pattern[0][0][1])*mean_fid_size) <= mean_fid_size
		if x_cond and y_cond:
			ordering[4] = fid_mar
			detected = detected + 1

	return ordering, detected

## Finding the ordering and number of correctly detected fiducials given the Upper Right Fiducial
## Uses the known structure of the frame
def reorder_detected_fiducials_coords_wrt_ru(params, fid_markers, mean_fid_size, current_fid, fid_pattern):
	allowed_tol = params['glasses']['fiducials_distance_tolerance']
	ordering = [[np.nan, np.nan, np.nan, np.nan]] * params['glasses']['number_of_fids']
	ordering[1] = current_fid
	detected = 1

	x, y, w, h = current_fid[0], current_fid[1], current_fid[2], current_fid[3]
	x_mid, y_mid = x + (w / 2), y + (h / 2)
	## Ordering other fiducials wrt upper left
	for fid_mar in fid_markers:
		x2, y2, w2, h2 = fid_mar[0], fid_mar[1], fid_mar[2], fid_mar[3]
		x_mid2, y_mid2 = x2 + (w2 / 2), y2 + (h2 / 2)
		
		## Finding #0
		x_cond = abs(abs(x_mid2-x_mid) - abs(fid_pattern[1][0][0]-fid_pattern[1][1][0])*mean_fid_size) <= mean_fid_size
		y_cond = abs(abs(y_mid2-y_mid) - abs(fid_pattern[1][0][1]-fid_pattern[1][1][1])*mean_fid_size) <= mean_fid_size
		if x_cond and y_cond:
			ordering[0] = fid_mar
			detected = detected + 1

		## Finding #2
		x_cond = abs(abs(x_mid2-x_mid) - abs(fid_pattern[1][2][0]-fid_pattern[1][1][0])*mean_fid_size) <= mean_fid_size
		y_cond = abs(abs(y_mid2-y_mid) - abs(fid_pattern[1][2][1]-fid_pattern[1][1][1])*mean_fid_size) <= mean_fid_size
		if x_cond and y_cond:
			ordering[2] = fid_mar
			detected = detected + 1

		## Finding #3
		x_cond = abs(abs(x_mid2-x_mid) - abs(fid_pattern[1][3][0]-fid_pattern[1][1][0])*mean_fid_size) <= mean_fid_size
		y_cond = abs(abs(y_mid2-y_mid) - abs(fid_pattern[1][3][1]-fid_pattern[1][1][1])*mean_fid_size) <= mean_fid_size
		if x_cond and y_cond:
			ordering[3] = fid_mar
			detected = detected + 1

		## Finding #4
		x_cond = abs(abs(x_mid2-x_mid) - abs(fid_pattern[1][4][0]-fid_pattern[1][1][0])*mean_fid_size) <= mean_fid_size
		y_cond = abs(abs(y_mid2-y_mid) - abs(fid_pattern[1][4][1]-fid_pattern[1][1][1])*mean_fid_size) <= mean_fid_size
		if x_cond and y_cond:
			ordering[4] = fid_mar
			detected = detected + 1

	return ordering, detected

## Finds the ordering of the detected fiducials by assuming known Upper Left and Upper Right Fiducial. Returns the ordering with maximum number of detections.
## 0: Top-left, 1: Top-Right and so on...
def find_best_reordering(params, fid_markers, mean_fid_size, img_size):
	best_ordering, best_detections = [[np.nan, np.nan, np.nan, np.nan]] * params['glasses']['number_of_fids'], 0
	# eye_side = params['input']['video'].split('_')[0]
	if (params['input']['video'].split('_')[0] == 'R'):
		eye_side = 'R'
	else:
		eye_side = 'L'
	fid_pattern = get_frame_struct_coord(eye_side = eye_side, frame_number = params['glasses']['frame_size'], frame_type = params['glasses']['frame_type'])

	## If no fiducial detected return
	if len(fid_markers) == 0:
		return best_ordering, best_detections
	
	## Upper Left Fiducial
	best_dist = img_size[0]**2 + img_size[1]**2
	for fid_mar in fid_markers:
		dist = fid_center_from_point(fid_mar, [0,0])
		if (dist < best_dist):
			left_upper = [fid_mar[0], fid_mar[1], fid_mar[2], fid_mar[3]]
			best_dist = dist

	print (fid_markers)
	print (fid_pattern)
	lu_ordering, lu_detections = reorder_detected_fiducials_coords_wrt_lu(params, fid_markers, mean_fid_size, left_upper, fid_pattern)
	if (lu_detections == params['glasses']['number_of_fids']):
		return lu_ordering, lu_detections

	## Upper Right Fiducial
	best_dist = img_size[0]**2 + img_size[1]**2
	for fid_mar in fid_markers:
		dist = fid_center_from_point(fid_mar, [img_size[1],0])
		if (dist < best_dist):
			right_upper = [fid_mar[0], fid_mar[1], fid_mar[2], fid_mar[3]]
			best_dist = dist
	
	ru_ordering, ru_detections = reorder_detected_fiducials_coords_wrt_ru(params, fid_markers, mean_fid_size, right_upper, fid_pattern)
	if (ru_detections == params['glasses']['number_of_fids']):
		return ru_ordering, ru_detections

	## Return ordering wrt case in which more fiducials are detected
	if (lu_detections >= ru_detections):
		return lu_ordering, lu_detections
	return ru_ordering, ru_detections

## Updates the missing fiducials wrt a tracker across the list
def update_fid_bbox_with_tracker(params, original_frames, frame_indices, fiducials_bbox_list, number_detected_fids_list, tracker_type = 'CSRT'):
	empty_fid = [[np.nan, np.nan, np.nan, np.nan]] * params['glasses']['number_of_fids']

	## 0 - Missing Fiducials, 1 - Reinitialized Fiducial, 2 - Updated Missing Fiducial with tracker
	fid_type = []
	initialization = False
	updated_fid_bbox_list = []
	tracker_list = []
	tracker_reinitialized = 0
	tracker_updated = 0

	for idx in range(len(original_frames)):
		frame_copy = original_frames[idx].copy()

		if not initialization:
			## Initialize the tracker on the fiducials for the first time
			if (number_detected_fids_list[idx] == params['glasses']['number_of_fids']):
				print ("Trackers initialized for the first time at frame number: " + str(frame_indices[idx]))
				for fid_idx in range(params['glasses']['number_of_fids']):
					tracker = initialize_tracker(frame_copy, fiducials_bbox_list[idx][fid_idx], tracker_type)
					tracker_list.append(tracker)
				initialization = True
				updated_fid_bbox_list.append(fiducials_bbox_list[idx])
				fid_type.append(1)
			else:
				## Less fiducials detected.... SKIP the frame
				print ("Trackers initialized missed at frame number: " + str(frame_indices[idx]))
				updated_fid_bbox_list.append(empty_fid.copy())
				fid_type.append(0)
		else:
			## Re-initialize the trackers based on new detections
			if (number_detected_fids_list[idx] == params['glasses']['number_of_fids']):
				print ("Trackers reinitialized at frame number: " + str(frame_indices[idx]))
				for fid_idx in range(params['glasses']['number_of_fids']):
					tracker = initialize_tracker(frame_copy, fiducials_bbox_list[idx][fid_idx], tracker_type)
					tracker_list[fid_idx] = tracker
				updated_fid_bbox_list.append(fiducials_bbox_list[idx])
				tracker_reinitialized = tracker_reinitialized + 1
				fid_type.append(1)

			else:
				# Update the fiducials based on tracker which are missing
				print ("Trackers updated at frame number: " + str(frame_indices[idx]))
				updated_current_fids = fiducials_bbox_list[idx] # current fiducial list
				# only update the missing (NAN) BBOX
				for fid_idx in range(params['glasses']['number_of_fids']):
					_, bbox = tracker_list[fid_idx].update(frame_copy)
					bbox = [int(x) for x in bbox]
					curr_x, curr_y, curr_w, curr_h = updated_current_fids[fid_idx]
					if np.isnan(updated_current_fids[fid_idx][0]):
						updated_current_fids[fid_idx] = bbox
					elif not (bbox[0] < curr_x < bbox[0]+bbox[2] and bbox[1] < curr_y < bbox[1]+bbox[3]):
						updated_current_fids[fid_idx] = bbox

				updated_fid_bbox_list.append(updated_current_fids)
				tracker_updated = tracker_updated + 1
				fid_type.append(2)
	print ("Tracker reinitialized for: " + str(tracker_reinitialized))
	print ("Tracker updated for: " + str(tracker_updated))
	return updated_fid_bbox_list, fid_type

## Calculates the fiducial center using intensity based averaging
def calculate_fiducials_center_intensity_averaging(params, fid_markers, fid_type, frame, frame_idx = None, output_dir = None):
	frame_copy = frame.copy()
	gray_frame = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY)

	r,c = np.shape(gray_frame)
	r_ = np.linspace(0,r,r+1)
	c_ = np.linspace(0,c,c+1)
	x_m, y_m = np.meshgrid(c_, r_, sparse=False, indexing='xy')

	inverted_gray_frame = cv2.bitwise_not(gray_frame)
	fid_centers = [[np.nan, np.nan]] * params['glasses']['number_of_fids']
	for idx, marker in enumerate(fid_markers):
		print(idx, marker)
		if ((marker[0] is not np.nan) and (fid_type !=0)):
			x,y,w,h = marker[0], marker[1], marker[2], marker[3]
			weights = inverted_gray_frame[y:y+h, x:x+w]

			roi_grid_x = x_m[y:y+h,x:x+w]
			roi_grid_y = y_m[y:y+h,x:x+w]

			weighted_x = weights * roi_grid_x
			weighted_y = weights * roi_grid_y

			cX = (np.sum(weighted_x) / np.sum(weights))
			cY = (np.sum(weighted_y) / np.sum(weights))

			fid_centers[idx] = [cX, cY]

			if (fid_type == 1):
				frame_copy = cv2.rectangle(frame_copy, (x, y), (x + w, y + h), (0,0,255), 1)
			elif (fid_type == 2):
				frame_copy = cv2.rectangle(frame_copy, (x, y), (x + w, y + h), (0,255,0), 1)
	
	## Save the final output image
	if (output_dir is not None):
		glasses_results_directory = os.path.join(output_dir, params['output_path']['glasses_frames'])
		cv2.imwrite(glasses_results_directory + '/final_' + '{0:05}'.format(frame_idx) + '.png', frame_copy)		
			
	return fid_centers
