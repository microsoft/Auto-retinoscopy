import numpy as np
import csv
import cv2
import os, sys, time
import yaml
import argparse
from glasses import *
from fiducials import *
from beam import *
from eyes import *
from utils import *

if __name__ == '__main__':

	# Initialize parser
	parser = argparse.ArgumentParser()	
	parser.add_argument("-v", "--video", help = "Input Video")
	
	# Read arguments from command line
	args = parser.parse_args()

	yaml_directory = './'
	input_yaml_file = 'input_params.yaml'
	with open(os.path.join(yaml_directory, input_yaml_file), 'r') as f:
		params = yaml.safe_load(f)

	## Loading params
	params['input']['video'] = args.video
	input_dir = params['input']['directory_path']
	video_name = params['input']['video']
	video_path = os.path.join(input_dir, video_name)
	output_dir = os.path.join(params['output']['directory_path'], video_name.split('.')[0])

	resolution_model = initialize_super_resolution_model('fsrcnn', './FSRCNN_x4.pb', 4)
	create_output_directory_structure(params, output_dir)

	## Reading necessary parameters from csv and storing in params dict
	init_bbox_str, frame_size, frame_type = read_init_bbox_csv(params)
	params['glasses']['frame_size'] = frame_size
	params['glasses']['frame_type'] = frame_type
	init_bbox = list(int(num.strip()) for num in init_bbox_str.replace('(', '').replace(')', '').split(','))
	params['input']['init_bbox'] = init_bbox

	## Save original frames of the video
	if (params['output']['save_original_frames']):
		original_frames_dir = os.path.join(output_dir, params['output_path']['original_frames'])
		print ("Saving original frames")
		save_original_frames(params, video_path, original_frames_dir)

	## Track the bounding box for the beam in the video	
	if (params['input']['load_images_from_video']):
		frame_indices, original_frames, crop_bboxes = track_bbox_in_video(params, video_path, tracking_object = "beam")
	else:
		input_frame_dir = os.path.join(output_dir, params['output_path']['original_frames'])	
		frame_indices, original_frames, crop_bboxes = track_bbox_in_images(params, input_frame_dir, tracking_object = "beam")

	## Run fiducial marker detection for localization
	if (params['output']['run_glasses_detection']):
		fiducials_center_list, fiducials_bbox_list, number_detected_fids_list  = [], [], []
		
		for idx in range(len(crop_bboxes)):
			fiducials_bbox1, number_detected_fids1 = get_fiducial_bbox_from_image(params, original_frames[idx], crop_bboxes[idx], frame_indices[idx], output_dir)
			fiducials_bbox2, number_detected_fids2 = get_fiducial_bbox_from_image(params, original_frames[idx], crop_bboxes[idx], frame_indices[idx], output_dir, default_kernel=(5,5))
			if (number_detected_fids2 > number_detected_fids1):
				fiducials_bbox_list.append(fiducials_bbox2)
				number_detected_fids_list.append(number_detected_fids2)
			else:
				fiducials_bbox_list.append(fiducials_bbox1)
				number_detected_fids_list.append(number_detected_fids1)

		updated_fiducials_bbox_list, fid_type_list = update_fid_bbox_with_tracker(params, original_frames, frame_indices, fiducials_bbox_list, number_detected_fids_list)
		fid_side_len = -1
		
		for idx in range(len(original_frames)):
			fiducials_center = calculate_fiducials_center_intensity_averaging(params, updated_fiducials_bbox_list[idx], fid_type_list[idx], original_frames[idx], frame_indices[idx], output_dir)
			fiducials_center_list.append(fiducials_center)

			if fid_type_list[idx] != 0 and fid_side_len == -1:
				fid_side_len = (updated_fiducials_bbox_list[idx][0][2]+updated_fiducials_bbox_list[idx][0][3])//2

		updated_corners = get_all_corners(params, original_frames, crop_bboxes, frame_indices, updated_fiducials_bbox_list, fid_type_list, output_dir)
		fid_bbox_mean_size, working_distance_list = estimate_working_distance_from_fiducials_corners(params, updated_corners, fid_type_list, original_frames[0].shape)

		## Corrective for perspective distortion in image
		warped_images, warped_fid_centers = fix_perspective_via_homography(params, fid_bbox_mean_size, updated_fiducials_bbox_list, fid_type_list, original_frames, frame_indices, output_dir)
	
	## Run beam segmentation on each frame
	if (params['output']['run_beam_segmentation']):
		left_edge_beam_list, right_edge_beam_list = beam_detection_white_strip(params, warped_images, frame_indices, warped_fid_centers, fid_type_list, output_dir, delta=fid_side_len)

	## Run reflex segmentation on each frame
	if (params['output']['run_reflex_segmentation']):
		input_frame_dir = os.path.join(output_dir, params['output_path']['original_frames'])
		images_list = [img for img in sorted(os.listdir(input_frame_dir)) if (img.endswith(".png"))]

		## Saving the selected frames with good reflex
		selected_frames_dir = os.path.join(output_dir, params['output_path']['selected_frame'])
		scaling_factor = params['scaling_factor']

		left_edge_reflex_list, right_edge_reflex_list, pupil_center_list, mean_pupil_intensity_list = [],[],[],[]
		pupil_circle_cropped_list = []
		center_x_delta, center_y_delta, center_r = [], [], []

		## Running pupil detection on all the frames
		for idx in range(len(frame_indices)):
			## If fid is not detected, add nan	
			if fid_type_list[idx] == 0:
				pupil_center_list.append(None)
				pupil_circle_cropped_list.append(None)
				center_x_delta.append(np.nan)
				center_y_delta.append(np.nan)
				center_r.append(np.nan)
				continue
			
			## Taking the eye crop from original image if fiducials are present
			eyes_bbox, cropped_eye = crop_warped_image(params, warped_images, idx, warped_fid_centers[idx])
			
			pupil_circle = get_pupillary_circle_via_hough(params, cropped_eye.copy(), frame_indices[idx], output_dir)

			## Manually get the point for the specular reflex in first frame and then track wrt to center
			if (pupil_circle is not None):
				pupil_circle_cropped_list.append(pupil_circle)
				pupil_center_list.append([pupil_circle[0][0] + eyes_bbox[0], pupil_circle[0][1] + eyes_bbox[1], pupil_circle[0][2]])

				## Saving the pupil center delta from fid marker
				fid0_x, fid0_y = [int(x) for x in warped_fid_centers[idx][0]]
				pupil_image = warped_images[idx].copy()

				pupil_x, pupil_y, pupil_r = (pupil_circle[0][0] + eyes_bbox[0])*scaling_factor, (pupil_circle[0][1] + eyes_bbox[1])*scaling_factor, (pupil_circle[0][2])*scaling_factor
				center_x_delta.append(pupil_x - fid0_x)
				center_y_delta.append(pupil_y - fid0_y)
				center_r.append(pupil_r)
				try:
					print ('Pupil Frame Selection:', idx, 'Fid', fid0_x, fid0_y, 'Pupil', pupil_x, pupil_y, pupil_r, center_x_delta[idx], center_y_delta[idx])
				except IndexError:
					print ('Pupil Frame Selection:','Index Error', idx, 'Fid', fid0_x, fid0_y, 'Pupil', pupil_x, pupil_y, pupil_r, len(center_x_delta), len(center_y_delta), len(right_edge_reflex_list))

				pupil_image = cv2.circle(pupil_image, (int(pupil_x), int(pupil_y)), int(pupil_r), (255, 255, 255), 1)
				pupil_image = cv2.circle(pupil_image, (int(fid0_x), int(fid0_y)), int(1), (255, 0,0), 1)
				cv2.imwrite(selected_frames_dir + '/pupil_' + '{0:05}'.format(frame_indices[idx]) + '.jpg', pupil_image)		
			
			## No Pupil
			else:
				pupil_center_list.append(None)
				pupil_circle_cropped_list.append(None)
				center_x_delta.append(np.nan)
				center_y_delta.append(np.nan)
				center_r.append(np.nan)

		## Selecting the required frames from the pupil detections and saving the array
		pupil_min_idx, pupil_max_idx, pupil_status = get_pupil_frame_indices(params, np.array(center_x_delta), np.array(center_y_delta), np.array(center_r), selected_frames_dir)

		## Performing reflex detection on the selected frames
		for idx in range(len(frame_indices)):
			flag = 0

			## Current frame does not lie in the range of selected frames
			for i in range(len(pupil_max_idx)):
				if (idx >= pupil_min_idx[i] and idx<= pupil_max_idx[i]):
					flag = 1
					break
			
			## Skip reflex detection if fiducial is not available or idx is not present in the range or pupil is not detected
			if (fid_type_list[idx] == 0 or flag == 0 or pupil_circle_cropped_list[idx] is None):
				left_edge_reflex_list.append(None)
				right_edge_reflex_list.append(None)
				mean_pupil_intensity_list.append(np.nan)
				continue

			eyes_bbox, cropped_eye = crop_warped_image(params, warped_images, idx, warped_fid_centers[idx])
			pupil_circle = pupil_circle_cropped_list[idx]
		
			print ('Reflex detection on the selected frame', idx)
			## Reflex Edges via thresholding and canny
			left_edge, right_edge, mean_pupil_int = get_reflex_edges(params, cropped_eye.copy(), pupil_circle, frame_indices[idx], output_dir, False)

			## Superresolving the image for final reflex edge estimation
			if (params['eyes']['scaling_factor'] == 4):
				super_resolved_image = rescale_image(cropped_eye.copy(), percent=400, super_res = True, sr_model= resolution_model)
			else:
				super_resolved_image = cropped_eye.copy()
			
			## Selecting the gradient channel for intensity averaging
			selected_channel = cv2.cvtColor(super_resolved_image, cv2.COLOR_BGR2LAB)[:,:,1]
			gX = cv2.Sobel(selected_channel, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=3)
			gX = cv2.convertScaleAbs(gX)
			weighted_image = cv2.bitwise_not(selected_channel)
			
			## Calculating final reflex edges based on intensity weighting
			weighted_left_edge, weighted_right_edge = calculated_weighted_reflex_edge(params, gX, super_resolved_image, eyes_bbox, left_edge, right_edge, frame_indices[idx], output_dir)
			left_edge_reflex_list.append(weighted_left_edge)
			right_edge_reflex_list.append(weighted_right_edge)
			mean_pupil_intensity_list.append(mean_pupil_int)

	## Save intermediate and final outputs if flag is True
	if (params['output']['save_tracking_frames']):
		tracking_results_directory = os.path.join(output_dir, params['output_path']['tracking_frames'])
		raw_images_directory = os.path.join(output_dir, params['output_path']['raw_warped_frames'])
		
		for idx in range(0, len(frame_indices)):
			if (fid_type_list[idx] !=0 and warped_images[idx] is not None):
				final_plotting_image = warped_images[idx].copy()

				## Plot fiducials center
				if (params['output']['run_glasses_detection']):
					for fid_idx, fid_center in enumerate(warped_fid_centers[idx]):
						final_plotting_image = cv2.circle(final_plotting_image, (int(fid_center[0]), int(fid_center[1])), 2, (255,255,255), -1)

				## Plot Beam
				if (params['output']['run_beam_segmentation']):
					left_edge, right_edge = left_edge_beam_list[idx], right_edge_beam_list[idx]
					if left_edge is not None:
						for pt in left_edge:
							x,y = pt[1], pt[0]
							final_plotting_image = cv2.circle(final_plotting_image, (int(y), int(x)), 1, (0,0,255), -1)

					if (right_edge is not None):
						for pt in right_edge:
							x,y = pt[1], pt[0]
							final_plotting_image = cv2.circle(final_plotting_image, (int(y), int(x)), 1, (0,255,0), -1)

				if (params['output']['run_reflex_segmentation']):
					scaling_factor = params['scaling_factor']
					## Pupil Plot
					if (pupil_center_list[idx] is not None):
						x,y,r = int(pupil_center_list[idx][1]*scaling_factor), int(pupil_center_list[idx][0]*scaling_factor), int(pupil_center_list[idx][2]*scaling_factor)
						final_plotting_image = cv2.circle(final_plotting_image, (y, x), r, (192, 112, 0), 2)							

					## Left Edge Plot
					if (left_edge_reflex_list[idx] is not None):
						for pt in left_edge_reflex_list[idx]:
							x,y = pt[0]*scaling_factor, pt[1]*scaling_factor
							if (np.isnan(x) or np.isnan(y)):
								pass
							else:
								final_plotting_image = cv2.circle(final_plotting_image, (int(y), int(x)), 1, (0,0,255), -1)

					## Right Edge Plot
					if (right_edge_reflex_list[idx] is not None):
						for pt in right_edge_reflex_list[idx]:
							x,y = pt[0]*scaling_factor, pt[1]*scaling_factor
							if (np.isnan(x) or np.isnan(y)):
								pass
							else:
								final_plotting_image = cv2.circle(final_plotting_image, (int(y), int(x)), 1, (0,255,0), -1)

				cv2.imwrite(tracking_results_directory + '/' + '{0:05}'.format(frame_indices[idx]) + '.jpg', final_plotting_image)
				cv2.imwrite(raw_images_directory + '/' + '{0:05}'.format(frame_indices[idx]) + '.jpg', warped_images[idx])
				cv2.destroyAllWindows
				print ("Saving tracking frames: ", idx)

	## Save intermediate and final numpy arrays if flag set to True
	if (params['output']['save_numpy_output']):
		numpy_output_dir = os.path.join(output_dir, params['output_path']['numpy_path'])
		np.save(numpy_output_dir + '/frames_indices',np.array(frame_indices))

		if (params['output']['run_glasses_detection']):
			np.save(numpy_output_dir + '/fiducials_bbox',np.array(updated_fiducials_bbox_list, dtype=object))
			np.save(numpy_output_dir + '/fiducials_type',np.array(fid_type_list, dtype=object))
			np.save(numpy_output_dir + '/fiducials_center',np.array(fiducials_center_list, dtype=object))
			np.save(numpy_output_dir + '/working_dist',np.array(working_distance_list, dtype=object))			
			np.save(numpy_output_dir + '/warped_fiducials_center',np.array(warped_fid_centers, dtype=object))
		
		if (params['output']['run_beam_segmentation']):
			np.save(numpy_output_dir + '/beam_left_edge',np.array(left_edge_beam_list, dtype=object))
			np.save(numpy_output_dir + '/beam_right_edge',np.array(right_edge_beam_list, dtype=object))

		if (params['output']['run_reflex_segmentation']):
			np.save(numpy_output_dir + '/reflex_left_edge',np.array(left_edge_reflex_list, dtype=object))
			np.save(numpy_output_dir + '/reflex_right_edge',np.array(right_edge_reflex_list, dtype=object))
			np.save(numpy_output_dir + '/pupil_center',np.array(pupil_center_list, dtype=object))
			np.save(numpy_output_dir + '/pupil_intensity',np.array(mean_pupil_intensity_list, dtype=object))
			np.save(numpy_output_dir + '/pupil_min_idx',np.array(pupil_min_idx, dtype=object))
			np.save(numpy_output_dir + '/pupil_max_idx',np.array(pupil_max_idx, dtype=object))
