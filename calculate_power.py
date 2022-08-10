import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import csv
from matplotlib.pyplot import figure
from utils import *


def selected_frame_data(auto_start_frame, auto_end_frame, left_beam, right_beam, left_reflex, right_reflex):
	## Sampling data when left edge is moving
	## Start Frame, LE beam, RE beam, LE reflex, RE reflex, End Frame, LE beam, RE beam, LE reflex, RE reflex
	final_data = []
	for pass_number in range(len(auto_start_frame)):
		start_frame = auto_start_frame[pass_number]
		end_frame = auto_end_frame[pass_number]
		start_frame_data = [start_frame, left_beam[start_frame], right_beam[start_frame], left_reflex[start_frame], right_reflex[start_frame]]
		end_frame_data = [end_frame, left_beam[end_frame], right_beam[end_frame], left_reflex[end_frame], right_reflex[end_frame]]
		final_data.append(start_frame_data + end_frame_data)

	return final_data

def plot_graphs(params, frame_indices, fiducials_center_list, fid_type_list, left_edge_beam_list, right_edge_beam_list, left_edge_reflex_list, right_edge_reflex_list, center_reflex_list, pupil_line_offset, pupil_min_idx, pupil_max_idx):
	scaling_factor = params['scaling_factor']

	## Calculating the deltas per frame
	left_edge_beam_dist, right_edge_beam_dist = [], []
	left_edge_reflex_dist, right_edge_reflex_dist = [], []
	
	for idx in range(len(left_edge_beam_list)):
		flag = 0

		## Check if frame idx lies in the required range
		for i in range(len(pupil_min_idx)):
			if (idx >= pupil_min_idx[i] and idx<= pupil_max_idx[i]):
				flag = 1
				break

		## Skip if fiducials are not detected
		if ((fid_type_list[idx] == 0) or flag == 0):
			left_edge_beam_dist.append(np.nan)
			right_edge_beam_dist.append(np.nan)
			left_edge_reflex_dist.append(np.nan)
			right_edge_reflex_dist.append(np.nan)
			continue

		fid0_x, fid0_y = [int(x) for x in fiducials_center_list[idx][0]]

		## TODO Define shape of image... Dont hardcode
		# h,w,_ = 1080,  1920, 3
		h,w,_ = 2160,  3840, 3

		## Beam Plotting
		hori_line_image = np.zeros((h,w)).astype(np.uint8)
		hori_line_image = cv2.line(hori_line_image, (0,fid0_y), (w, fid0_y), (255, 255, 255), 1)
		
		## Check if beam left edge is detected
		if left_edge_beam_list[idx] is None:
			left_edge_beam_dist.append(np.nan)
		else:
			left_beam_edge = np.zeros((h,w)).astype(np.uint8)
			left_beam_edge[left_edge_beam_list[idx][:,1], left_edge_beam_list[idx][:,0]] = 255

			## Taking intersection of beam edges and horizontal line
			left_beam_edge = hori_line_image*left_beam_edge
			y,x = np.argwhere(left_beam_edge>0)[:,0], np.argwhere(left_beam_edge > 0)[:,1]
			y,x = np.average(y), np.average(x)

			left_edge_beam_dist.append(np.abs(fid0_x-x))
		
		## Check if beam right edge is detected
		if right_edge_beam_list[idx] is None:
			right_edge_beam_dist.append(np.nan)
		else:
			right_beam_edge = np.zeros((h,w)).astype(np.uint8)
			right_beam_edge[right_edge_beam_list[idx][:,1], right_edge_beam_list[idx][:,0]] = 255

			## Taking intersection of beam edges and horizontal line
			right_beam_edge = hori_line_image*right_beam_edge
			y,x = np.argwhere(right_beam_edge>0)[:,0], np.argwhere(right_beam_edge > 0)[:,1]
			y,x = np.average(y), np.average(x)

			right_edge_beam_dist.append(np.abs(fid0_x-x))

		## Reflex Plotting
		## Check if pupil center is detected
		if (center_reflex_list[idx] is None):
			left_edge_reflex_dist.append(np.nan)
			right_edge_reflex_dist.append(np.nan)
			continue

		r = center_reflex_list[idx][2]*scaling_factor

		## Check if reflex left edge is detected
		if (left_edge_reflex_list[idx] is None or len(left_edge_reflex_list[idx]) == 0):
			left_edge_reflex_dist.append(np.nan)
		else:
			left_temp = np.array(left_edge_reflex_list[idx])*scaling_factor
			x_temp = left_temp[:,0]
			y_temp = left_temp[:,1]
			mask1 = np.logical_and(np.logical_and(x_temp>=0, x_temp<=h), np.logical_and(y_temp>=0, y_temp<=w))
			mask2 = np.logical_and(mask1, (np.abs(x_temp - (fid0_y+pupil_line_offset)) <= params['power_calculation']['horizontal_line_width_reflex']))
			x_temp_final = x_temp[mask2]
			y_temp_final = y_temp[mask2]

			y,x = np.nanmean(y_temp_final), np.nanmean(x_temp_final)
			
			left_edge_reflex_dist.append(np.abs(fid0_x-y))
			
		## Check if reflex right edge is detected
		if (right_edge_reflex_list[idx] is None or len(right_edge_reflex_list[idx]) == 0):
			right_edge_reflex_dist.append(np.nan)
		else:
			right_temp = np.array(right_edge_reflex_list[idx])*scaling_factor
			x_temp = right_temp[:,0]
			y_temp = right_temp[:,1]
			mask1 = np.logical_and(np.logical_and(x_temp>=0, x_temp<=h), np.logical_and(y_temp>=0, y_temp<=w))
			mask2 = np.logical_and(mask1, (np.abs(x_temp - (fid0_y+pupil_line_offset)) <= params['power_calculation']['horizontal_line_width_reflex']))

			x_temp_final = x_temp[mask2]
			y_temp_final = y_temp[mask2]

			y,x = np.nanmean(y_temp_final), np.nanmean(x_temp_final)

			right_edge_reflex_dist.append(np.abs(fid0_x-y))
	
	return left_edge_beam_dist, right_edge_beam_dist, left_edge_reflex_dist,right_edge_reflex_dist

## Returns the first and last timestamps from the given list left and right sides in sorted order
def select_first_last_timestamp(left_side_1, left_side_2, right_side_1, right_side_2):
	start_frame, end_frame = [], []

	for i in range(0, len(left_side_1)):
		if ((len(left_side_1[i]) > 0) and (len(left_side_2[i]) > 0)):
			if (left_side_1[i][0] < left_side_2[i][0]):
				start_frame.append(left_side_1[i][0])
				end_frame.append(left_side_2[i][-1])
			else:
				start_frame.append(left_side_2[i][0])
				end_frame.append(left_side_1[i][-1])

	for i in range(0, len(right_side_1)):
		if ((len(right_side_1[i]) > 0) and (len(right_side_2[i]) > 0)):
			if (right_side_1[i][0] < right_side_2[i][0]):
				start_frame.append(right_side_1[i][0])
				end_frame.append(right_side_2[i][-1])
			else:
				start_frame.append(right_side_2[i][0])
				end_frame.append(right_side_1[i][-1])

	return sorted(start_frame), sorted(end_frame)

## Function detects the start and end frame timestamps based on defined heuristic
def auto_frame_selection(params, fid_centers, pupil_center_list, pupil_min_idx, pupil_max_idx, reflex_right_edge_list, reflex_left_edge_list):
	left_side_1, left_side_2 = [[], [], [], []], [[], [], [], []]
	right_side_1, right_side_2 = [[], [], [], []], [[], [], [], []]

	for idx in range(0, len(pupil_center_list)):
		flag = 0
		for current_pass in range(min(len(pupil_max_idx), 4)):
			if (idx >= pupil_min_idx[current_pass] and idx<= pupil_max_idx[current_pass] and pupil_center_list[idx] is not None):
				flag = 1
				break
		
		if (flag == 0):
			pass
		else:
			fid_offset = fid_centers[idx][0]
			pupil_center, radius = pupil_center_list[idx][0] - fid_offset[0], pupil_center_list[idx][2]
			
			if (reflex_left_edge_list[idx] is not None):
				if (len(reflex_left_edge_list[idx]) > 0):
					x_center = np.nanmedian(np.array(reflex_left_edge_list[idx])[:,1]) - fid_offset[0]
					per = (x_center - pupil_center + radius)/(2*radius)
					if (per >= params['power_calculation']['start_per1'] and per <= params['power_calculation']['start_per2']):
						left_side_1[current_pass].append(idx)
					if (per >= params['power_calculation']['end_per1'] and per <= params['power_calculation']['end_per2']):
						left_side_2[current_pass].append(idx)

			if (reflex_right_edge_list[idx] is not None):
				if (len(reflex_right_edge_list[idx]) > 0):
					x_center = np.nanmedian(np.array(reflex_right_edge_list[idx])[:,1]) - fid_offset[0]
					per = (x_center - pupil_center + radius)/(2*radius)
					if (per >= params['power_calculation']['start_per1'] and per <= params['power_calculation']['start_per2']):
						right_side_1[current_pass].append(idx)
					if (per >= params['power_calculation']['end_per1'] and per <= params['power_calculation']['end_per2']):
						right_side_2[current_pass].append(idx)
	
	start_frame, end_frame = select_first_last_timestamp(left_side_1, left_side_2, right_side_1, right_side_2)

	return start_frame, end_frame

## This function returns the vertical offset of lines from fiducial centers
def get_pupil_lines_offset(params, fiducials_center_list, fid_type_list, center_reflex_list, pupil_min_idx, pupil_max_idx):
	vertical_dist = []

	for idx in range(len(center_reflex_list)):
		## Skip reflex in case it lies outside the range
		flag = 0

		for i in range(len(pupil_min_idx)):
			if (idx >= pupil_min_idx[i] and idx<= pupil_max_idx[i]):
				flag = 1
				break

		if (fid_type_list[idx] == 0 or center_reflex_list[idx] is None or flag == 0):
			continue
		
		fid0_x, fid0_y = fiducials_center_list[idx][0][0], fiducials_center_list[idx][0][1]
		center_x, center_y, pupil_r = center_reflex_list[idx][0]*params['scaling_factor'], center_reflex_list[idx][1]*params['scaling_factor'], center_reflex_list[idx][2]*params['scaling_factor']
		pupil_points = np.linspace(center_y - params['power_calculation']['pupil_vertical_allowed_range']*pupil_r, center_y + params['power_calculation']['pupil_vertical_allowed_range']*pupil_r, params['power_calculation']['number_lines'])
		vertical_dist = [np.abs(x-fid0_y) for x in pupil_points]
		return vertical_dist

	return None

## This function returns the power given the working distance for the neutralization cases
def get_power_from_working_distance(params, working_dist):
	return (-1)/working_dist

## This function returns the power given the reflex/beam ratio from the formulation
def get_power_from_ratio(params, ratio, working_dist):
	nr = (ratio * (params['power_calculation']['effective_light_distance'] + working_dist)) - params['power_calculation']['effective_light_distance']
	dr = ratio * working_dist * (params['power_calculation']['effective_light_distance'] + working_dist)
	return ((-1) * nr / dr)
