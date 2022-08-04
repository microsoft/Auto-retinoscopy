import numpy as np
import os, sys
import cv2
from utils import *

# This function returns the contours for left and right edge of the beam between the 2 fiducials
def beam_detection_white_strip(params, original_frames, frame_indices, fiducials_center_list, fid_type_list, output_dir, delta=5):
	beam_results_directory = os.path.join(output_dir, params['output_path']['beam_frames'])

	thresh_cutoff = 120
	left_edge_list, right_edge_list = [], []

	for idx, frame in enumerate(original_frames):
		left_edge_list.append(None)
		right_edge_list.append(None)
		print('BEAM FRAME:', idx, fid_type_list[idx])

		## Skip the frame as no fiducial present
		if fid_type_list[idx] == 0:
			continue

		fid0_x, fid0_y = [int(x) for x in fiducials_center_list[idx][0]]
		fid1_x, fid1_y = [int(x) for x in fiducials_center_list[idx][1]]

		print('BEAM: FRAME: ', idx, ' Fiducial Positions: ', fid0_x, fid0_y, fid1_x, fid1_y, delta)

		# Bilateral filtering to remove false edges
		frame = cv2.bilateralFilter(frame,15,80,80)

		frame_red = frame.copy()
		frame_red[:,:,0] = frame_red[:,:,2]
		frame_red[:,:,1] = frame_red[:,:,2]

		# Crop between the fiducial markers
		frame_crop = frame_red[fid0_y-delta//2:fid0_y+delta//2, fid0_x+delta//2:fid1_x-delta//2]

		## OTSU edge detection inside the cropped portion
		frame_crop = cv2.fastNlMeansDenoisingColored(frame_crop,None,10,10,7,21) # has untracked params
		frame_edge_crop_canny, thresh_img, T = otsu_canny(frame_crop.copy())

		if T < thresh_cutoff:
			continue
		
		## Auto canny for edge detection
		frame_edge_crop_otsu = auto_canny(thresh_img)
		sobelx = cv2.Sobel(frame_crop,cv2.CV_64F,1,0,ksize=5) 
		
		## Accumulator canny
		frame_edge_crop_accum = accum_canny(frame_crop)

		## Taking an OR operation between different canny outputs
		frame_edge_crop_merged = cv2.bitwise_or(frame_edge_crop_canny, frame_edge_crop_otsu)
		frame_edge_crop_merged = cv2.bitwise_or(frame_edge_crop_accum, frame_edge_crop_merged)
	
		## Contour detection
		contours, hierarchy = cv2.findContours(frame_edge_crop_merged, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

		## Detecting left and right contours based on direction of gradients
		for jdx, cnt in enumerate(contours):
			cnt_grad_sum_x = sobelx[contours[jdx][:,0,1], contours[jdx][:,0,0]].sum() # x gradient
			print("BEAM: {} contour sum".format(jdx+1), cnt_grad_sum_x)
			if cnt_grad_sum_x < 0:
				# Right edge
				frame_crop = cv2.drawContours(frame_crop, [contours[jdx]], -1, (0, 255, 0), 2)
				right_edge_list[idx] = np.array(contours[jdx])[:,0,:]
				right_edge_list[idx] += [fid0_x+delta//2, fid0_y-delta//2]
			else:
				# Left edge
				frame_crop = cv2.drawContours(frame_crop, [contours[jdx]], -1, (0, 0, 255), 2)
				left_edge_list[idx] = np.array(contours[jdx])[:,0,:]
				left_edge_list[idx] += [fid0_x+delta//2, fid0_y-delta//2]
		
		## Plotting the fiducials and detected edges
		cv2.imwrite(beam_results_directory+'/inter_'+'{0:05}'.format(frame_indices[idx]) + '.png', frame_crop)
		frame[fid0_y-delta//2:fid0_y+delta//2, fid0_x+delta//2:fid1_x-delta//2] = frame_crop

		frame = cv2.rectangle(frame, (fid0_x-delta//2, fid0_y-delta//2), (fid0_x+delta//2, fid0_y+delta//2), (0,0,255), 2)
		frame = cv2.rectangle(frame, (fid1_x-delta//2, fid1_y-delta//2), (fid1_x+delta//2, fid1_y+delta//2), (0,0,255), 2)

		cv2.imwrite(beam_results_directory + '/' + '{0:05}'.format(frame_indices[idx]) + '.png', frame)

	return left_edge_list, right_edge_list
