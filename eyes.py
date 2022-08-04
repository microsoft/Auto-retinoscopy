import numpy as np
import cv2
import os, sys
from utils import *
from skimage import morphology as morp
from matplotlib import pyplot as plt
from numpy import unravel_index
np.set_printoptions(threshold=sys.maxsize)


resolution_model = initialize_super_resolution_model('fsrcnn', './FSRCNN_x4.pb', 4)

## The function returns the center and radius of the pupil in the cropped eye image
def get_pupillary_circle_via_hough(params, image, frame_idx=None, output_dir = None):
	## Pupil min and max radius
	min_radius = params['eyes']['pupil_min_radius']
	max_radius = params['eyes']['pupil_max_radius']

	## Red channel for pupil detection
	single_channel = image[:,:,2]
	blurred_frame = cv2.medianBlur(single_channel, 3)
	gaussian_blurred = cv2.GaussianBlur(blurred_frame, (3,3), 0)

	edges, upper_thresh = auto_canny(gaussian_blurred.copy(), upper_thresh=True)

	temp_image = single_channel.copy()

	flag = 0
	## Changing hough circle parameters till we get just 1 circle representing the pupil
	for param in range(params['eyes']['pupil_hough_param2_max_value'], params['eyes']['pupil_hough_param2_min_value'], -1):
		for dp in np.linspace(1,2,11):
			
			## Find the Hough Circles with varying dp till we get a single hough circle
			circles = cv2.HoughCircles(gaussian_blurred, cv2.HOUGH_GRADIENT, dp, minDist = 50, param1 = upper_thresh, param2 = param, minRadius= min_radius, maxRadius=max_radius)
			if circles is not None:
				circles = np.round(circles[0, :]).astype("int")
				if (len(circles) == 1):
					flag = 1
					for (x, y, r) in circles:
						print ("Index: " + str(frame_idx) + " Param value:" + str(param) + " Number of circles:" + str(len(circles)) + " Radius:" + str(r))
						
						## Plot the estimated hough circle with pupillary margin
						temp_image = cv2.circle(temp_image, (x, y), r, (255, 255, 255), 1)
				break
		if (flag == 1):
			break
	
	final_image = np.hstack((single_channel, gaussian_blurred, edges, temp_image))
	if (output_dir is not None):
		eyes_results_directory = os.path.join(output_dir, params['output_path']['eyes_frames'])
		cv2.imwrite(eyes_results_directory + '/pupil_' + '{0:05}'.format(frame_idx) + '.png', final_image)		

	return circles

## The function returns the list of points for left and right edges
## If no edge found, it returns None for both the edges
def get_reflex_edges(params, image, pupil_circle, frame_idx, output_dir, erode = False):
	## If no pupil detected return None
	right_edge, left_edge = None, None
	if (pupil_circle is None):
		return left_edge, right_edge, None
	
	## Mask out the pupillary region
	scaling_factor = params['eyes']['scaling_factor']
	pupil_mask = np.zeros_like(image)
	pupil_mask = cv2.circle(pupil_mask, (pupil_circle[0][0], pupil_circle[0][1]), pupil_circle[0][2] + params['eyes']['pupillary_margin'], (255, 255, 255), -1)
	pupil_image = cv2.bitwise_and(image, pupil_mask)

	pupil_mask_without_margin = np.zeros_like(image)
	pupil_mask_without_margin = cv2.circle(pupil_mask_without_margin, (pupil_circle[0][0], pupil_circle[0][1]), pupil_circle[0][2], (255, 255, 255), -1)
	grayscale_pupil_mask  = cv2.cvtColor(cv2.bitwise_and(image, pupil_mask_without_margin), cv2.COLOR_BGR2GRAY)
	pupil_pixel_locs = np.where(grayscale_pupil_mask > 0)
	mean_pupil_intensity = np.nanmean(grayscale_pupil_mask[pupil_pixel_locs])

	## Selecting LAB color space
	selected_image_type = cv2.cvtColor(pupil_image, cv2.COLOR_BGR2LAB)
	single_channel = selected_image_type[:,:,1]

	## Rescaling the image if set in yaml file
	if (scaling_factor == 4):
		pupil_rescaled = rescale_image(pupil_image.copy(), percent=scaling_factor * 100, super_res = True, sr_model= resolution_model)
		rescaled_channel = rescale_image(single_channel, percent=scaling_factor * 100	, super_res = True, sr_model= resolution_model)
	else:
		pupil_rescaled = pupil_image.copy()
		rescaled_channel = single_channel.copy()

	## Otsu thresholding for the reflex
	(_, thresh) = cv2.threshold(rescaled_channel, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

	## Erosion to disconnect from specular reflex based on param
	if (erode):
		se = morp.square(9,dtype='uint8')
		erosion = morp.erosion(thresh, se)
	else:
		erosion = thresh.copy()

	contours, _ = cv2.findContours(np.uint8(erosion), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	sorted_contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

	## Filling all the contours except the reflex (to fill the specular reflex)
	if (len(sorted_contours) > 3):
		for ct in sorted_contours[3:]:
			erosion = cv2.drawContours(erosion, [ct], -1, (0,0, 0), -1)
	
	## Dilating back to compensate for the erosion
	if (erode):
		erosion = morp.dilation(erosion, se)
	selected_contour = None

	## Finding the boundary for the edge using canny on the reflex contour
	final_contours, _ = cv2.findContours(erosion, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	temp_image = np.zeros_like(rescaled_channel)
	if (len(final_contours) > 2):
		selected_contour = sorted(final_contours, key=lambda x: cv2.contourArea(x), reverse=True)[2]
		temp_image = cv2.drawContours(temp_image, [selected_contour], -1, (255,255, 255), -1)

	## Removing the specular reflex in the image by traversing the image column wise
	temp_specular = remove_specular_reflex(params, temp_image.copy(), pupil_circle[0], frame_idx)
	edges = auto_canny(temp_specular)

	## Finding all the points on the reflex boundary
	pixelpoints = np.transpose(np.nonzero(edges))

	## Finding the right and left edge of reflex
	pixel_count_contour = np.zeros_like(rescaled_channel)
	if (selected_contour is not None):
		for pt in pixelpoints:
			r,c = pt[0], pt[1]
			pixel_count_contour[int(r)][int(c)] = pixel_count_contour[int(r)][int(c)] + 1

		left_edge, right_edge = [], []
		for r in range(pixel_count_contour.shape[0]):
			sorted_non_zero_ind = sorted(np.nonzero(pixel_count_contour[r])[0])
			if (len(sorted_non_zero_ind) > 0):
				left_edge.append([r, sorted_non_zero_ind[0]])
				right_edge.append([r, sorted_non_zero_ind[-1]])

		## Plotting the	edges
		## Blue - Left Edge, Red - Right Edge
		for pt in right_edge:
			r,c = pt[0], pt[1]
			pupil_rescaled = cv2.circle(pupil_rescaled, (int(c), int(r)), 0, (0,255,0), 3)

		for pt in left_edge:
			r,c = pt[0], pt[1]
			pupil_rescaled = cv2.circle(pupil_rescaled, (int(c), int(r)), 0, (0,0,255), 3)

	final_image = np.hstack((F(rescaled_channel), F(erosion), F(temp_image), F(temp_specular), F(edges), pupil_rescaled))
	if (output_dir is not None):
		eyes_results_directory = os.path.join(output_dir, params['output_path']['eyes_frames'])
		cv2.imwrite(eyes_results_directory + '/relfex_edges_' + '{0:05}'.format(frame_idx) + '.png', final_image)		

	return left_edge, right_edge, mean_pupil_intensity

def calculated_weighted_reflex_edge(params, weighted_image, plotting_image, eyes_bbox, left_edge, right_edge, frame_idx = None, output_dir = None):
	final_image = plotting_image.copy()
	
	scaling_factor = params['eyes']['scaling_factor']
	weighted_left_edge, weighted_right_edge = [], []
	avg_win = params['eyes']['averaging_window']
	if (left_edge is not None and right_edge is not None):

		r,c = np.shape(weighted_image)
		c_ = np.linspace(0,c,c+1)

		## Left Edge
		for pt in left_edge:
			try:
				x, y = pt[0], pt[1]
				roi_grid_y = c_[y - avg_win: y + avg_win + 1]
				weights = weighted_image[x, y - avg_win: y + avg_win + 1]
				weighted_y = weights * roi_grid_y
				cY = (np.sum(weighted_y) / np.sum(weights))
				cX = x
				weighted_left_edge.append([cX/scaling_factor + eyes_bbox[1], cY/scaling_factor + eyes_bbox[0]])
				if (np.isnan(cX) or np.isnan(cY)):
					pass
				else:
					final_image = cv2.circle(final_image, (int(cY), int(cX)), 1, (0,0,255), 3)
			except:
				pass


		## Right Edge
		for pt in right_edge:
			try:
				x,y = pt[0], pt[1]
				roi_grid_y = c_[y - avg_win: y + avg_win + 1]
				weights = weighted_image[x, y - avg_win: y + avg_win + 1]
				weighted_y = weights * roi_grid_y
				cY = (np.sum(weighted_y) / np.sum(weights))
				cX = x
				weighted_right_edge.append([cX/scaling_factor + eyes_bbox[1], cY/scaling_factor + eyes_bbox[0]])
				if (np.isnan(cX) or np.isnan(cY)):
					pass
				else:
					final_image = cv2.circle(final_image, (int(cY), int(cX)), 1, (0,255,0), 3)
			except:
				pass

	else:
		return None, None


	eyes_results_directory = os.path.join(output_dir, params['output_path']['eyes_frames'])
	cv2.imwrite(eyes_results_directory + '/weighted_' + '{0:05}'.format(frame_idx) + '.png', final_image)
	return weighted_left_edge, weighted_right_edge

## This function removes the part of reflex if it occupies less than x% of the vertical space inside the pupil.
def remove_specular_reflex(params, image, pupil_coord ,idx):
	scaling_factor = params['eyes']['scaling_factor']
	x, y, r = pupil_coord[0]*scaling_factor, pupil_coord[1]*scaling_factor, pupil_coord[2]*scaling_factor
	pupil_mask = np.zeros_like(image)
	pupil_mask = cv2.circle(pupil_mask, (x, y), r, (255, 255, 255), -1)
	for c in range(image.shape[1]):
			reflex_column_pixels = sorted(np.nonzero(image[:,c])[0])
			pupil_column_pixels = sorted(np.nonzero(pupil_mask[:,c])[0])
			if (len(reflex_column_pixels) <= len(pupil_column_pixels) * params['eyes']['reflex_vertical_column_percent']):
				image[:,c] = 0
	return image

## This function returns the indices when pupil is detected correctly based on the center and radius of detected pupil wrt fiducial.	
def get_pupil_frame_indices(params, pupil_x_delta, pupil_y_delta, pupil_r, output_dir):
	f = plt.figure(figsize=(15,5))
	ax1 = f.add_subplot(111)
	
	## Create a 2D histogram plot of x and y coordinates of center wrt fiducial
	histogram_bar_size = params['eyes']['histogram_bar_size']
	n_bins_x = int((np.nanmax(pupil_x_delta) - np.nanmin(pupil_x_delta))/histogram_bar_size)
	n_bins_y = int((np.nanmax(pupil_y_delta) - np.nanmin(pupil_y_delta))/histogram_bar_size)

	(count, bins_x, bins_y, im) = ax1.hist2d(pupil_x_delta[~np.isnan(pupil_x_delta)], pupil_y_delta[~np.isnan(pupil_y_delta)], bins=(n_bins_x, n_bins_y), cmap=plt.cm.jet)
	f.colorbar(im, ax=ax1)
	
	plt.savefig(output_dir + '/histogram_2D.png')
	bin_centers_x = (bins_x[:-1] + bins_x[1:]) / 2
	bin_centers_y = (bins_y[:-1] + bins_y[1:]) / 2

	x_ind, y_ind = unravel_index(count.argmax(), count.shape)
	reqd_x_delta = int(bin_centers_x[x_ind])
	reqd_y_delta = int(bin_centers_y[y_ind])

	pupil_status = [False] * len(pupil_x_delta)
	min_x, max_x = reqd_x_delta - (histogram_bar_size), reqd_x_delta + (histogram_bar_size)
	min_y, max_y = reqd_y_delta - (histogram_bar_size), reqd_y_delta + (histogram_bar_size)


	## Check the delta x and delta y of pupil center to find pupil radius
	temp_radius_list = []
	for idx in range(len(pupil_x_delta)):
		if (pupil_x_delta[idx] >= min_x and pupil_x_delta[idx] <= max_x and pupil_y_delta[idx] >= min_y and pupil_y_delta[idx] <= max_y):
			temp_radius_list.append(pupil_r[idx])
			print ('Pupillary Radius', idx, pupil_r[idx])

	median_pupil_radius = np.nanmedian(np.array(temp_radius_list))
	print ('Median Pupil radius: ', median_pupil_radius)
	pupil_margin  = params['eyes']['median_pupil_radius_margin']
	min_pupil = (1 - pupil_margin) * median_pupil_radius
	max_pupil = (1 + pupil_margin) * median_pupil_radius

	for idx in range(len(pupil_x_delta)):
		## Check the delta x and delta y of pupil center
		if (pupil_x_delta[idx] >= min_x and pupil_x_delta[idx] <= max_x and pupil_y_delta[idx] >= min_y and pupil_y_delta[idx] <= max_y):
			## Check the pupil radius
			if (min_pupil <= pupil_r[idx] <= max_pupil):
				pupil_status[idx] = True

	pupil_pass_sep = params['eyes']['pupil_pass_separator']
	ind = [i for i in range(len(pupil_status)) if pupil_status[i] is True]

	pupil_min_ind = []
	pupil_max_ind = []

	last_ind = ind[0]
	for idx, curr_ind in enumerate(ind):
		if (idx == 0):
			pupil_min_ind.append(curr_ind)
		elif (curr_ind - last_ind >= pupil_pass_sep):
			pupil_min_ind.append(curr_ind)
			pupil_max_ind.append(last_ind)
			last_ind = curr_ind
		elif (idx == len(ind) - 1):
			pupil_max_ind.append(curr_ind)
		else:
			last_ind = curr_ind

	print ('Pupil_min_idx', pupil_min_ind)
	print ('Pupil_max_idx', pupil_max_ind)
	return pupil_min_ind, pupil_max_ind, pupil_status
