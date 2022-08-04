import numpy as np
import cv2

curr_right_fiducial_coord_2 = [[[0,0], [9,0], [1,3], [9,4.5], [0,6]],
								[[-9,0], [0,0], [-8,3], [0,4.5], [-9,6]]]
curr_left_fiducial_coord_2 = [[[0,0], [9,0], [0,4.5], [8,3], [9,6]],
								[[-9,0], [0,0], [-9,4.5], [-1,3], [0,6]]]


## Return the frame structure based on Frame Type, Old/New, Left or Right
def get_frame_struct_coord(eye_side = 'L', frame_number = '2', frame_type = 'current'):
	if (eye_side == 'R' and frame_number == '2' and frame_type == 'current'):
		return curr_right_fiducial_coord_2
	elif (eye_side == 'L' and frame_number == '2' and frame_type == 'current'):
		return curr_left_fiducial_coord_2

## Returns the reference image and template image for the given frame
def get_reference_data(params):
	if (params['glasses']['frame_size'] == '2' and params['glasses']['frame_type'] == 'current'):
		if (params['input']['video'].split('_')[0] == 'R'):
			reference_pts = np.array(params['glasses']['fid_centers_right_curr_2'])
		else:
			reference_pts = np.array(params['glasses']['fid_centers_left_curr_2'])
		template_image = cv2.imread('template_2_curr.png')
	return reference_pts, template_image
