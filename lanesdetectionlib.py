import cv2
import glob
import pickle
import numpy as np


def select_yellow(image, bgr):
    if bgr:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    else:
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lower = np.array([20,60,60])
    upper = np.array([38,174, 250])
    mask = cv2.inRange(hsv, lower, upper)
    return mask

def select_white(image, bgr):
    if bgr:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    lower = np.array([202,202,202])
    upper = np.array([255,255,255])
    mask = cv2.inRange(image, lower, upper)
    return mask

def comb_thresh(image,bgr):
    yellow = select_yellow(image,bgr)
    white = select_white(image,bgr)
    combined_binary = np.zeros_like(yellow)
    combined_binary[(yellow >= 1) | (white >= 1)] = 1
    return combined_binary
    
""" Takes a BGR image, converts it to HLS space and apply a treshold filter on the S channel"""
def apply_s_treshold(image, tresh):
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    s_channel = hls[:,:,2]
    s_filtered = np.zeros_like(s_channel)
    s_filtered[(s_channel >= tresh[0]) & (s_channel <= tresh[1])] = 1
    return s_filtered

""" Takes a BGR image, calculates Sobel_x operator (x gradient) and apply a treshold filter on it"""
def apply_x_gradient_treshold(image, tresh, sobel_kernel):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # First convert to gray, since gradient makes sense only on one channel
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)  # Take the derivative in x
    abs_sobel_x = np.absolute(sobel_x)             # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobel_x/np.max(abs_sobel_x)) # Scale it to have (0,255) range
    sobel_filtered = np.zeros_like(scaled_sobel)
    sobel_filtered[(scaled_sobel >= tresh[0]) & (scaled_sobel <= tresh[1])] = 1
    return sobel_filtered

""" Takes a BGR image, calculates Sobel_y operator (y gradient) and apply a treshold filter on it"""
def apply_y_gradient_treshold(image, tresh, sobel_kernel):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # First convert to gray, since gradient makes sense only on one channel
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)  # Take the derivative in x
    abs_sobel_y = np.absolute(sobel_y)             # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobel_y/np.max(abs_sobel_y)) # Scale it to have (0,255) range
    sobel_filtered = np.zeros_like(scaled_sobel)
    sobel_filtered[(scaled_sobel >= tresh[0]) & (scaled_sobel <= tresh[1])] = 1
    return sobel_filtered

""" Takes a BGR image, calculates the gradient magnitude and apply a treshold filter on it"""
def apply_mag_gradient_treshold(image, mag_thresh, sobel_kernel):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # First convert to gray, since gradient makes sense only on one channel
    # Apply x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    sobel = np.sqrt(sobelx ** 2 + sobely ** 2)
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*sobel/np.max(sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel > mag_thresh[0]) & (scaled_sobel < mag_thresh[1])] = 1
    # Return the result
    return binary_output

""" Takes a gray image, calculates the gradient direction and apply a treshold filter on it"""
def apply_dir_gradient_treshold(image, thresh, sobel_kernel):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # First convert to gray, since gradient makes sense only on one channel
    # Apply x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    
    # Error statement to ignore division and invalid errors
    with np.errstate(divide='ignore', invalid='ignore'):
        absgraddir = np.absolute(np.arctan(sobely/sobelx))
        dir_binary =  np.zeros_like(absgraddir)
        dir_binary[(absgraddir > thresh[0]) & (absgraddir < thresh[1])] = 1
    # Return the result
    return dir_binary

def filter_roi(img, vertices):
    mask = np.zeros_like(img)
    # Handle both grayscale and colored images
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    roi_image = cv2.bitwise_and(img, mask)
    return roi_image

def append_lane_coordinates(lanes, lane_center, x_lane, y_lane, l_r, x_window, y_bottom, y_top, dire, window_ind, i,app):
    # Append coordinates to the left lane arrays
    lane_arr = lanes[(lanes[:,1]>=lane_center-x_window) & (lanes[:,1]<lane_center+x_window) &
                                 (lanes[:,0]<=y_bottom) & (lanes[:,0]>=y_top)]
    x_lane += lane_arr[:,1].flatten().tolist()
    y_lane += lane_arr[:,0].flatten().tolist()
    if l_r == 0:         # Left
        if not math.isnan(np.mean(lane_arr[:,1])):
            left_lane.windows[window_ind, i] = np.mean(lane_arr[:,1])
            app[i+2,0] = np.mean(lane_arr[:,1])
        else:
            app[i+2,0] = app[i+1,0] + dire
            left_lane.windows[window_ind, i] = app[i+2,0]
    else:               # Right
        if not math.isnan(np.mean(lane_arr[:,1])):
            right_lane.windows[window_ind, i] = np.mean(lane_arr[:,1])
            app[i+2,1] = np.mean(lane_arr[:,1])
        else:
            app[i+2,1] = app[i+1,1] + dire
            right_lane.windows[window_ind, i] = app[i+2,1]
    return x_lane,  y_lane, app



def evaluate_poly(indep, poly_coeffs):
    return poly_coeffs[0]*indep**2 + poly_coeffs[1]*indep + poly_coeffs[2]

def highlight_lane_line_area(mask_template, left_poly, right_poly, start_y=0, end_y =720):
    area_mask = mask_template
    for y in range(start_y, end_y):
        left = evaluate_poly(y, left_poly)
        right = evaluate_poly(y, right_poly)
        area_mask[y][int(left):int(right)] = 1

    return area_mask

def lane_poly(yval, poly_coeffs):
    """Returns x value for poly given a y-value.
    Note here x = Ay^2 + By + C."""
    return poly_coeffs[0]*yval**2 + poly_coeffs[1]*yval + poly_coeffs[2]


def draw_poly(img, poly, poly_coeffs, steps, color=[255, 0, 0], thickness=10, dashed=False):
    img_height = img.shape[0]
    pixels_per_step = img_height // steps

    for i in range(steps):
        start = i * pixels_per_step
        end = start + pixels_per_step

        start_point = (int(poly(start, poly_coeffs=poly_coeffs)), start)
        end_point = (int(poly(end, poly_coeffs=poly_coeffs)), end)

        if dashed == False or i % 2 == 1:
            img = cv2.line(img, end_point, start_point, color, thickness)

    return img


