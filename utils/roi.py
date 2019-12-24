import cv2
import numpy as np

def polyAp(img):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert to grayscale
	blur = cv2.blur(gray, (3, 3)) # blur the image
	ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY)
	# Finding contours for the thresholded image
	contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	print(len(contours))
	# create hull array for convex hull points
	hull = []
	# calculate points for each contour
	for i in range(len(contours)):
		# creating convex hull object for each contour
		hull.append(cv2.convexHull(contours[i], False))
	# create an empty black image
	drawing = np.zeros((thresh.shape[0], thresh.shape[1], 3), np.uint8)
	 
	# draw contours and hull points
	for i in range(len(contours)):
		color_contours = (0, 255, 0) # green - color for contours
		color = (255, 0, 0) # blue - color for convex hull
		x, y, w, h = cv2.boundingRect(contours[i])
		# draw ith contour
		# cv2.drawContours(drawing, contours, i, color_contours, 1, 8, hierarchy)
		# draw ith convex hull object
		# cv2.drawContours(drawing, hull, i, color, cv2.FILLED, 8)
		# cv2.fillPoly(drawing, pts =[contours], color=(255,255,255))
	# plt.imshow(drawing)
	# plt.show()
	return drawing