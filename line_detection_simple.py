import cv2
import numpy as np


# CANNY EDGE DETECTION
def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = 5
    blur = cv2.GaussianBlur(gray, (kernel, kernel), 0)
    canny_ = cv2.Canny(blur, 50, 150)
    return canny_


# REGION OF INTEREST
def region_of_interest(image):
    height = image.shape[0]
    width = image.shape[1]
    mask = np.zeros_like(image)
    triangle = np.array([[(200, height), (800, 350), (1200, height)]], np.int32)
    cv2.fillPoly(mask, triangle, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

# RHOUGH LINES
def houghLines(image):
    houghLines_ = cv2.HoughLinesP(image, 2, np.pi / 180, 100,
                                  np.array([]), minLineLength=40, maxLineGap=5)
    return houghLines_

# DISPLAY LINES
def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            # cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
            cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return image

# DISPLAY LINES AVERAGES
def display_lines_avaerage(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 10)
    return line_image

# CALCULATE END POINTS OF LINES
def make_points(image, lineSI):
    slope, intercept = lineSI
    height = image.shape[0]
    y1 = int(height)
    y2 = int(y1*3.0/5)
    x1 = int((y1-intercept)/slope)
    x2 = int((y2-intercept)/slope)
    return [[x1, y1, x2, y2]]

# CALCULATE AVERAGE OF LINE POINT
def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope = fit[0]
            intercept = fit[1]

            if slope < 0:
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))

    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)

    left_line = make_points(image, left_fit_average)
    right_line = make_points(image, right_fit_average)

    return np.array(([left_line, right_line]))


capture = cv2.VideoCapture('test1.mp4')

while True:
    ret, frame = capture.read()
    if not ret:
        break

    canny_output = canny(frame)
    masked_output = region_of_interest(canny_output)
    lines = houghLines(masked_output)
    average_lines = average_slope_intercept(frame, lines)
    line_image = display_lines_avaerage(frame, average_lines)
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    cv2.imshow('LineDetection', combo_image)

    if cv2.waitKey(27) | 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
