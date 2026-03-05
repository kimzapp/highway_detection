import cv2
import numpy as np

polygons = []
current_poly = []

def mouse_callback(event, x, y, flags, param):
    global current_poly
    if event == cv2.EVENT_LBUTTONDOWN:
        current_poly.append((x, y))

img = cv2.imread(r"E:\highway_detection\highway.png")
cv2.namedWindow("image")
cv2.setMouseCallback("image", mouse_callback)

while True:
    temp = img.copy()
    for p in current_poly:
        cv2.circle(temp, p, 5, (0,255,0), -1)

    cv2.imshow("image", temp)

    key = cv2.waitKey(1)
    if key == ord("c"):  # complete polygon
        polygons.append(current_poly.copy())
        current_poly = []
    if key == 27:
        break

cv2.destroyAllWindows()