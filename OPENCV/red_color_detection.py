# only Red color detection
import cv2
import numpy as np

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:   # If no frame is captured
        break

    # Convert to HSV
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define RED range
    low_red = np.array([161, 155, 84])
    high_red = np.array([179, 255, 255])

    # Create mask
    red_mask = cv2.inRange(hsv_frame, low_red, high_red)

    # Extract red color from frame
    red = cv2.bitwise_and(frame, frame, mask=red_mask)

    # Show original and red detected
    cv2.imshow("Frame", frame)
    cv2.imshow("Red Only", red)

    # Press ESC (27) to exit
    key = cv2.waitKey(1)
    if key == 27:
        break

# Release resources
cap.release()   # releases the webcam resource.
cv2.destroyAllWindows()  # closes all OpenCV windows
