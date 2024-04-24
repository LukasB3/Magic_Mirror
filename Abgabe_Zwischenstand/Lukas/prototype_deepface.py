import cv2
from deepface import DeepFace
import logging as logger

# logging
logger.basicConfig(level=logger.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# used for video capturing (make sure to use 1, 0 somehow not working)
cap = cv2.VideoCapture(0)

image_path = "deepface_pic_lukas.jpg"

logger.info("Start Programm")

# keep recording (stay in loop) until Esc is pressed. later this will run in the background until system is shutdown
while True:

    ret, frame = cap.read()

    if not ret:
        logger.warning("No Frame from the camera")
        continue

    # we will be using this to resize the output according our screen size #TODO: find best interpolation
    frame_resized = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

    try:
        result = DeepFace.verify(image_path, frame, enforce_detection=True)

        if result["verified"]:
            # (20, 20) = starting 20 pixels right and 50 pixels down from top left, 1 = scale, (3-tuple) = color, 2 = thickness
            cv2.putText(frame_resized, "Hallo, Lukas!", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame_resized, "Du bist nicht Lukas", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    except Exception as e:

        cv2.putText(frame_resized, "Kein Gesicht erkannt", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow('Input', frame_resized)

    c = cv2.waitKey(1)
    if c == 27:
        logger.info("Closing Camera capturing")  
        break

cap.release()
cv2.destroyAllWindows()

logger.info("Programm terminated")
