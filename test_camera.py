import cv2
 

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
cap.set(cv2.CAP_PROP_FPS, 30.0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
 
    # if frame is read correctly ret is True
    if ret:
        # Display the resulting frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break
 
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
