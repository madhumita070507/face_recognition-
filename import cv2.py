import cv2

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Frame not captured, closing program.")
        break  # Exit loop if frame is not captured
    
    cv2.imshow("Webcam Test", frame)  # Just show the webcam

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("User pressed 'q', closing...")
        break

cap.release()
cv2.destroyAllWindows()
