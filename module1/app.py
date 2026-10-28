# the goal for the module 1 is that open camera and read frames and releasing it 

import cv2

video_captured = cv2.VideoCapture(0)
while True:
    ret,frame =video_captured.read()
    flipped_frame =cv2.flip(frame,1)
    gray_flipped_frame = cv2.cvtColor(flipped_frame ,cv2.COLOR_BGR2GRAY)
    if not ret:
        print("Video is not loading")
    cv2.imshow("webcam footage",gray_flipped_frame)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        print("program is executed")
        break
video_captured.release() 
cv2.destroyAllWindows()   
