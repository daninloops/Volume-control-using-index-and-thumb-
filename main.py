import cv2
import mediapipe as mp 
import pyautogui
x1=y1=x2=0

#initialise camera
webcam=cv2.VideoCapture(0)

#correct mediapipe hands initialization
hands_module=mp.solutions.hands
my_hands=hands_module.Hands() #the actual hand detection model instance 
drawing_utils=mp.solutions.drawing_utils #utility funtions for drawing hand landmark on image 



while True:
    ret,frame=webcam.read()
    frame_height, frame_width, _ = frame.shape
    if not ret:
        break
   
    #convert bgr to rgb (mediapipe expects rgb input)
    rgb_image=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    output=my_hands.process(rgb_image)
    hands=output.multi_hand_landmarks


    if hands:
        for hand_landmarks in hands:
            drawing_utils.draw_landmarks(frame, hand_landmarks, hands_module.HAND_CONNECTIONS)
            landmarks=hand_landmarks.landmark
            for id, landmark in enumerate(landmarks):
                x=int(landmark.x*frame_width)
                y=int(landmark.y*frame_height)
                if id==8:
                    cv2.circle(frame,center=(x,y),radius=8,color=(127, 96, 251),thickness=3)
                    x1=x
                    y1=y
                if id==4:
                    cv2.circle(frame,center=(x,y),radius=8,color=(128, 0, 128),thickness=3)
                    x2=x
                    y2=y
            dist=((x2-x1)**2 + (y2-y1)**2)**(0.5)//4
            cv2.line(frame,(x1,y1),(x2,y2),(0,255,0),5)
            if dist>50:
                        pyautogui.press("volumeup")
            else:
                        pyautogui.press("volumedown")

            
        
    cv2.imshow("Volume Control using hand gestures",frame)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
webcam.release()    
cv2.destroyAllWindows() 




