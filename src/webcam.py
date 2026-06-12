import cv2
import torch as pt
from detector import detector
from model import Model
import eye_extraction
import mediapipe as mp

device = pt.device("cuda" if pt.cuda.is_available() else "cpu")
print(f"Working on: {device}")
model=Model(device)
saved_params = pt.load("../models/blink_model_weights.pt")
with pt.no_grad():
    for param, saved_param in zip(model.parameters(), saved_params):
        param.copy_(saved_param)


threshold=3
both_closed=0
right_closed=0
left_closed=0
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    ret, frame = cap.read()
 
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result = detector.detect(mp_image)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = gray[:,:,None]
    eyes=eye_extraction.combined_image(gray,result)
    if eyes is not None:
        eyes_cuda=pt.tensor(eyes,device=device,dtype=pt.float32)
        eyes_cuda=eyes_cuda/255.0
        eyes_cuda=eyes_cuda.unsqueeze(0)
        eyes_cuda=eyes_cuda.unsqueeze(0)
        prediction=pt.argmax(model(eyes_cuda), dim=1).item()
        if prediction==0:
            both_closed=0
            right_closed=0
            left_closed=0
        if prediction==1:
            left_closed+=1
            if left_closed==threshold:
                print("Left_eye_blink_detected")
        if prediction==2:
            right_closed+=1
            if right_closed==threshold:
                print("Right_eye_blink_detected")
        if prediction==3 :
            both_closed+=1
            if both_closed==threshold:
                print("Full_blink_detected")
    
    
    cv2.imshow("frames",eyes)
    if cv2.waitKey(1) == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()