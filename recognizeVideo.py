import cv2
import math
from watson_developer_cloud import VisualRecognitionV3
apikey='your-api-key'
thresh='0.3'
videoFile = "videoplayback.mp4"
results=[]

visual_recognition = VisualRecognitionV3(
    '2018-03-19',
    iam_api_key=apikey)

cap = cv2.VideoCapture(videoFile)
frameRate = cap.get(5) #frame rate
while(cap.isOpened()):
    frameId = cap.get(1) #current frame number
    ret, frame = cap.read()
    if (ret != True):
        break
    if (frameId % math.floor(frameRate) == 0):
        resize = cv2.resize(frame, (700, 700), interpolation = cv2.INTER_LINEAR)
        cv2.imshow('frame',frame)
        cv2.imwrite('buffer.png',resize)
        with open('buffer.png', 'rb') as images_file:
            classes = visual_recognition.classify(
                images_file,
                threshold=thresh
                )
        for a in classes['images'][0]['classifiers'][0]['classes']:
            print(a['class'])
            results.append(a['class'])
        cv2.waitKey(1)
cap.release()
print ("Done!")
