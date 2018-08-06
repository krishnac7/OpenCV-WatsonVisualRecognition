import numpy as np
import cv2,json,time
from watson_developer_cloud import VisualRecognitionV3
apikey='your-api-key'
thresh='0.6'
waitTime=15.0
visual_recognition = VisualRecognitionV3(
    '2018-03-19',
    iam_api_key=apikey)
cap = cv2.VideoCapture(0)
results=[]
while(True):
    start_time = time.time()
    ret, frame = cap.read()
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
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    time.sleep(waitTime - time.time() + start_time)
cap.release()
cv2.destroyAllWindows()
