
# coding: utf-8

# Real time Object Detection

# In[3]:


import cv2
import numpy as np
import streamlit as st

def detect_objects(video_path=None):
    net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
    classes = []
    with open("coco.names", "r") as f:
        classes = f.read().splitlines()

    if video_path is None:
        cap = cv2.VideoCapture(0)  # Using webcam
    else:
        cap = cv2.VideoCapture(video_path)

    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(100, 3))

    while True:
        _, img = cap.read()
        if img is None:
            break
        height, width, _ = img.shape

        blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)
        net.setInput(blob)
        output_layers_names = net.getUnconnectedOutLayersNames()
        layerOutputs = net.forward(output_layers_names)

        boxes = []
        confidences = []
        class_ids = []

        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.2:
                    center_x = int(detection[0]*width)
                    center_y = int(detection[1]*height)
                    w = int(detection[2]*width)
                    h = int(detection[3]*height)

                    x = int(center_x - w/2)
                    y = int(center_y - h/2)

                    boxes.append([x, y, w, h])
                    confidences.append((float(confidence)))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)

        if len(indexes)>0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = str(round(confidences[i],2))
                color = colors[i]
                cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
                cv2.putText(img, label + " " + confidence, (x, y+20), font, 2, (255,255,255), 2)

        cv2.imshow('Image', img)
        key = cv2.waitKey(1)
        if key==27:
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    st.title("Object Detection")

    option = st.radio("Choose Input Source:", ("Webcam", "Upload Video"))

    if option == "Webcam":
        detect_objects()
    else:
        uploaded_file = st.file_uploader("Choose a video file", type=["mp4"])
        if uploaded_file is not None:
            detect_objects(uploaded_file.name)

if __name__ == "__main__":
    main()


# In[ ]:




