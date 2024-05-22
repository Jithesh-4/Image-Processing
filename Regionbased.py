import cv2
import numpy as np
import tkinter as tk

# Load YOLO
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load the COCO class labels
with open('coco.names', "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Initialize tkinter to get screen size
root = tk.Tk()
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
root.destroy()

# Initialize video capture (0 for webcam or provide a video file path)
cap = cv2.VideoCapture(0)

# Set the frame size to full screen
cap.set(cv2.CAP_PROP_FRAME_WIDTH, screen_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, screen_height)

# Define the coordinates of the restricted area (example: a polygon)
restricted_area = np.array([[100, 50], [300, 50], [300, 400], [100, 400]])

while True:
    _, frame = cap.read()
    height, width, channels = frame.shape

    # Detect objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and classes[class_id] == "person":
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    restricted_count = 0
    non_restricted_count = 0

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = (0, 255, 0)
            if cv2.pointPolygonTest(restricted_area, (x + w // 2, y + h // 2), False) >= 0:
                color = (0, 0, 255)
                restricted_count += 1
            else:
                non_restricted_count += 1
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Draw restricted area
    cv2.polylines(frame, [restricted_area], isClosed=True, color=(0, 0, 255), thickness=2)

    # Display counts
    cv2.putText(frame, f"Restricted Area: {restricted_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f"Non-Restricted Area: {non_restricted_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("Frame", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
