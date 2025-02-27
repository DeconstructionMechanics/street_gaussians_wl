from ultralytics import YOLO
import cv2

# yolo11n.pt yolo11s.pt yolo11m.pt yolo11l.pt yolo11x.pt

video_path = 'output/waymo_full_exp/waymo_train_002/trajectory/ours_100000_fog_0.005/color.mp4'

model = YOLO("yolo11x.pt")

# Load video
cap = cv2.VideoCapture(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Output video writer
# output_path = 'output_11x_snow_0.5.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

conf_tot = 0
n = 0

# Process each frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform tracking on the frame
    results = model.track(source=frame, save=False, show=False)

    # Annotate frame with bounding boxes and confidence scores
    for result in results:
        # print(result)
        # print(result.boxes.xyxy)
        for box in result.boxes:
            # print(box)
            confidence, class_id = box.conf, int(box.cls[0]+0.5)
            x1 = box.xyxy[0][0]
            y1 = box.xyxy[0][1]
            x2 = box.xyxy[0][2]
            y2 = box.xyxy[0][3]
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            confidence = float(confidence)

            # Draw bounding box
            color = (0, 255, 0)  # Green color for bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Add confidence text
            label = f"names: {result.names[class_id]}, conf: {confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
            
            if result.names[class_id] == 'car':
                conf_tot += confidence
                n += 1

    # Write annotated frame to output video
    # out.write(frame)

# Release resources
cap.release()
# out.release()

# print(f"Output video saved at {output_path}")
print(f"Average confidence: {conf_tot/n:.2f}")