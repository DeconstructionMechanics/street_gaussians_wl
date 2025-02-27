import cv2

# Input video paths
video_A_path = "output/waymo_full_exp/waymo_train_002/trajectory/ours_100000_fog_0.1/color.mp4"
video_B_path = "output/waymo_full_exp/waymo_train_002/trajectory/ours_100000_fog_0.03/color.mp4"
video_C_path = "output/waymo_full_exp/waymo_train_002/trajectory/ours_100000_fog_0.1/depth.mp4"
video_D_path = "output/waymo_full_exp/waymo_train_002/trajectory/ours_100000_fog_0.1/color_gt.mp4"

# video_A_path = "output_11x_snow_1.0.mp4"
# video_B_path = "output_11x_snow_0.75.mp4"
# video_C_path = "output_11x_snow_0.5.mp4"
# video_D_path = "output_11x_gt.mp4"

# Output video path
output_path = "stacked_video_fog_depth.mp4"

# Open the video files
cap_A = cv2.VideoCapture(video_A_path)
cap_B = cv2.VideoCapture(video_B_path)
cap_C = cv2.VideoCapture(video_C_path)
cap_D = cv2.VideoCapture(video_D_path)

# Get video properties (assuming all videos have the same properties)
fps = int(cap_A.get(cv2.CAP_PROP_FPS))
frame_width = int(cap_A.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap_A.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

# Output video properties
out_width = frame_width
out_height = frame_height * 4  # Stacking three videos vertically

# Create the VideoWriter object
out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))

while True:
    # Read frames from each video
    ret_A, frame_A = cap_A.read()
    ret_B, frame_B = cap_B.read()
    ret_C, frame_C = cap_C.read()
    ret_D, frame_D = cap_D.read()
    
    # If any video ends, break
    if not (ret_A and ret_B and ret_C and ret_D):
        break

    # Stack frames vertically
    stacked_frame = cv2.vconcat([frame_A, frame_B, frame_C, frame_D])

    # Write the stacked frame to the output video
    out.write(stacked_frame)

# Release all resources
cap_A.release()
cap_B.release()
cap_C.release()
cap_D.release()
out.release()

print(f"Video saved at {output_path}")