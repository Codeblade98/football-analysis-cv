import cv2

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = [] #initialize an empty list to store the frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    return frames

def save_video(output_frames, output_path):
    fourcc = cv2.VideoWriter_fourcc(*'XVID') # defining output format
    out = cv2.VideoWriter(output_path, fourcc, 24, (output_frames[0].shape[1], output_frames[0].shape[0]))
    for frame in output_frames:
        out.write(frame)
    out.release()