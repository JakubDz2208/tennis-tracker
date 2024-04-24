import cv2

def convert_mp4_to_avi(input_file, output_file):
    # Open the input video file
    cap = cv2.VideoCapture(input_file)
    if not cap.isOpened():
        print("Error: Couldn't open input video file")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    # Read and write frames until the end of the video
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    # Release the video capture and writer objects
    cap.release()
    out.release()
    print("Conversion completed successfully.")

# Input and output file paths
input_file = 'tenis.mp4'
output_file = 'output_video.avi'

# Convert .mp4 to .avi
convert_mp4_to_avi(input_file, output_file)