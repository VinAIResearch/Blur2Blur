import cv2

def crop_video(input_video_path, output_video_path, crop_position):
    # Open the input video
    cap = cv2.VideoCapture(input_video_path)
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Format for the output video
    
    # Set up the output video writer
    # Adjust 'width' and 'height' to your desired output dimensions
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (crop_position[2], crop_position[3]))
    
    while True:
        # Read each frame
        ret, frame = cap.read()
        if not ret:
            break
        
        # Crop the frame
        x, y, width, height = crop_position
        cropped_frame = frame[y:y+height, x:x+width]
        
        # Write the cropped frame to the output video
        out.write(cropped_frame)
        cv2.imwrite("temp.png", cropped_frame)
    
    # Release resources
    # cap.release()
    # out.release()
    # cv2.destroyAllWindows()
    # exit(0)

# Example usage
input_video_path = 'Extreme.mp4'
output_video_path = 'Extreme_1.mp4'
# crop_position = (50, 0, 600, 1920) 
crop_position = (0, 190, 1920, 700)  # Example crop position (x, y, width, height)

crop_video(input_video_path, output_video_path, crop_position)

