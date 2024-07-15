from moviepy.editor import VideoFileClip

def crop_video_width(video_path, output_path, start_width):
    # Load the video clip
    clip = VideoFileClip(video_path)
    
    # Crop the video from start_width to the end of its width
    cropped_clip = clip.crop(x1=start_width)
    
    # Write the output video file
    cropped_clip.write_videofile(output_path)

# Example usage
video_path = 'Teaser_vid.mp4'
output_path = 'Teaser_vid_1.mp4'
start_width = 2  # Example: crop from 2 pixels (or units) onwards from the left side

crop_video_width(video_path, output_path, start_width)
