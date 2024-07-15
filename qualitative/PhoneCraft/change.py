from moviepy.editor import VideoFileClip

def convert_mov_to_mp4(mov_file_path, mp4_file_path):
    # Load the MOV video
    clip = VideoFileClip(mov_file_path)
    
    # Write the video in MP4 format
    clip.write_videofile(mp4_file_path)

# Example usage
mov_file_path = 'ExtremeBlur.mov'
mp4_file_path = 'ExtremeBlur.mp4'

convert_mov_to_mp4(mov_file_path, mp4_file_path)

