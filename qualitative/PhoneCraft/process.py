import cv2, os

def resize_image_cv2(input_path, output_path, new_width, new_height):
    # Read the image from the file
    img = cv2.imread(input_path)
    
    # Resize the image
    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    # Save the resized image to the specified path
    cv2.imwrite(output_path, resized_img)

# Example usage
root = "REDS/"
root1 = "REDS_z/"
for each in os.listdir(root):
    if "png" not in each: continue
    
    input_path = root + each
    output_path = root + each
    new_width = 300
    new_height = 300 

    resize_image_cv2(input_path, output_path, new_width, new_height)
