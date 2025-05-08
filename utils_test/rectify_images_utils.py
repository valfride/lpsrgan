import cv2  # OpenCV library for image processing
import numpy as np  # Numerical library for handling arrays and mathematical functions
import json  # Library for handling JSON files
from pathlib import Path  # Path library to handle file paths
from matplotlib import pyplot as plt  # Matplotlib for plotting images

# Define a class for rectifying images
class imageRectifier:    
    # Method to load points (coordinates) from a JSON file associated with the image
    def get_pts(self, file):
        file = Path(file)  # Convert the file path to a Path object
        file = file.with_suffix('.json')  # Ensure the file has a .json suffix
        with open(file, 'r') as j:  # Open the JSON file in read mode
            pts = json.load(j)['shapes'][0]['points']  # Load the points from the first shape in the JSON file
        
        return pts  # Return the points for the image

    # Method to load an image, with optional color conversion from BGR to RGB
    def open_image(self, img, cvt=True):
        img = cv2.imread(img)  # Read the image file
        if cvt is True:  # Convert color if specified
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert image color from BGR to RGB
        return img  # Return the image

    # Method to rectify (transform) the image to a top-down perspective
    def rectify_img(self, img, pts, margin=2):
        # Unpack the four points in a consistent order: top-left, top-right, bottom-right, bottom-left
        (tl, tr, br, bl) = pts
     
        # Calculate the width of the new image based on the maximum width between opposite sides
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))  # Width between bottom-left and bottom-right
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))  # Width between top-left and top-right
        maxWidth = max(int(widthA), int(widthB))  # Use the maximum of these two widths
     
        # Calculate the height of the new image based on the maximum height between opposite sides
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))  # Height between top-right and bottom-right
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))  # Height between top-left and bottom-left
        maxHeight = max(int(heightA), int(heightB))  # Use the maximum of these two heights
    
        # Add a margin to the calculated width and height for extra padding around the rectified image
        maxWidth += margin * 2
        maxHeight += margin * 2
     
        # Define the destination points for the perspective transform, ordered top-left, top-right, bottom-right, bottom-left
        ww = maxWidth - 1 - margin
        hh = maxHeight - 1 - margin
        c1 = [margin, margin]  # Top-left corner with margin
        c2 = [ww, margin]      # Top-right corner with margin
        c3 = [ww, hh]          # Bottom-right corner with margin
        c4 = [margin, hh]      # Bottom-left corner with margin
    
        # Create arrays for the source and destination points for perspective transformation
        dst = np.array([c1, c2, c3, c4], dtype='float32')  # Destination points
        pts = np.array(pts, dtype='float32')  # Convert input points to float32
        # Calculate the perspective transform matrix M
        M = cv2.getPerspectiveTransform(pts, dst)
        # Apply the perspective transformation to warp the image to the new perspective
        warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
     
        return warped  # Return the rectified (warped) image

    # Callable method to handle the entire rectification process for a given image path
    def __call__(self, img_path):
        img = self.open_image(img_path)  # Load the image
        pts = self.get_pts(img_path)     # Get the points for rectification
        
        rect_image = self.rectify_img(img, pts)  # Rectify the image using the points
    
        return rect_image  # Return the rectified image
    
# Main block to run the code if executed as a script
if __name__ == '__main__':
    # Instantiate the imageRectifier class
    rectifier = imageRectifier()
    # Display the rectified image using Matplotlib
    plt.imshow(rectifier("./dataset_intelbras_1920x1080/mercosur/plate-000009/hr-002.png"))
