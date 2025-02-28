import cv2
import numpy as np

def perform_image_analysis(image_path):
    # Load the image
    image = cv2.imread(image_path)
    
    # Check if the image was successfully loaded
    if image is None:
        raise FileNotFoundError(f"Image at path {image_path} could not be loaded.")
    
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Otsu's thresholding to segment the lesion
    _, lesion_mask = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Find contours in the lesion mask
    contours, _ = cv2.findContours(lesion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Assume the largest contour is the lesion
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Calculate the area of the largest contour
    area = cv2.contourArea(largest_contour)
    
    # Translate pixel area to real world area
    real_world_area = translate_to_real_world_area(area)
    
    # Draw the contour on the original image
    cv2.drawContours(image, [largest_contour], -1, (0, 255, 0), 2)
    
    # Display the final image with the contour
    cv2.imshow('Processed Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return real_world_area

def translate_to_real_world_area(area):
    # Translate pixel area to real world area
    # Assuming a conversion factor (e.g., 1 pixel = 0.1 mm^2)
    conversion_factor = 0.1
    return area * conversion_factor

# Example usage
image_path = "/Users/yabserabekele/Downloads/Lesion_images/testimage.jpeg"
try:
    lesion_area = perform_image_analysis(image_path)
    print("Area of lesion:", lesion_area)
except (FileNotFoundError, ValueError) as e:
    print(e)
