import cv2
import pytesseract

def preprocess_image(image_path):
    # Read the image using OpenCV
    image = cv2.imread(image_path)

    # Adjust image contrast using histogram equalization
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab_planes = list(lab_planes)  # Convert tuple to list
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to create a binary image
    _, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return threshold

def extract_text(image_path):
    # Preprocess the image
    processed_image = preprocess_image(image_path)

    # Save the processed image
    cv2.imwrite('processed_image.jpg', processed_image)

    # Perform OCR using Tesseract
    extracted_text = pytesseract.image_to_string(processed_image)

    return extracted_text

# Provide the path to the image you want to process
image_path = 'test.png'

# Extract text from the image
extracted_text = extract_text(image_path)

# Display the extracted text
print("Extracted Text:")
print(extracted_text)
