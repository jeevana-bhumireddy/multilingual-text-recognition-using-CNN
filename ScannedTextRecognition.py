import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

# def preprocessingAnImage(img):

#     # Reading an Image using OpenCV
#     input_img=cv2.imread(img)

#     # Rotating the image
#     rotate_img=cv2.rotate(input_img, cv2.ROTATE_90_CLOCKWISE)

#     # Inverting the image colors using bitwise_not
#     inverted_image=cv2.bitwise_not(rotate_img)

#     # Converting the image to grayscale
#     gray_image=cv2.cvtColor(inverted_image, cv2.COLOR_BGR2GRAY)

#     # Binarizing the image
#     thresh, im_bw=cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

#     # Noise removal using Morphological Transformations
#     kernel = np.ones((1, 1), np.uint8)
#     noise_removal_image=cv2.dilate(im_bw, kernel, iterations=1)
#     erosion_image=cv2.erode(noise_removal_image, kernel, iterations=1)
#     morphological_image=cv2.morphologyEx(erosion_image, cv2.MORPH_CLOSE, kernel)
#     median_blur_image=cv2.medianBlur(morphological_image, 3)

#     # Detecting and removing the borders
#     contours, hierarchy = cv2.findContours(median_blur_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     cntsSorted=sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
#     cnt=cntsSorted[-1]
#     x, y, w, h=cv2.boundingRect(cnt)
#     crop=median_blur_image[y:y+h, x:x+w]


#     # Detecting the text using pytesseract
#     text=pytesseract.image_to_string(crop, lang='eng')
#     print(text)


key=cv2.waitKey(1)
cam=cv2.VideoCapture(1)
while True:
    try:
        check, frame=cam.read()
        cv2.imshow("Capturing", frame)
        key=cv2.waitKey(1)
        if key==ord('s'):
            cv2.imwrite(filename="./Outputs/ScannedImage.jpg", img=frame)
            cam.release()
            cv2.destroyAllWindows()
            # Reading an Image using OpenCV
            input_img=cv2.imread("./Outputs/ScannedImage.jpg")

            # Rotating the image
            rotate_img=cv2.rotate(input_img, cv2.ROTATE_90_CLOCKWISE)

            # Inverting the image colors using bitwise_not
            inverted_image=cv2.bitwise_not(rotate_img)

            # Converting the image to grayscale
            gray_image=cv2.cvtColor(inverted_image, cv2.COLOR_BGR2GRAY)

            # Binarizing the image
            thresh, im_bw=cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

            # Noise removal using Morphological Transformations
            kernel = np.ones((1, 1), np.uint8)
            noise_removal_image=cv2.dilate(im_bw, kernel, iterations=1)
            erosion_image=cv2.erode(noise_removal_image, kernel, iterations=1)
            morphological_image=cv2.morphologyEx(erosion_image, cv2.MORPH_CLOSE, kernel)
            median_blur_image=cv2.medianBlur(morphological_image, 3)

            # Detecting and removing the borders
            contours, hierarchy = cv2.findContours(median_blur_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cntsSorted=sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
            cnt=cntsSorted[-1]
            x, y, w, h=cv2.boundingRect(cnt)
            crop=median_blur_image[y:y+h, x:x+w]

            # Detecting the text using pytesseract
            text=pytesseract.image_to_string(crop, lang='eng')
            print(text)
            break
        elif key==ord('q'):
            print("Turning off camera.")
            cam.release()
            print("Camera off.")
            print("Program ended.")
            cv2.destroyAllWindows()
            break
    
    except(KeyboardInterrupt):
        print("Turning off camera.")
        cam.release()
        print("Camera off.")
        print("Program ended.")
        cv2.destroyAllWindows()
        break