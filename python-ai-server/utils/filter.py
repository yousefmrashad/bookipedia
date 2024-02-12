import cv2 as cv
import numpy as np

'''
Function to map values in range [in_min, in_max] to the range [out_min, out_max]
'''
def map_values(img, in_min, in_max, out_min, out_max):
    return (img - in_min) * ((out_max - out_min) / (in_max - in_min)) + out_min

def filter_image(img: np.ndarray | None, kSize=55, whitePoint=120, blackPoint=70):
    
    # Applying high pass filter
    print("Applying high pass filter")
    
    if kSize % 2 == 0:
        kSize += 1
        
    kernel = np.ones((kSize, kSize), np.float32) / (kSize * kSize)
    filtered = cv.filter2D(img, -1, kernel)
    
    filtered = img.astype('float32') - filtered.astype('float32')
    filtered = filtered + 127 * np.ones(img.shape, np.uint8)
    filtered = filtered.astype('uint8')
    
    print("Selecting white point...")
    _, img = cv.threshold(filtered, whitePoint, 255, cv.THRESH_TRUNC)
    
    img = map_values(img.astype('int32'), 0, whitePoint, 0, 255).astype('uint8')
    
    print("Adjusting black point for final output...")
    img = map_values(img.astype('int32'), blackPoint, 255, 0, 255)
    
    _, img = cv.threshold(img, 0, 255, cv.THRESH_TOZERO)
    img = img.astype('uint8')

    lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    (l, a, b) = cv.split(lab)
    img = cv.add(cv.subtract(l, b), cv.subtract(l, a))
    sharpened = cv.filter2D(img, -1, np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]))
    
    print("\nDone.")
    
    return sharpened