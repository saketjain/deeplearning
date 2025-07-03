import cv2
import numpy as np
import imutils

def align_images(image, template,   max_features=500, keep_percent=0.2, debug=False):
    
    # Convert to Gray scale
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    
    # use orb to detect key points and local binary invariants
    orb = cv2.ORB_create()
    (kpsA, descsA) = orb.detectAndCompute(image_gray, None)
    (kpsB, descsB) = orb.detectAndCompute(template_gray, None)
    
    # match features
    method = cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING
    matcher = cv2.DescriptorMatcher_create(method)
    matches = matcher.match(descsA, descsB)
    
    #sort matches on distance
    matches = sorted(matches, key=lambda x: x.distance)
    
    # keep top matches
    keep = int(len(matches) * keep_percent)
    matches = matches[:keep]
    
    if debug:
        matchedVis = cv2.drawMatches(image, kpsA, template, kpsB, matches, None)
        cv2.imshow("Matched Key Points", matchedVis)
        cv2.waitKey(0)
        
    # allocate memory for the keypoints (x, y)-coordinates from the
	# top matches -- we'll use these coordinates to compute our
	# homography matrix
    ptsA = np.zeros((len(matches), 2), dtype="float")
    ptsB = np.zeros((len(matches), 2), dtype="float")
	
    # loop over the top matches
    for (i, m) in enumerate(matches):
		
        # indicate that the two keypoints in the respective images
		# map to each other
        ptsA[i] = kpsA[m.queryIdx].pt
        ptsB[i] = kpsB[m.trainIdx].pt


	# compute the homography matrix between the two sets of matched
	# points
    (H, mask) = cv2.findHomography(ptsA, ptsB, method=cv2.RANSAC)
	
    # use the homography matrix to align the images
    (h, w) = template.shape[:2]
    aligned = cv2.warpPerspective(image, H, (w, h))
	
    # return the aligned image
    return aligned
    
    
    
