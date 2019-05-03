
# coding: utf-8

# # HW3: Automatic Panoramic Image Stitching

# In[16]:


import numpy as np
import cv2
import matplotlib.pyplot as plt


# In[17]:

BFMATCHER_DISTANCE = 0.65

RANSAC_TRAILS = 4000
RANSAC_SAMPLE_NUMBER = 10
RANSAC_INLINER_DISTANCE = 4
PLT_IMAGE_SIZE = 1000


# In[18]:


def brute_force_matcher(featureA, featureB, ratio_threshold):
    """
        ref: chapter 7, P103
        SIFT descriptor 1 -> [a1,a2,....a128]
        SIFT descriptor 2 -> [b1,b2,....b128]
        (DMatch) -> Euclidean distance = sqrt[(a1-b1)^2 + (a2-b2)^2 +...+(a128-b128)^2]
    """
    def _compute_dist(a, b):
        '''
            ref: Chapter 7, P81
            Two methods to calculate ratio distance:
            1. using for loop to subtracting each other and square it.
            2. using matrix dot operation to do it.
            
            (feature is a 1*128 array)
        '''
        diff = featureA[a] - featureB[b]
        return np.dot(diff, np.transpose(diff))
    bestpair = []
    for a in range(featureA.shape[0]):
        min_dist, min2nd_dist = 10000000, 10000000
        minindxA, minindxB = 0, 0
        for b in range(featureB.shape[0]):      
            # the ratio distance of two feature
            distance = _compute_dist(a, b)      
            #record min and second smallest dist and match index of min dist
            if distance <= min_dist :
                min_dist, min2nd_dist = distance, min_dist
                minindxA, minindxB = a, b
            elif distance < min2nd_dist:
                min2nd_dist = distance
        ratio_distance = min_dist / min2nd_dist;
        if ratio_distance < ratio_threshold :
            bestpair.append((minindxA, minindxB, min_dist))
    return bestpair

def sort_matched_points(matches, slice_matched_points=None):
    '''
        sort two points array by ratio distance
    '''
    dtype = [('indexA', int), ('indexB', int), ('Dist', float)]
    matched_point_order = np.array(matches, dtype=dtype) 
    matched_point_order = np.sort(matched_point_order, order='Dist')
    if slice_matched_points != None:
        matched_point_order = matched_point_order[:slice_matched_points]
    return matched_point_order

def show_matched_image(imageA, imageB, KeypointsA, KeypointsB, matched_points, draw_line=True, circle_size=5):
    '''
        showing matched points with line
    '''
    
    # prepare an image to combine two image
    (hA, wA) = imageA.shape[:2]
    (hB, wB) = imageB.shape[:2]
    vis = np.zeros((max(hA, hB), wA + wB, 3), dtype='uint8')
    vis[0:hA, 0:wA] = imageA
    vis[0:hB, wA:] = imageB

    for (indexA, indexB, dist) in matched_points:
        
        # random a line color
        color = np.random.randint(0, high=255, size=(3,))
        color = tuple([int(x) for x in color])

        # draw line by cv2.line
        ptA = (int(KeypointsA[indexA].pt[0]), int(KeypointsA[indexA].pt[1]))
        ptB = (int(KeypointsB[indexB].pt[0] + wA), int(KeypointsB[indexB].pt[1]))
        cv2.circle(vis, ptA, circle_size, color, -1)
        cv2.circle(vis, ptB, circle_size, color, -1)
        if (draw_line):
            cv2.line(vis, ptA, ptB, color, 2)
        
    return vis


# ## Compute Homography By RANSAC

# In[19]:


def get_matched_points(matched_point_order, KeypointsA, KeypointsB):
    '''
        organize two points by the order of these matched points
    '''
    pointA = np.zeros((matched_point_order.shape[0],2))
    pointB = np.zeros((matched_point_order.shape[0],2))
    i=0
    for (indexA, indexB, dist) in matched_point_order:
        pointA[i] = (int(KeypointsA[indexA].pt[0]), int(KeypointsA[indexA].pt[1]))
        pointB[i] = (int(KeypointsB[indexB].pt[0]), int(KeypointsB[indexB].pt[1]))
        i += 1
    return pointA, pointB

def get_normalization_matrix(objp, imgp):
    '''
        get the normalization matrix
    '''
    def _normalization(fpt):
        x_mean, y_mean = np.mean(fpt, axis=0)
        var_x, var_y = np.var(fpt, axis=0)
        s_x , s_y = np.sqrt(2/var_x), np.sqrt(2/var_y)
        n = np.array([[s_x, 0, -s_x*x_mean], 
                      [0, s_y, -s_y*y_mean], 
                      [0, 0, 1]])
        return n.astype(np.float64)
    
    # collect all points into a array
    object_normalization_matrix = _normalization(objp)
    image_normalization_matrix = _normalization(imgp)
    
    return object_normalization_matrix, image_normalization_matrix

def normalize_points(objps, imgps, object_normalization_matrix, image_normalization_matrix):
    num_of_point = len(objps)
    normalized_objpoint = np.zeros((num_of_point,2), dtype=np.float64)
    normalized_impoint = np.zeros((num_of_point,2), dtype=np.float64)
 
    # the z value is depth, let the z value equal to one.
    homograpy_object_points = np.array([ [[each[0]], [each[1]], [1.0]] for each in objps])
    homograpy_image_points = np.array([ [[each[0]], [each[1]], [1.0]] for each in imgps])
    
    # all points are homogeneous
    for i in range(num_of_point):
        n_o = np.matmul(object_normalization_matrix, homograpy_object_points[i])
        homograpy_object_points[i] = n_o/n_o[-1]
        n_u = np.matmul(image_normalization_matrix, homograpy_image_points[i])
        homograpy_image_points[i] = n_u/n_u[-1]  
    
    normalized_objpoint = homograpy_object_points.reshape(homograpy_object_points.shape[0], homograpy_object_points.shape[1])
    normalized_impoint = homograpy_image_points.reshape(homograpy_image_points.shape[0], homograpy_image_points.shape[1])
    normalized_objpoint = normalized_objpoint[:,:-1]        
    normalized_impoint = normalized_impoint[:,:-1]
    
    return normalized_objpoint, normalized_impoint

def get_camera_matrix_for_each_point(objp, imgp):
    '''
        (P.72) pi = MPi, M = 
    '''
    num_of_point = len(objp)
    M = np.zeros((2*num_of_point, 9), dtype=np.float64)
    
    for i in range(num_of_point):
        X = objp[i] # get the object point (X, Y, 1)
        u, v = imgp[i]  # get a image point (u,v)
        M[2*i] = np.array([ -X[0], -X[1], -1, 0, 0, 0, X[0]*u, X[1]*u, u])
        M[2*i + 1] = np.array([ 0, 0, 0, -X[0], -X[1], -1, X[0]*v, X[1]*v, v])
        
    return M

def get_homography_matrix_by_SVD(camera_matrix, object_normalization_matrix, image_normalization_matrix):
    '''
        (P.73)
    '''
    
    U, S, V = np.linalg.svd(camera_matrix)
    
    # set h equal to the smallest sigular value
    h = V[np.argmin(S)]
    
    # v is 9x9 matrix, will to reshape it to 3x3 matrix
    h = h.reshape(3, 3)
    
    # Denormalized homography matrix
    H = np.matmul(np.matmul(np.linalg.inv(image_normalization_matrix), h), object_normalization_matrix)
    H = H[:,:]/H[2, 2]
    
    return H

def get_homography(objpoints, imgpoints):
    
     # get the normalization matrix
    object_normalization_matrix, image_normalization_matrix = get_normalization_matrix(objpoints, imgpoints)
    
    # normalize objpoints and imgpoints
    norm_objpoints, norm_imgpoints  = normalize_points(objpoints, imgpoints, object_normalization_matrix, image_normalization_matrix)
    
    # using pi = MPi to get the camera matrix
    m = get_camera_matrix_for_each_point(norm_objpoints, norm_imgpoints)
    
    # using SVD to get Homography matrix M
    H = get_homography_matrix_by_SVD(m, object_normalization_matrix, image_normalization_matrix)
    
    return H


# In[20]:


def sample_match_points(matched_point_order, pointA, pointB, num):
    '''
        random to sample points
    '''
    sample_point_index = np.random.randint(int(matched_point_order.shape[0]*0.5), size=num)
    sample_pointsA = np.zeros((num,2))
    sample_pointsB = np.zeros((num,2))
    for i in range(num):
        index = sample_point_index[i]
        sample_pointsA[i] = pointA[index]
        sample_pointsB[i] = pointB[index]
    return sample_pointsA, sample_pointsB

def get_liner_number(PointA, PointB, H, threshold):
    '''
        Compute Different between pointA and pointA' and count Inlier number
    '''
    
    inliner_number = 0
    for i in range(PointB.shape[0]):
        
        ww = 1./(H[2,0]*PointA[i,0] + H[2,1]*PointA[i,1] + 1.)
        dx = (H[0,0]*PointA[i,0] + H[0,1]*PointA[i,1] + H[0,2])*ww - PointB[i,0]
        dy = (H[1,0]*PointA[i,0] + H[1,1]*PointA[i,1] + H[1,2])*ww - PointB[i,1]
        diff = (float)(dx*dx + dy*dy)
        
        if diff <= threshold*threshold:
            inliner_number = inliner_number + 1

    return inliner_number

def RANSAC(matched_point_order, pointA, pointB, inliner_threshold):
    '''
        find H with max inliercount
        for i = 0 ~ 1000 
            Sample 4 pair 
            B' = H*pointA
            if || B - B'|| < threshold
                inliercount++
    '''
    max_inliner_number = 0
    best_homograhpy_matrix = np.zeros((3,3))
    for i in range(RANSAC_TRAILS):
        
        # randomly sample point
        sample_pointA, sample_pointB = sample_match_points(matched_point_order, pointA, pointB, RANSAC_SAMPLE_NUMBER)
        
        # using sample point to get homography (HW1)
        sampled_homography_matrix = get_homography(sample_pointA, sample_pointB)
        
        # calculate the number of inliner
        inliner_number = get_liner_number(pointA, pointB, sampled_homography_matrix, inliner_threshold)
        
        # get the best homography matrix with the smallest number of outliers
        if inliner_number >= max_inliner_number:
            max_inliner_number = inliner_number
            best_homograhpy_matrix = sampled_homography_matrix
            
    print('max inlie number:', max_inliner_number)
    return best_homograhpy_matrix


#  ## Warping & Stitching

# In[21]:


def warp_image(imageA, imageB, H):
    
    # prepare the warp image
    dsize = (imageA.shape[0], imageA.shape[1] + imageB.shape[1])
    resultImg = np.zeros((dsize[0], dsize[1],3),  dtype=np.uint8)
    
    # read the image B to the warp image
    xh = np.linalg.inv(H)
    for y in range(dsize[0]):
        for x in range(dsize[1]):
            
            invB_p = np.dot(xh, np.array([x,y,1]))
            invB_p = invB_p/invB_p[-1]
            invB_p = np.rint(invB_p)#convert float to int
            
            # detecte the image edge
            if (invB_p[1] < imageB.shape[0]) and (invB_p[0] < imageB.shape[1]) and (invB_p[0] >= 0) and (invB_p[1] >= 0): 
                resultImg[y,x] = imageB[int(invB_p[1]),int(invB_p[0])]
                
    return resultImg

def linerblending(imageA, warpimg, blendrange):
    
    # prepare the image
    blendingimg = np.zeros(warpimg.shape,  dtype=np.uint8)
    blendingimg = warpimg[:]
    
    # search for the leftmost pixel and the rightmost pixel
    mean = imageA.shape[0]//2
    Wmin = 0
    Wmax = imageA.shape[1]
    for x in range(imageA.shape[1]):
        # the pixel is black, it is the edge of the warp image
        if np.array_equal(warpimg[mean,x], np.array([0,0,0])) == False:
            if Wmin==0 : Wmin = x
            Wmax = x

    # blending
    center = (Wmin + Wmax)//2
    minblend = center - blendrange
    maxblend = center + blendrange 
    for y in range(imageA.shape[0]):
        for x in range(imageA.shape[1]):
            if np.array_equal(warpimg[y,x],np.array([0,0,0])) or x <= minblend:
                blendingimg[y,x] = imageA[y,x]
            elif x <= maxblend:
                p = (x - minblend) / (blendrange*2)
                blendingimg[y,x] = warpimg[y,x]*p + imageA[y,x]*(1-p)
    return blendingimg


# In[22]:


def switch(imgA,imgB):
    #SIFT
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    (KeypointsA, descriptorA) = sift.detectAndCompute(imgA, None)
    (KeypointsB, descriptorB) = sift.detectAndCompute(imgB, None)

    #Find the matching point
    matches = brute_force_matcher(descriptorA, descriptorB, BFMATCHER_DISTANCE)

    #Sort the matching points by Dist
    matched_point_order = sort_matched_points(matches)

    #Show resultimage with line between matching points
    matched_feature_image = show_matched_image(imgA, imgB, KeypointsA, KeypointsB, matched_point_order)
    plt.figure(figsize=(PLT_IMAGE_SIZE, PLT_IMAGE_SIZE))
    plt.subplot(411),plt.imshow(matched_feature_image),plt.axis('off')
    
    #Convert Match pair into point set pointA and PointB
    pointsA, pointsB = get_matched_points(matched_point_order, KeypointsA, KeypointsB)
    
    #Get BestHomographyMatrix by RANSAC
    H = RANSAC(matched_point_order, pointsB, pointsA, RANSAC_INLINER_DISTANCE)
    print(H)
    
    #Show Warp image
    resultImg = warp_image(imgA, imgB, H)
    plt.subplot(412),plt.imshow(resultImg/255),plt.axis('off')
    
    #Show Result image by liner blending
    blendingImg = linerblending(imgA, resultImg, 10)
    plt.subplot(413),plt.imshow(resultImg/255),plt.axis('off')
    
    #Show Result image
    (hA, wA) = imgA.shape[:2]
    resultImg[0:hA, 0:wA] = imgA
    plt.subplot(414),plt.imshow(resultImg/255),plt.axis('off')
    plt.show()
    return blendingImg