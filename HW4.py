
# coding: utf-8

# # HW4: Structure-from-motion
# 
# Ref:
# 
# - https://blog.csdn.net/haizimin/article/details/49836077
# - https://github.com/jesolem/PCV/blob/master/pcv_book/sfm.py
# - multiple view geometry in computer vision
# http://cvrs.whu.edu.cn/downloads/ebooks/Multiple%20View%20Geometry%20in%20Computer%20Vision%20(Second%20Edition).pdf

# In[168]:


import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random

import hw3
np.set_printoptions(suppress=True)


# ## Input Images

# In[169]:


DEBUG_IMAGE_INDEX = 1
RANSAC_INLINER_THRESHOLD = 0.000003
RANSAC_SAMPLE_NUMBER = 2000

if(DEBUG_IMAGE_INDEX==1):
    image1 = cv2.imread('./data/Mesona1.JPG')
    image2 = cv2.imread('./data/Mesona2.JPG')
elif(DEBUG_IMAGE_INDEX == 2):
    image1 = cv2.imread('./data/Statue1.bmp')
    image2 = cv2.imread('./data/Statue2.bmp')
image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)


# ## Intrinsic Matrix

# In[170]:


intrinsic_matrix1 = np.zeros((3,3))
intrinsic_matrix1 = np.zeros((3,3))
rotation_matrix1, rotation_matrix2 = None, None
transform_matrix1, transform_matrix2 = None, None

if(DEBUG_IMAGE_INDEX == 1):
    intrinsic_matrix1 = np.array([[1.4219, 0.005, 0.5092],
                                  [0, 1.4219, 0.3802],
                                  [0, 0, 0.0010]])
    intrinsic_matrix1 = intrinsic_matrix1 / intrinsic_matrix1[2,2]
    intrinsic_matrix2 = intrinsic_matrix1
else:
    intrinsic_matrix1 = np.array([[5426.566895, 0.678017, 330.096680],
                 [0.000000, 5423.133301, 648.950012],
                 [0.000000, 0.000000, 1.000000]])
    rotation_matrix1 = np.array([[0.140626, 0.989027, -0.045273],
              [0.475766, -0.107607, -0.872965],
              [-0.868258, 0.101223, -0.485678]])
    transform_matrix1 = np.array([67.479439, -6.020049, 40.224911])

    intrinsic_matrix2 = np.array([[5426.566895, 0.678017, 387.430023],
                  [0.000000, 5423.133301, 620.616699],
                  [0.000000, 0.000000, 1.000000]])
    rotation_matrix2 = np.array([[0.336455, 0.940689, -0.043627],
              [0.446741, -0.200225, -0.871970],
              [-0.828988, 0.273889, -0.487611]])
    transform_matrix2 = np.array([62.882744, -21.081516, 40.544052])
    
print("Intrinsic Matrix 1\n", intrinsic_matrix1)
print("Intrinsic Matrix 2\n", intrinsic_matrix2)

if(rotation_matrix1 is not None):
    print('rotation_matrix1\n', rotation_matrix1)
    print('rotation_matrix2\n', rotation_matrix2)
    print('transform_matrix1\n', transform_matrix1)
    print('transform_matrix1\n', transform_matrix2)


# ## Feature Points

# In[171]:


def get_feature_points(img1, img2):
    sift = cv2.xfeatures2d.SIFT_create()

    (kp1, des1) = sift.detectAndCompute(image1, None)
    (kp2, des2) = sift.detectAndCompute(image2, None)

    BF_MACTHER_DISTANCE = 0.65
    matches = hw3.brute_force_matcher(des1, des2, BF_MACTHER_DISTANCE)
    matched_pt_order = hw3.sort_matched_points(matches)
    imgpts1, imgpts2 = hw3.get_matched_points(matched_pt_order, kp1, kp2)
    
    # FLANN parameters
#     FLANN_INDEX_KDTREE = 0
#     index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
#     search_params = dict(checks=50)
    
#     flann = cv2.FlannBasedMatcher(index_params,search_params)
#     matches = flann.knnMatch(des1,des2,k=2)
    
#     good = []
#     pts1 = np.zeros((len(matches), 2), dtype=np.float64)
#     pts2 = np.zeros((len(matches), 2), dtype=np.float64)
    
#     # ratio test as per Lowe's paper
#     for i,(m,n) in enumerate(matches):
#         if m.distance < 0.8*n.distance:
#             good.append(m)
#             pts1[i] = (kp1[m.queryIdx].pt)
#             pts2[i] = (kp2[m.trainIdx].pt)
            
#     return pts1,pts2,good
    
    return imgpts1, imgpts2

imgpts1, imgpts2 = get_feature_points(image1, image2)
matched_feature_image = hw3.show_matched_image(image1, image2, imgpts1, imgpts2, draw_line=False, circle_size=10)
plt.imshow(matched_feature_image), plt.axis('off'), plt.show()


# ## Fundamental and Essential Matrix

# In[172]:


def get_normalization_matrix(img_shape):
    '''
        get the normalization matrix
    '''
    T = np.array([[2/img_shape[1], 0, -1],
                   [0, 2/img_shape[0], -1],
                   [0, 0, 1]])
    return T

def normalization(imgpts1, imgpts2, img1, img2):
    '''
        ref: lecture P.54
    '''
    # t1: image1 normalization matrix, t2: image2 normalization matrix
    t1 = get_normalization_matrix(img1.shape)
    t2 = get_normalization_matrix(img2.shape)
    
    # to homography coordinate
    homopts1 = np.array([ [each[0], each[1], 1.0] for each in imgpts1])
    homopts2 = np.array([ [each[0], each[1], 1.0] for each in imgpts2])
    
    num_of_point = len(imgpts1)
    for i in range(num_of_point): 
        
        # the Homogeneous coefficient should be one
        p2h = np.matmul(t1, homopts1[i])
        homopts1[i] = p2h/p2h[-1]
        
        p2h1 = np.matmul(t2, homopts2[i])
        homopts2[i] = p2h1/p2h1[-1]
    
    normalpts1 = np.delete(homopts1, -1, axis=1)
    normalpts2 = np.delete(homopts2, -1, axis=1)
    
    return normalpts1, normalpts2, t1, t2

def sample_points(pointA, pointB, sample_number):
    sample_point_index = random.sample(range(pointA.shape[0]), sample_number)
    sample_pointsA = np.zeros((sample_number,2))
    sample_pointsB = np.zeros((sample_number,2))
    for i in range(sample_number):
        index = sample_point_index[i]
        sample_pointsA[i] = pointA[index]
        sample_pointsB[i] = pointB[index]
    return sample_pointsA, sample_pointsB

def denormalize_fundamental_mat(normalmat1, normalmat2, normalize_fundamental):
    '''
        ref: Multiple View Geometry in Computer Vision - Algorithm 11.1
    '''
    F = np.matmul(np.matmul(normalmat2.T, normalize_fundamental), normalmat1)
    F = F/F[-1,-1]
    return F

def get_fundamental(normalpts1, normalpts2):
    '''
        ref: Multiple View Geometry in Computer Vision - Chapter 11.1, lecture P.50
    '''
    
    A = np.zeros((len(normalpts1), 9), dtype=np.float64)
    for i in range(len(normalpts1)):
        x1, y1 = normalpts1[i][0], normalpts1[i][1]
        x2, y2 = normalpts2[i][0], normalpts2[i][1]
        A[i] = np.array([x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1])
    
    # resolve det(f) = 0
    u, s, v = np.linalg.svd(A)
    F = v[-1]
    F = F.reshape(3, 3)
    u, s, v = np.linalg.svd(F)
    s = np.array([[s[0], 0 ,0],
                 [0, s[1], 0],
                 [0 , 0, 0]])
    F = np.matmul(np.matmul(u, s), v)

    return F

def get_geometric_error(testpts1, testpts2, fundamentalmat, inliner_threshold):
    '''
        ref: Multiple View Geometry 11.4.3
    '''
    error = 0
    inliner_number = 0
    inlinerpt_indexs = np.zeros((len(testpts1),1), dtype=np.int)
    
    
    # transform test points to homography coordinate
    x1 = np.array([ [each[0], each[1], 1.0] for each in testpts1]).T
    x2 = np.array([ [each[0], each[1], 1.0] for each in testpts2]).T
    
    
    fx = np.dot(fundamentalmat.T, x1)
    ftx = np.dot(fundamentalmat, x2)
    
    d = np.power(fx[0], 2) + np.power(fx[1], 2) + np.power(ftx[0], 2) + np.power(ftx[1], 2)
    m = np.diag(np.dot(np.dot(x2.T, fundamentalmat), x1))
    m = np.power(m, 2)
    sampson = m/d
    
    for i in range(sampson.shape[0]):
        if(sampson[i] <= inliner_threshold):
            error += sampson[i]
            inlinerpt_indexs[inliner_number] = i
            inliner_number += 1
            
    return error, inliner_number, inlinerpt_indexs[:inliner_number, :]

def get_inliner_points(x1, x2, inliner_indexs):
    inlinerpts1 = np.zeros((len(inliner_indexs), 2))
    inlinerpts2 = np.zeros((len(inliner_indexs), 2))
    
    print(inliner_indexs.shape)
    print(inliner_indexs[0])
    
    for i in range(inliner_indexs.shape[0]):
        index = inliner_indexs[i]
        inlinerpts1[i] = x1[index]
        inlinerpts2[i] = x2[index]
    
    return inlinerpts1, inlinerpts2

def get_essential_mat(K1, K2, F):
    '''
        ref: Multiple View Geometry 9.12
    '''
    if(K1[-1,-1] != 1):
        K1 = K1 / K1[-1,-1]
    if(K2[-1,-1] != 1):
        K2 = K2 / K2[-1,-1]
        
    E = np.matmul( K2.T , np.matmul(F,K1))
    
    return E

def find_fundamental_by_RANSAC(imgpts1, imgpts2, img1, img2, inliner_threshold, ransac_iteration = 2000):
    '''
        ref: Multiple View Geometry 11.6
    '''
    
    best_fundamental = np.zeros((3,3))
    best_inlinernum = -1
    
    print("Key Point Number\n", len(imgpts1))
    ransac_sample_number = 8
    
    # normalization the key points
    normalpts1, normalpts2, nomalmat1, normalmat2 = normalization(imgpts1, imgpts2, img1, img2)
    best_error = 0
    best_inlinerpt_index = []
    
    for i in range(ransac_iteration):
        
        sampts1, sampts2 = sample_points(normalpts1, normalpts2, ransac_sample_number)
        unnormalized_f = get_fundamental(sampts1, sampts2)
        error, inlinernum, inlinerpt_indexs = get_geometric_error(normalpts1, normalpts2, unnormalized_f, inliner_threshold)
        
        if(inlinernum > best_inlinernum):
            best_error = error
            best_fundamental = unnormalized_f
            best_inlinernum = inlinernum
            best_inlinerpt_index = inlinerpt_indexs
            
    # homographs coefficient and denormalize the fundamental matrix
    best_fundamental = denormalize_fundamental_mat(nomalmat1, normalmat2, best_fundamental)
    best_essential = get_essential_mat(intrinsic_matrix1, intrinsic_matrix2, best_fundamental)
    best_inlinerpts1, best_inlinerpts2 = get_inliner_points(imgpts1, imgpts2, best_inlinerpt_index)
    
    print("RANSC Error\n", best_error)
    print("Inliner Number\n", best_inlinernum)
    
    return best_fundamental, best_essential, best_inlinerpts1, best_inlinerpts2 


# In[173]:


# find the fundamnetal matrix
fundamentalmat, essentialmat, inlinerpts1, inlinerpts2 = find_fundamental_by_RANSAC(imgpts1, imgpts2, image1, image2, RANSAC_INLINER_THRESHOLD, RANSAC_SAMPLE_NUMBER)

print('keypts1.shape\n', imgpts1.shape)
print("F\n", fundamentalmat)
print("E\n", essentialmat)
print('inlinerpts1.sahpe\n', inlinerpts1.shape)
print('inlinerpts2.sahpe\n', inlinerpts2.shape)

# by opencv
fundamentalmat_opencv, _= cv2.findFundamentalMat(imgpts1, imgpts2, method =cv2.FM_8POINT + cv2.FM_RANSAC)
essentialmat_opencv = get_essential_mat(intrinsic_matrix1, intrinsic_matrix2, fundamentalmat_opencv)

print("F by opencv\n", fundamentalmat_opencv)
print("E by opencv\n", essentialmat_opencv)


# ## Draw Epipolar Lines

# In[174]:


def compute_correspond_epilines(keypts, which_image, fundamental):
    '''
        ref: https://github.com/opencv/opencv/blob/f5801ee7dac4114ac2995a5fd3866ac7775752f7/modules/calib3d/src/fundam.cpp#L836
        l = Fx'
        l' = F^Tx
    '''
    lines = np.zeros((len(keypts), 3))
    
    if (which_image == 2):
        fundamental = np.transpose(fundamental)
    
    for i, p in enumerate(keypts):
        hp = np.array([p[0], p[1], 1])
        l = np.matmul(fundamental, np.transpose(hp))
        
        a, b, c = l[0], l[1], l[2]
        check = a*a + b*b
        if check != 0:
            check = np.sqrt(check)
        else:
            check = 1
        lines[i] = np.array([a/check, b/check ,c/check])
        
    return lines

def draw_epilines(img1, img2, lines, pts1, pts2, colors):
    '''
        ref: https://docs.opencv.org/3.4.4/da/de9/tutorial_py_epipolar_geometry.html
        x0, y0 = (0, -b/c)
        x1, y1 = (w, -(aw+c)/b)
    '''
    imgA = np.copy(img1)
    imgB = np.copy(img2)
    h, w, _ = img1.shape
    
    i=0
    for r,pt1, pt2 in zip(lines, pts1, pts2):
        x0, y0 = (0, int(-r[2]/r[1]))
        x1, y1 = (w, int(-(r[0]*w+r[2])/r[1]))
        imgA = cv2.line(imgA, (x0, y0), (x1, y1), colors[i], 5)
        imgA = cv2.circle(imgA, (int(pt1[0]), int(pt1[1])), 15, (255, 0, 0), -1)
        imgB = cv2.circle(imgB, (int(pt2[0]), int(pt2[1])), 15, (255, 0, 0), -1)
        i += 1
    return imgA, imgB

# prepare line color
colors = np.zeros((len(inlinerpts1), 3))
for i in range(len(inlinerpts1)):
    colors[i] = tuple(np.random.randint(0, 255, 3).tolist())

# show image epilines
lines1 = compute_correspond_epilines(inlinerpts2, 2, fundamentalmat)
img3, _ = draw_epilines(image1, image2, lines1, inlinerpts1, inlinerpts2, colors)
lines2 = compute_correspond_epilines(inlinerpts1, 1, fundamentalmat)
img4, _ = draw_epilines(image2, image1, lines2, inlinerpts1, inlinerpts2, colors)

# by opencv
lines3 = cv2.computeCorrespondEpilines(inlinerpts2, 2, fundamentalmat_opencv)
lines3 = lines3.reshape(-1,3)
img5, _ = draw_epilines(image1, image2, lines3, inlinerpts1, inlinerpts2, colors)
lines4 = cv2.computeCorrespondEpilines(inlinerpts1, 1, fundamentalmat_opencv)
lines4 = lines4.reshape(-1,3)
img6, _ = draw_epilines(image2, image1, lines4, inlinerpts1, inlinerpts2, colors)

plt.figure(figsize=(10, 10))
plt.subplot(321), plt.imshow(image1)
plt.subplot(322), plt.imshow(image2)
plt.subplot(323), plt.imshow(img3)
plt.subplot(324), plt.imshow(img4)
plt.subplot(325),plt.imshow(img5)
plt.subplot(326),plt.imshow(img6)
plt.show()


# ## Linear Triangulation
# project matirx = K[R|t] = 3x4
# 
# camera matrix = intrinsic matrix = k = 3x3
# 
# external rotation matrix = R = 3x3
# 
# translation matrix = t = 1x3

# In[175]:


def combine_external(r, t):
    ex = np.array([[r[0,0], r[0,1], r[0,2], t[0]],
                   [r[1,0], r[1,1], r[1,2], t[1]],
                   [r[2,0], r[2,1], r[2,2], t[2]],])
    return ex

def get_external(e, index=None):
    '''
        get external parameter
        ref: homework lecture page 3
    '''
    w = np.array([[0.0, -1.0, 0.0],[1.0, 0.0, 0.0],[0.0, 0.0, 1.0]])
    z = np.array([[0.0, 1.0, 0.0],[-1.0, 0.0, 0.0],[0.0, 0.0, 0.0]])
    
    U, S, V = np.linalg.svd(e)
    m = (S[0] + S[1])/2
    S[0] = m
    S[1] = m
    S[2] = 0

    T1 = U[:,2]
    T2 = -U[:,2]
    R1 = np.matmul(np.matmul(U, w.T), V)
    R2 = np.matmul(np.matmul(U, w), V)

    if(np.linalg.det(R1)<0):
        R1 = -1*R1
    if(np.linalg.det(R2)<0):
        R2 = -1*R2
    
    if(index is None):
        return T1, T2, R1, R2
    elif(index == 0):
        return T1, R1
    elif(index == 1):
        return T1, R2
    elif(index == 2):
        return T2, R1
    elif(index == 3):
        return T2, R2

def linear_LS_Triangulation(x1, p1, x2, p2):
    '''
        ref: 1995 Triangulation ch5.1 Linear Triangulation
        ref: Multiple View Geometry 12.2
        https://perception.inrialpes.fr/Publications/1997/HS97/HartleySturm-cviu97.pdf
    '''
    
    A = np.array([x1[0]*p1[2,:] - p1[0,:], 
                  x1[1]*p1[2,:] - p1[1,:],
                  x2[0]*p2[2,:] - p2[0,:],
                  x2[1]*p2[2,:] - p2[1,:]])
    
    U, S, V = np.linalg.svd(A)
    X = V[-1]/V[-1,3]
    return X

def in_front_of_camera(R, T, pts,is_opencv=False):
    num_of_points = 0
    camera_center = -np.dot(np.transpose(R),T)
    vide_direciton = np.transpose(R)[2,:]
    for i in range(pts.shape[0]):
        if is_opencv:
            hp = pts[:,i]
            hp = hp/hp[-1]
            X_location = hp[:3] - camera_center
        else: 
            X_location = pts[i] - camera_center
        if np.dot(X_location,vide_direciton) > 0 :
            num_of_points = num_of_points + 1
    return num_of_points

def get_triangulatepts(r1, t1, r2, t2, x1, x2, p1, p2):
    
    front_count = 0
    project_points = np.zeros((x1.shape[0], 3))
    
    # linear triangulate
    for i in range(x1.shape[0]):
        x = linear_LS_Triangulation(x1[i], p1, x2[i], p2)
        project_points[i] = x[0:3]
    
    front_count = in_front_of_camera(r1, t1, project_points)
    front_count += in_front_of_camera(r2, t2, project_points)
    
    return project_points, front_count

def show_cloud_points(plt_title, pts, cv_pts=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    fig.suptitle(plt_title, fontsize=16)
    ax.set_xlabel('x axis')
    ax.set_ylabel('y axis')
    ax.set_zlabel('z axis')
    for i, p in enumerate(pts):
        xs = p[0]
        ys = p[1]
        zs = p[2]
        ax.scatter(xs, ys, zs, color='#054E9F', s=3)
        
    if(cv_pts is not None):
        for i in range(cv_pts.shape[1]):
            hp = cv_pts[:,i] 
            x = hp[0]
            y = hp[1]
            z = hp[2]
            ax.scatter(x, y, z, color='#000000', s=3)
    plt.show()

def triangulate_points(e, x1, x2, k1, k2, r1=None, r2=None, c1=None, c2=None):
    '''
        ref: Multiple View Geometry 9.13
    '''
    # check the intrinc matrix Homogeneous coefficient
    
    if(k1[-1,-1] != 1.):
        k1 = k1/k1[-1,-1]
    if(k2[-1,-1] != 1.):
        k2 = k2/k2[-1,-1]
    
    p1 = None
    p2 = None
    triangulation_points = None
    
    if(r1 is not None):
        # ref: http://ksimek.github.io/2012/08/22/extrinsic/?fbclid=IwAR27rAaOz1pu5eFdSkabeU9Mu1MGnkNNipiYUIrxMqpd5AXNpVpRpi79ZKI
        T1 = -np.matmul(r1, c1)
        T2 = -np.matmul(r2, c2)
        p1 = np.matmul(k1, combine_external(r1, T1))
        p2 = np.matmul(k2, combine_external(r2, T2))
        
        triangulation_points, front_count = get_triangulatepts(r1, T1, r2, T2, x1, x2, p1, p2)
        
    else:
        # camera 1 project matrix
        R1 = np.array([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1]])
        T1= np.array([0 , 0, 0])
        p1 = np.matmul(k1, combine_external(R1, T1))

        max_front_count = -1
        correct_p2_index = -1
        triangulation_points = np.zeros((x1.shape[0], 3))

        for p2_index in range(4):
            
            # Thre have four camera direction (ref: Multiple View Geometry 9.14)
            T2, R2 = get_external(e, p2_index)
            the_p2 = np.matmul(k2, combine_external(R2, T2))
            project_points, front_count = get_triangulatepts(R1, T1, R2, T2, x1, x2, p1, the_p2) 
            
            show_cloud_points('P2-'+str(p2_index), project_points)
            
            # check the front points
            if(front_count > max_front_count):
                max_front_count = front_count
                triangulation_points = np.copy(project_points)
                p2 = the_p2

    return triangulation_points, p1, p2

def get_cv_projectpts(pts):
    projectpts = np.transpose(pts)
    return projectpts

# our triangulate method
project_pts, p1, p2 = triangulate_points(essentialmat,
                                         inlinerpts1, inlinerpts2,
                                         intrinsic_matrix1, intrinsic_matrix2,
                                         rotation_matrix1, rotation_matrix2,
                                         transform_matrix1, transform_matrix2)

# opencv triagulate method, need transform the format of inliner points
# cv_inlinerpts1 = get_cv_projectpts(inlinerpts1)
# cv_inlinerpts2 = get_cv_projectpts(inlinerpts2)
# cloudpts1_cv = cv2.triangulatePoints(p1, p2, cv_inlinerpts1, cv_inlinerpts2)

#show_cloud_points(project_pts, cloudpts1_cv)
show_cloud_points('3D reconstructed', project_pts)


# In[176]:


import answer

a_x1, a_x2, _ = answer.sift_detector(image1, image2)

a_x1 = np.float64(a_x1)
a_x2 = np.float64(a_x2)

# normalization the key points
normalpts1, normalpts2, nomalmat1, normalmat2 = answer.get_normalize(a_x1, a_x2, image1.shape, image2.shape)
answer_f, mask = answer.get_fundamental(normalpts1, normalpts2, nomalmat1, normalmat2)
print('anser f\n', answer_f)

anser_e = get_essential_mat(intrinsic_matrix1, intrinsic_matrix2, answer_f)

a_inlinerpts1 = a_x1[mask.ravel()==1]
a_inlinerpts2 = a_x2[mask.ravel()==1]
X = answer.cal_P(anser_e, a_inlinerpts1, a_inlinerpts2, intrinsic_matrix1, intrinsic_matrix2)

show_cloud_points('answer', X)


# ## Save Data For MatLab

# In[177]:


file_name = ''

if(DEBUG_IMAGE_INDEX == 1):
    file_name = 'Mesona1'
elif (DEBUG_IMAGE_INDEX == 2):
    file_name = 'Statue1'
else:
    file_name = 'our'
    
np.savetxt("./data/two_d_points_"+file_name+".csv", inlinerpts1, delimiter=",")
np.savetxt("./data/three_d_points_"+file_name+".csv", project_pts, delimiter=",")
np.savetxt("./data/camera_matrix_"+file_name+".csv", p2, delimiter=",")

