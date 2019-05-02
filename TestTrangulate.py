import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import hw3


# ## 將像素對應到三維空間

# In[35]:


def linear_LS_Triangulation(x1, camera_matrix1, x2, camera_matrix2):
    '''
        ref: 1995 Triangulation, ch5.1 Linear Triangulation
        https://perception.inrialpes.fr/Publications/1997/HS97/HartleySturm-cviu97.pdf
    '''
    A = np.array([[(x1[0]*camera_matrix1[2,0]-camera_matrix1[0,0]), (x1[0]*camera_matrix1[2,1]-camera_matrix1[0,1]), (x1[0]*camera_matrix1[2,2]-camera_matrix1[0,2])],
                  [(x1[1]*camera_matrix1[2,0]-camera_matrix1[1,0]), (x1[1]*camera_matrix1[2,1]-camera_matrix1[1,1]), (x1[1]*camera_matrix1[2,2]-camera_matrix1[1,2])],
                  [(x2[0]*camera_matrix2[2,0]-camera_matrix2[0,0]), (x2[0]*camera_matrix2[2,1]-camera_matrix2[0,1]), (x2[0]*camera_matrix2[2,2]-camera_matrix2[0,2])],
                  [(x2[1]*camera_matrix2[2,0]-camera_matrix2[1,0]), (x2[1]*camera_matrix2[2,1]-camera_matrix2[1,1]), (x2[1]*camera_matrix2[2,2]-camera_matrix2[1,2])]])
    
    B = np.array([-(x1[0]*camera_matrix1[2,3]-camera_matrix1[0,3]),
                -(x1[1]*camera_matrix1[2,3]-camera_matrix1[1,3]),
                -(x2[0]*camera_matrix2[2,3]-camera_matrix2[0,3]),
                -(x2[1]*camera_matrix2[2,3]-camera_matrix2[1,3])])
    
    state, X = cv2.solve(A, B, flags=cv2.DECOMP_SVD)
    return state, X

def iterative_linear_LS_triangulation(u, p, u1, p1):
    '''
        ref: Triangulation, Hartley, R.I. and Sturm, P.
        http://www.morethantechnical.com/2012/01/04/simple-triangulation-with-opencv-from-harley-zisserman-w-code/
    '''
    wi, wi1 = 1, 1
    X = np.zeros((4,1))
    EPSILON = 0.1
    
    #Hartley suggests 10 iterations at most
    for i in range(10):
        s, aX = linear_LS_Triangulation(u, p, u1, p1)
        X = np.array([aX[0], aX[1], aX[2], 1])
        
        # recalculate weights
        p2x = np.dot(np.transpose(p[2]), X)
        p2x1 = np.dot(np.transpose(p1[2]), X)
        
        # breaking point
        if(abs(wi - p2x) <= EPSILON) and (abs(wi1 - p2x1) <= EPSILON):
            break
        
        # reweight equations and SVD
        A = np.array([[(np.dot(u[0], p[2,0]) - p[0,0])/wi, (np.dot(u[0], p[2,1]) - p[0,1])/wi, (np.dot(u[0], p[2,2]) - p[0,2])/wi],
                     [(np.dot(u[1], p[2,0]) - p[1,0])/wi, (np.dot(u[1], p[2,1]) - p[1,1])/wi, (np.dot(u[1], p[2,2]) - p[1,2])/wi],
                     [(np.dot(u1[0], p1[2,0]) - p1[0,0])/wi1, (np.dot(u1[0], p1[2,1]) - p1[0,1])/wi1, (np.dot(u1[0], p1[2,2]) - p1[0,2])/wi1],
                     [(np.dot(u1[1], p1[2,0]) - p1[1,0])/wi1, (np.dot(u1[1], p1[2,1]) - p1[1,1])/wi1, (np.dot(u1[1], p1[2,2]) - p1[1,2])/wi1]])
                      
        B = np.array([-(np.dot(u[0], p[2,3]) - p[0,3])/wi,
                     -(np.dot(u[1], p[2,3]) - p[1,3])/wi,
                     -(np.dot(u1[0], p1[2,3]) - p1[0,3])/wi1,
                     -(np.dot(u1[1], p1[2,3]) - p1[1,3])/wi1])
        state, aX = cv2.solve(A, B, flags=cv2.DECOMP_SVD)
        X = np.array([aX[0], aX[1], aX[2], 1])
    return X
                      
def triangulate_points(keypts1, keypts2, kinv, kinv1, p, p1):
    
    points = np.zeros((len(keypts1), 3))
    for i in range(len(keypts1)):
    
        # convert to normalized homogeneous coordinates
        kp1 = keypts1[i]
        x1 = np.array([kp1[0], kp1[1], 1])
        x1 = np.dot(kinv, x1)
        
        # convert to normalized homogeneous coordinates
        kp2 = keypts2[i]
        x2 = np.array([kp2[0], kp2[1], 1])
        x2 = np.dot(kinv1, x2)

        # triangulate
        points[i] = iterative_linear_LS_triangulation(x1, p, x2, p1)[:3,]
        
    return points        

def show_cloud_points(pts):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    for i, p in enumerate(pts):
        xs = pts[i][0]
        ys = pts[i][1]
        zs = pts[i][2]
        ax.scatter(xs, ys, zs, color='#054E9F', s=3)
    plt.show()
    
image3 = cv2.imread('./data/Statue1.bmp')
image4 = cv2.imread('./data/Statue2.bmp')
image3 = cv2.cvtColor(image3, cv2.COLOR_BGR2RGB)
image4 = cv2.cvtColor(image4, cv2.COLOR_BGR2RGB)

#plt.subplot(121), plt.imshow(image3), plt.axis('off')
#plt.subplot(122), plt.imshow(image4), plt.axis('off')
#plt.show()

# get feature points
sift2 = cv2.xfeatures2d.SIFT_create()
(keypt3, desc3) = sift2.detectAndCompute(image3, None)
(keypt4, desc4) = sift2.detectAndCompute(image4, None)
BF_MACTHER_DISTANCE = 0.2
matches2 = hw3.brute_force_matcher(desc3, desc4, BF_MACTHER_DISTANCE)
matched_pt_order2 = hw3.sort_matched_points(matches2)
matched_feature_image = hw3.show_matched_image(image3, image4, keypt3, keypt4, matched_pt_order2, draw_line=False)
#plt.imshow(matched_feature_image), plt.axis('off'), plt.show()
imgpts3, imgpts4 = hw3.get_matched_points(matched_pt_order2, keypt3, keypt4)
print("Key Points Number:\n", len(imgpts3))

def get_second_data_cameramat(k, r, t):
    '''
        ref: HW4 lecture page 6
    '''
    T = -np.dot(r, t)
    e = np.array([[r[0,0], r[0,1], r[0,2], T[0]],
              [r[1,0], r[1,1], r[1,2], T[1]],
              [r[2,0], r[2,1], r[2,2], T[2]]])
    C = np.dot(k, e)
    C = C/C[-1,-1]
    return C

k3 = np.array([[5426.566895, 0.678017, 330.096680],
             [0.000000, 5423.133301, 648.950012],
             [0.000000, 0.000000, 1.000000]])
r3 = np.array([[0.140626, 0.989027, -0.045273],
              [0.475766, -0.107607, -0.872965],
              [-0.868258, 0.101223, -0.485678]])
t3 = np.array([67.479439, -6.020049, 40.224911])
cammat3 = get_second_data_cameramat(k3, r3, t3)
print("cammat3\n", cammat3)

k4 = np.array([[5426.566895, 0.678017, 387.430023],
              [0.000000, 5423.133301, 620.616699],
              [0.000000, 0.000000, 1.000000]])
r4 = np.array([[0.336455, 0.940689, -0.043627],
              [0.446741, -0.200225, -0.871970],
              [-0.828988, 0.273889, -0.487611]])
t4 = np.array([62.882744, -21.081516, 40.544052])
cammat4 = get_second_data_cameramat(k4, r4, t4)
print("cammat4\n", cammat4)

cloudpts = triangulate_points(imgpts3, imgpts4, np.linalg.inv(k3), np.linalg.inv(k4), cammat3, cammat4)
show_cloud_points(cloudpts)
