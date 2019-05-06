import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import answer
import HW4

DEBUG_IMAGE_INDEX = 1

if(DEBUG_IMAGE_INDEX==1):
    image1 = cv2.imread('./data/Mesona1.JPG')
    image2 = cv2.imread('./data/Mesona2.JPG')
elif(DEBUG_IMAGE_INDEX == 2):
    image1 = cv2.imread('./data/Statue1.bmp')
    image2 = cv2.imread('./data/Statue2.bmp')
image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

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

############################################################################################

a_x1, a_x2, _ = answer.sift_detector(image1, image2)

print('a_x1.shape\n', len(a_x1))

a_x1 = np.float64(a_x1)
a_x2 = np.float64(a_x2)

print('a_x1.shape\n', a_x1.shape)

# normalization the key points
a_normalpts1, a_normalpts2, nomalmat1, normalmat2 = answer.get_normalize(a_x1, a_x2, image1.shape, image2.shape)
answer_f, mask = answer.get_fundamental(a_normalpts1, a_normalpts2, nomalmat1, normalmat2)
anser_e = HW4.get_essential_mat(intrinsic_matrix1, intrinsic_matrix2, answer_f)
a_inlinerpts1 = a_x1[mask.ravel()==1]
a_inlinerpts2 = a_x2[mask.ravel()==1]
X = answer.cal_P(anser_e, a_inlinerpts1, a_inlinerpts2, intrinsic_matrix1, intrinsic_matrix2)


# x1, x2 = HW4.get_feature_points(image1, image2)
RANSAC_INLINER_THRESHOLD = 0.000005
RANSAC_SAMPLE_NUMBER = 4000
fundamentalmat, essentialmat, inlinerpts1, inlinerpts2 = HW4.find_fundamental_by_RANSAC(a_x1, a_x2, image1, image2, RANSAC_INLINER_THRESHOLD, RANSAC_SAMPLE_NUMBER)
project_pts, p1, p2 = HW4.triangulate_points(essentialmat,
                                         inlinerpts1, inlinerpts2,
                                         intrinsic_matrix1, intrinsic_matrix2,
                                         rotation_matrix1, rotation_matrix2,
                                         transform_matrix1, transform_matrix2)

print('answer f\n', answer_f)
print('our f\n', fundamentalmat)
print('anser e\n', anser_e)
print('our e\n', essentialmat)
print('our p2\n', p2)

HW4.show_cloud_points(X)
HW4.show_cloud_points(project_pts)