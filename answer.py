import cv2
import numpy as np
from matplotlib import pyplot as plt
import random 
from mpl_toolkits.mplot3d import axes3d, Axes3D

K1 = np.array([[5426.566895, 0.678017   , 330.096680],
               [0.000000   , 5423.133301, 648.950012],
               [0.000000   , 0.000000   , 1.000000  ]])
R1 = np.array([[0.140626 , 0.989027 ,-0.045273],
               [0.475766 ,-0.107607 ,-0.872965],
               [-0.868258, 0.101223 ,-0.485678]] )
t1 = np.array( [67.479439  ,-6.020049   ,40.224911  ])
K2 = np.array([[5426.566895, 0.678017   , 387.430023],
               [0.000000   , 5423.133301, 620.616699],
               [0.000000   , 0.000000   , 1.000000  ]])
R2 = np.array([[ 0.336455 , 0.940689 ,-0.043627],
               [ 0.446741 ,-0.200225 ,-0.871970],
               [-0.828988 , 0.273889 ,-0.487611]] )
t2 = np.array( [62.882744  ,-21.081516  ,40.544052  ])
K  = np.array([[1.4219     , 0.0005     , 0.5092],
               [0          , 1.4219     , 0.3802],
               [0          , 0          , 0.0010]] )/0.001
K_11 = np.vstack( (np.eye(3, dtype=int), np.array([0,0,0]))).T
#random.seed('foobar')  
img1 = cv2.imread('./data/Mesona1.JPG',0)  # left image set 1
img2 = cv2.imread('./data/Mesona2.JPG',0)  # right
#img1 = cv2.imread('./data/Statue1.bmp',0)   # left image set 2
#img2 = cv2.imread('./data/Statue2.bmp',0)   # right 

def sift_detector( im1, im2):
    sift = cv2.xfeatures2d.SIFT_create()
    
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    
    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)
    
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    
    good = []
    pts1 = []
    pts2 = []
    
    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.8*n.distance:
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
	
    return pts1,pts2,good	

def get_normalize(x1, x2, shape1, shape2):
    '''
    normalize (x, y) coordinate to -1~1 and get the transformation matrix
    '''
    #print("x1",x1[:,0:2].shape)# (402,2)
    #print(shape1)
    ones = np.ones(x1.shape[0]).reshape(-1, 1)
    x1 = np.concatenate((x1, ones), axis=1).T
    x2 = np.concatenate((x2, ones), axis=1).T
    #print("x1",x1.shape)
    T1 = np.array([[2/shape1[1], 0, -1],
                   [0, 2/shape1[0], -1],
                   [0, 0, 1]])
    T2 = np.array([[2/shape2[1], 0, -1],
                   [0, 2/shape2[0], -1],
                   [0, 0, 1]])
    x1 = np.dot(T1, x1)
    x2 = np.dot(T2, x2)
    # print("x1\n",x1.shape)
    # print('x1[0]\n', x1[0].shape, '--end--')
    #print("new pts min and max: ",np.min(x1),np.max(x1))
    return x1, x2, T1, T2

def get_fundamental(x1, x2, T1, T2):
    # x1 = [u v 1] x2 = [u' v' 1]
    best_F = np.zeros((3,3))
    ransac_iter = 2000
    threshold_distance = 0.000005
    thre_inlier = 0
    sample = x1[0].shape[0]
    #print(x1[0].shape)
    for iter in range(ransac_iter):
        indexes = random.sample(range(x1[0].shape[0]), 8)   
        A = []
        inlier = 0
        for i in indexes:
            #[ uu' vu' u' uv' vv' v' u v 1]
            A.append([x1[0][i]*x2[0][i], x1[1][i]*x2[0][i], x1[2][i]*x2[0][i],
                      x1[0][i]*x2[1][i], x1[1][i]*x2[1][i], x1[2][i]*x2[1][i],
                      x1[0][i]*x2[2][i], x1[1][i]*x2[2][i], x1[2][i]*x2[2][i]])
        A = np.array(A)
        #print (A)
        U,S,V = np.linalg.svd(A)
        F = V[-1].reshape(3,3)
        #print("U\n", U, "\nU[-1]\n" ,U[:,-1])
        U,S,V = np.linalg.svd(F)
        #print("diag", S)
        S[2] = 0
        #print("diag", np.diag(S))
        F = np.dot(U, np.dot(np.diag(S), V))
        # sampson distance
        Fx1 = np.dot(F.T,x1)
        Fx2 = np.dot(F,x2)
        denom = Fx1[0]**2+Fx1[1]**2+Fx2[0]**2+Fx2[1]**2
        sampson = np.diag( np.dot(x2.T, np.dot(F,x1)) )**2/denom
        #print(sampson.shape)
        m=[]
        for j in range (sampson.shape[0]):
            if sampson[j]<= threshold_distance:
                inlier = inlier+1
                m.append(1)
            else:
                m.append(0)
                				
        
        if inlier>thre_inlier:
            #Denormalized
            F = np.dot(T2.T,np.dot(F,T1))
            best_F = F
            thre_inlier = (inlier)
            mask = np.array(m)
            #print("inlier: ",inlier, "x1_shape", mask.shape)
        
        
    return best_F/best_F[-1, -1], mask #best_F/best_F[-1, -1]
	
pts1, pts2, matches = sift_detector(img1,img2)	
pts1 = np.float64(pts1)
pts2 = np.float64(pts2)	

F_cv, mask_cv = cv2.findFundamentalMat(pts1,pts2,method =cv2.FM_8POINT + cv2.FM_RANSAC)
n_pts1, n_pts2, T1, T2 = get_normalize(pts1, pts2, img1.shape, img2.shape) #img1.shape = (height, width)
F, mask = get_fundamental(n_pts1, n_pts2, T1, T2)

print("F_CV: \n",F_cv)
print("F: \n",F)
np.savetxt("fundamental.txt", F, delimiter=",")

# only inlier points
pts1 = pts1[mask.ravel()==1]
pts2 = pts2[mask.ravel()==1]

np.savetxt("2dpoint.csv", pts1, delimiter=",")
np.savetxt("point1.txt", pts1, delimiter=",")
np.savetxt("point2.txt", pts2, delimiter=",")

def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2

lines1 = cv2.computeCorrespondEpilines(pts2[:30].reshape(-1,1,2), 2,F)
lines1 = lines1.reshape(-1,3)
img5,img6 = drawlines(img1,img2,lines1,pts1[:30],pts2[:30])

# Find epilines corresponding to points in left image (first image) and
# drawing its lines on right image
lines2 = cv2.computeCorrespondEpilines(pts1[:30].reshape(-1,1,2), 1,F)
lines2 = lines2.reshape(-1,3)
img3,img4 = drawlines(img2,img1,lines2,pts2[:30],pts1[:30])

# plt.subplot(121),plt.imshow(img4),plt.title("left image")
# plt.subplot(122),plt.imshow(img3),plt.title("right image")
# plt.show()

# plt.subplot(121),plt.imshow(img5),plt.title("left image")
# plt.subplot(122),plt.imshow(img6),plt.title("right image")
# plt.show()

### check epipolar constraint
ones = np.ones(pts1.shape[0]).reshape(-1, 1)
print("one.shape\n",ones.shape)
pts1_cal_homo = np.concatenate((pts1, ones), axis=1) # (395,3)
print("pts1_cal_homo\n",pts1_cal_homo.shape)
print("pts_cal_hom[0]", pts1_cal_homo[0])
pts2_cal_homo = np.concatenate((pts2, ones), axis=1)

print("epi constraint: ")
for i in range(10):
    print(np.dot(pts2_cal_homo[i,:],np.dot(F,pts1_cal_homo[i,:].T)))

	
	
### Essential Matrix
E = np.dot( K.T , np.dot(F,K))   #image set 1
print("E: \n",E)
def cal_P (E, x1,x2,K1,K2):
    '''
        E : essential matrix 3*3
        x1: match set1  N*2
        x2: match set2  N*2
	    return X: 4*N (last row = 1)
    '''
    W = np.array([[0 , -1 , 0],
                  [1 ,  0 , 0],
			      [0 ,  0 , 1]] )
    U , S , V = np.linalg.svd(E)
    P1 =  np.dot(K1, np.vstack( (np.eye(3, dtype=int), np.array([0,0,0]))).T)
    x1_num = x1.shape[0]
    print(x1_num)

    R1= np.dot(U,np.dot(W,V))
    R2= np.dot(U,np.dot(W.T,V))
    
    T1= U[:,-1].reshape(3,1)
    T2= -U[:,-1].reshape(3,1)
    
    P2_1 = np.concatenate((R1,T1),axis=1)
    P2_2 = np.concatenate((R1,T2),axis=1)
    P2_3 = np.concatenate((R2,T1),axis=1)
    P2_4 = np.concatenate((R2,T2),axis=1)
    
    lll = [ np.dot(K2,P2_1), np.dot(K2,P2_2), np.dot(K2,P2_3), np.dot(K2,P2_4)]
    X = []
    C =[]
    for p2 in lll:
        #print(p2)
        count=0
        t,RR1,RR2= get_tR(E)

        # ready x1 points
        for i in range(x1.shape[0]):
            A = np.array([
                        x1[i,0]*P1[2,:]-P1[0,:], 
                        x1[i,1]*P1[2,:]-P1[1,:],
                        x2[i,0]*p2[2,:]-p2[0,:],
                        x2[i,1]*p2[2,:]-p2[1,:]])
            # print("A\n", A)
            U,S,V = np.linalg.svd(A)
            # print("U\n", U)
            # print("S\n", S)
            # print("V\n", V)
            x = V[-1]/V[-1,3]
            X.append(x)

            # if this point is front on our camera
            result = np.dot(x[0:3], RR1[-1,:])
            if result>0:
                count=count+1

        C.append(count)    
            
    C=np.array(C)        
    argmax = np.argmax(C)
    X = np.array(X)
    
    return X[argmax*x1_num:(argmax+1)*x1_num,:]
    

def get_tR(E):
    w = np.array([[0.0, -1.0, 0.0],
                 [1.0, 0.0, 0.0],
                 [0.0, 0.0, 1.0]])
    z = np.array([[0.0, 1.0, 0.0], 
                  [-1.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0]])
    
    U, S, V = np.linalg.svd(E)
    S[0] = (S[0] + S[1]) / 2
    S[1] = S[0]
    S[2] = 0
    
    t = U.dot(z).dot(U.T)
    t = np.array([-t[1, 2], t[0, 2], -t[0, 1]])
    
    R1 = U.dot(w.T.dot(V))
    R2 = U.dot(w.dot(V))
    
    if np.linalg.det(R1) == -1.0:
        R1 = R1 * (-1.0)
        R2 = R2 * (-1.0)
    
    return t, R1, R2

    
#########
# X = cal_P (E,pts1,pts2,K,K)
# np.savetxt("3dpoint.csv", X, delimiter=",")
# fig = plt.figure()
# fig.suptitle('3D reconstructed', fontsize=16)
# ax = fig.add_subplot(111, projection='3d')
#ax.scatter(X[400:,0], X[400:,1], X[400:,2], marker='b.')

# ax.plot(X[:,0], X[:,1], X[:,2], 'b.')
# ax.set_xlabel('x axis')
# ax.set_ylabel('y axis')
# ax.set_zlabel('z axis')
# #ax.view_init(elev=135, azim=90)
# plt.show()

#camera_matrix = np.dot( K, np.vstack( (np.eye(3, dtype=int), np.array([0,0,0]))).T )
tt1,RR1,RR2 = get_tR(E)
print (RR1.T)
camera_matrix = np.dot( K, np.vstack( (RR1.T, tt1)).T )
print(camera_matrix)
np.savetxt("camera.csv", camera_matrix, delimiter=",")
