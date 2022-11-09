import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
from IPython.display import Image

import os
import sys

import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    print(f'added {module_path} to sys')
    sys.path.append(module_path)

from endosim_demo.end_utils import create_transform, \
                                  multiply_points_by_matrix, \
                                  multiply_point_by_matrix, pointer_to_mri,\
                                  camera_to_mri, mri_to_camera, add_noise_to_points,\
                                  add_noise_to_params, extract_rigid_body_parameters,\
                                  rigid_body_parameters_to_matrix, \
                                  convert_4x1_to_1x1x3, project_camera_point_to_image, \
                                  create_pnt_ref, create_pnt_ref_in_camera_space, create_pat_ref, \
                                  create_pat_ref_in_camera_space, get_ref_T_tip, calculate_euclid_dist, convert_points_nx3_to_1xnx3


def create_points():
    # creating 4 points in patient coordinates

    z = 200 # head length (about 20cm)
    x = 250 # menton to top of head (about 25cm)
    y = 150 # head bredth (about 15cm)

    points = np.array(
    [
        [-x,-y,z],
        [-x,y,z],
        [x,y,z],
        [x,-y,z],
    ])

    #points_converted = convert_points_nx3_to_1xnx3(points)
    tumour_point = convert_points_nx3_to_1xnx3(np.array([[0,0,0]]))

    return points, tumour_point


def plot_initial_points(points, tumour_point):
    plt.figure()
    plt.scatter(points[:,0], points[:,1])
    plt.scatter(tumour_point[:,:,0], tumour_point[:,:,1])

    plt.title('3D patient points')
    plt.show()


def project_points(points_converted,rvec, tvec, intrinsics, distortion):

    image_points, jacobian = cv2.projectPoints(points_converted, rvec, tvec, intrinsics, distortion)
    # removing extra empty dimension
    image_points = image_points.squeeze()

    return image_points, jacobian


def solve_pnp(points, image_points, intrinsics, distortion):
    success, R, T = cv2.solvePnP(points.astype('float32'),image_points.astype('float32'),intrinsics, distortion, flags=0)
    R, T = R.T, T.T
    return success, R, T

'''
def reverse_projection_pnt_1(point, tvec,rvec, intrinsics):

    params = np.concatenate((tvec, rvec), axis=0)
    # extrinsics
    mat = rigid_body_parameters_to_matrix(params[:,0]) # (4,4)

    new_img = np.append(point,1) # (4,1)

    # apply inverse of intrinsics
    im = new_img@np.linalg.inv(intrinsics)
    im3D = np.append(im, 1)
    # inv of extrinsics to get obj
    result = im3D @  np.linalg.inv(mat)

    return result


def reverse_projection_pnts_1(points, tvec,rvec, intrinsics):

    params = np.concatenate((tvec, rvec), axis=0)
    # extrinsics
    mat = rigid_body_parameters_to_matrix(params[:,0]) # (4,4)

    new_img = np.concatenate((points, np.ones((points.shape[0],1))), axis=1)

    # apply inverse of intrinsics
    im = new_img@np.linalg.inv(intrinsics)
    im3D = np.concatenate((im, np.ones((points.shape[0],1))), axis=1)
    # inv of extrinsics to get obj
    result = im3D @  np.linalg.inv(mat)

    return result
'''


def reverse_projection_pnts(points, tvec,rvec, intrinsics):

    params = np.concatenate((tvec, rvec), axis=0)
    # extrinsics
    mat = rigid_body_parameters_to_matrix(params[:,0]) # (4,4)

    if len(points.shape) == 1:
        new_img = np.append(points,[1]) # [1,1] points is (2,)
    else:
        new_img = np.concatenate((points, np.ones((points.shape[0],1))), axis=1) #points is (2,n)

    # apply inverse of intrinsics
    im = new_img@np.linalg.inv(intrinsics)
    if len(im.shape)==1:
        im3D = np.append(im, [1])
    else:
        im3D = np.concatenate((im, np.ones((points.shape[0], 1))), axis=1)

    # inv of extrinsics to get obj

    #result = im3D @  np.linalg.inv(mat)

    return im


def Get3Dfrom2D(List2D,t, R, K, d=570):
    # List2D : n x 2 array of pixel locations in an image
    # K : Intrinsic matrix for camera
    # R : Rotation matrix describing rotation of camera frame
    #     w.r.t world frame.
    # t : translation vector describing the translation of camera frame
    #     w.r.t world frame
    # [R t] combined is known as the Camera Pose.
    # List2D = np.array(List2D)
    R,_ = cv2.Rodrigues(R)
    List3D = []
    # t.shape = (3,1)

    if len(List2D.shape)==1:
        List2D = np.array([List2D])

    for p in List2D:
        # Homogeneous pixel coordinate
        p = np.array([p[0], p[1], 1]).T; p.shape = (3,1)
        print("pixel: \n", p)

        # Transform pixel in Camera coordinate frame
        p_cam = np.linalg.inv(K) @ p
        print("point in cam coords : \n", p_cam, p_cam.shape)

        # Transform pixel in World coordinate frame
        #p_world = t + (R.T@p_cam)
        p_world = (R @ (p_cam-t.T))
        print("point in world coords : \n", p_world, t.shape, R.shape, p_cam.shape)

        # Transform camera origin in World coordinate frame
        cam = np.array([0,0,0]).T; cam.shape = (3,1)
        #cam_world = t + R @ cam
        cam_world = (R @ (cam-t.T))
        # print("cam_world : \n", cam_world)

        # Find a ray from camera to 3d point
        vector = p_world - cam_world
        unit_vector = vector / np.linalg.norm(vector)
        # print("unit_vector : \n", unit_vector)

        # Point scaled along this ray
        p3D = cam_world + d * unit_vector
        # print("p3D : \n", p3D)
        List3D.append(p3D)

    return np.array(List3D).squeeze()


def inverse_proj(img_points, tvec,rvec, intrinsics):
    rvec = rvec.T
    tvec=tvec.T

    rotation_matrix, _ = cv2.Rodrigues(rvec)

    D3_pnts = []
    for point in img_points:
        uv_point = np.ones((3,1))
        uv_point[0,0] = point[0]
        uv_point[1, 0] = point[1]

        s, zConst = 1 , 0
        tempMat = np.linalg.inv(rotation_matrix) @ np.linalg.inv(intrinsics) @ uv_point
        tempMat2 = np.linalg.inv(rotation_matrix) @ tvec.T

        #s = zConst + tempMat2[2, 0]
        #s /= tempMat[2, 0]

        wcPoint = np.linalg.inv(rotation_matrix) @ (s * np.linalg.inv(intrinsics) @ uv_point - tvec)
        D3_pnts.append(wcPoint)

    return D3_pnts


def main():
    points, tumour_point = create_points()
    #plot_initial_points(points, tumour_point)

    ## PROJECTING POINTS (3D->2D)
    intrinsics = np.loadtxt('/Users/aure/Documents/i4health/project/endosim_demo/calibration/intrinsics.txt')
    distortion = np.loadtxt('/Users/aure/Documents/i4health/project/endosim_demo/calibration/distortion.txt')

    points_converted = convert_points_nx3_to_1xnx3(points)

    rvec = np.zeros((1,3))
    tvec = np.zeros((1,3))
    image_points, _ = project_points(points_converted, rvec, tvec, intrinsics, distortion)

    ## ADDING NOISE IN 2D
    image_points_noisy = add_noise_to_points(image_points,sigma)

    ## SOLVING TRANSFORMS FOR PROJECTIONS
    # finding transform to get us from 2D image points to 3D object
    success_true, rotation_vector_true, translation_vector_true = solve_pnp(points, image_points, intrinsics, distortion)

    success_true, rotation_vector_true, translation_vector_true = solve_pnp(points, image_points, intrinsics, distortion)
    # finding transform to get us from NOISY 2D image points to 3D object
    success, rotation_vector, translation_vector = solve_pnp(points, image_points_noisy, intrinsics, distortion)

    ## PROJECTING with new transforms
    point_projected2D_true, _ = project_points(points_converted, rotation_vector_true, translation_vector_true, intrinsics, distortion)
    point_projected2D_n, _ = project_points(points_converted, rotation_vector, translation_vector, intrinsics, distortion)

    # PROJECTING tumour with new transforms
    tumour_projected2D_true, _ = project_points(tumour_point, rotation_vector_true, translation_vector_true, intrinsics, distortion)
    tumour_projected2D_n, _ = project_points(tumour_point, rotation_vector, translation_vector, intrinsics, distortion)


    ## REVERSE PROJECTION
    '''
    # real
    result_tum = reverse_projection_pnts(tumour_projected2D_true, translation_vector_true,rotation_vector_true, intrinsics)
    print(result_tum)

    result_pnts = reverse_projection_pnts(point_projected2D_true, translation_vector_true,rotation_vector_true, intrinsics)
    print(result_pnts)

    # noisy
    result_tum_n = reverse_projection_pnts(tumour_projected2D_n, translation_vector_true,rotation_vector_true, intrinsics)
    print(result_tum_n)

    result_pnts_n = reverse_projection_pnts(point_projected2D_n, translation_vector_true,rotation_vector_true, intrinsics)
    print(result_pnts_n)
    '''

    # method 2-----
    # result_tum = Get3Dfrom2D(np.array([tumour_projected2D_true]), tvec,rvec, intrinsics)
    # print(result_tum)
    result_tum = Get3Dfrom2D(tumour_projected2D_true, translation_vector_true,rotation_vector_true, intrinsics)

    result_pnts = Get3Dfrom2D(point_projected2D_true, translation_vector_true,rotation_vector_true, intrinsics)
    print(result_pnts)

    # noisy
    result_tum_n = Get3Dfrom2D(np.array([tumour_projected2D_n]), translation_vector_true,rotation_vector_true, intrinsics)
    print(result_tum_n)

    result_pnts_n = Get3Dfrom2D(point_projected2D_n, translation_vector_true,rotation_vector_true, intrinsics)
    print(result_pnts_n)
    '''
    ## method 3 -------
    result_tum = inverse_proj(np.array([tumour_projected2D_true]), translation_vector_true,rotation_vector_true, intrinsics)
    print(result_tum)

    result_pnts = inverse_proj(point_projected2D_true, translation_vector_true,rotation_vector_true, intrinsics)
    print(result_pnts)

    # noisy
    result_tum_n = inverse_proj(np.array([tumour_projected2D_n]), translation_vector_true,rotation_vector_true, intrinsics)
    print(result_tum_n)

    result_pnts_n = inverse_proj(point_projected2D_n, translation_vector_true,rotation_vector_true, intrinsics)
    print(result_pnts_n)
    '''

    ################## plotting
    # ORIGINAL
    plt.figure()
    plt.scatter(points[:,0], points[:,1])
    plt.scatter(result_pnts[:,0], result_pnts[:,1])
    plt.scatter(result_pnts_n[:,0], result_pnts_n[:,1])

    plt.scatter(tumour_point[:,:,0], tumour_point[:,:,0])
    plt.scatter(result_tum[0], result_tum[1])
    plt.scatter(result_tum_n[0], result_tum_n[1])

    plt.legend(['original 3D points', 'new 3D points', 'new noisy points', 'original 3D tumour', 'new 3D tumour', 'new noisy tumour'])
    plt.title('original')
    plt.show()

    print(points)
    print(result_pnts)



    '''
    # UNPROJECTED
    plt.figure()
    plt.scatter(result_pnts[:,0], result_pnts[:,1])
    plt.scatter(result_pnts_n[:,0], result_pnts_n[:,1])

    plt.scatter(result_tum[0], result_tum[1])
    plt.scatter(result_tum_n[0], result_tum_n[1])

    plt.legend(['original image points', 'projected points', 'original tumour', 'noisy tumour'])
    plt.title('reprojected')
    plt.show()

    ##
    '''

    '''
    params = np.concatenate((translation_vector_true, rotation_vector_true), axis=0)
    mat = rigid_body_parameters_to_matrix(params[:,0]) # (4,4)
    
    ones=np.ones((1,))
    new_img = np.concatenate((tumour_projected2D,ones), axis=0) # (4,1)
    new_img_noise = np.concatenate((tumour_projected2D_true,ones), axis=0) # (4,1)


    ones_pnts = np.ones((4,2))
    pnts = np.concatenate((point_projected2D_true,ones_pnts), axis=1) # (4,1)
    pnts_noise = np.concatenate((point_projected2D,ones_pnts), axis=1) # (4,1)

    obj_3D = new_img @ np.linalg.inv(mat)
    obj_3D_noise = new_img_noise @ np.linalg.inv(mat)

    pnts_3D = pnts @ np.linalg.inv(mat)
    pnts_3D_noise =  pnts_noise @ np.linalg.inv(mat)
    '''


if __name__=='__main__':
    sigma=1
    main()