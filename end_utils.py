import copy
import numpy as np
import cv2
import random
from scipy.spatial.transform import Rotation as spr
from matplotlib import pyplot as plt
import sksurgerycore.algorithms.procrustes as pro
import sksurgerycore.transforms.matrix as mu


def create_transform(array_of_parameters):
    """
    Returns a homogeneous rigid transformation as 4x4 np.ndarray.

    :param array_of_parameters: list of [tx, ty, tz, rx, ry, rz] where rotations are in degrees.
    """
    rotation = mu.construct_rotm_from_euler(array_of_parameters[3],
                                            array_of_parameters[4],
                                            array_of_parameters[5],
                                            sequence='xyz',
                                            is_in_radians=False)
    translation = np.zeros((3,1))
    translation[0][0] = array_of_parameters[0]
    translation[1][0] = array_of_parameters[1]
    translation[2][0] = array_of_parameters[2]

    transform = mu.construct_rigid_transformation(rotation, translation)
    return transform


def multiply_points_by_matrix(matrix_4x4, matrix_of_points, do_transpose):
    """
    Multiplies all points by the same matrix.

    :param matrix_4x4: Numpy ndarray, 4x4, containing homogenous, rigid transformation
    :param matrix_of_points: Numpy ndarray, 4xN, containing N points as 4D homogeneous column vectors.
    :param do_transpose: if true, we also transpose
    """
    input_matrix = matrix_of_points

    if do_transpose:
        input_matrix = np.transpose(matrix_of_points)

    result = np.matmul(matrix_4x4, input_matrix)

    if do_transpose:
        result = np.transpose(result)

    return result


def multiply_point_by_matrix(transform, point_as_array):
    """
    multiply_point_by_matrix(transform, point_as_array)
    multiplies a 3D point by a 4X4 transformation

    Args:
        transform: 4x4 np array representing the transform we want to do on our point
        point_as_array: 3x1 np array representing the location of a point in 3D

    Returns:
        transformed 3D point in homogenous coordinates
    """

    # convert 3D point to homogenous coordinates
    in_point = np.ones((4, 1))
    in_point[0][0] = point_as_array[0]
    in_point[1][0] = point_as_array[1]
    in_point[2][0] = point_as_array[2]

    # multiply homogenous coord by transform
    out_point = np.matmul(transform, in_point)
    return out_point


##### POINTER SPECIFIC UTILS FUNCTIONS
def pointer_to_mri(Cam_T_PntRef, Cam_T_PatRef, PatRef_T_MRI, point_in_pointer_coords=[0, 0, 0]):
    """
    Converts a point in pointer space to MRI space.

    Args:
        Cam_T_PntRef: 4x4 transform from pointer's reference to camera coords
        Cam_T_PatRef: 4x4 transform from patient's reference to camera coords
        PatRef_T_MRI: 4x4 transform from MRI to patient reference coords
        point_in_pointer_coords: 3x1/1x3 list representing 3D point we are transforming from pointer to MRI space

    Returns:
        point_in_pointer_coords but in MRI space
    """

    transform = np.linalg.inv(PatRef_T_MRI) @ np.linalg.inv(Cam_T_PatRef) @ Cam_T_PntRef
    out_point = multiply_point_by_matrix(transform, point_in_pointer_coords)
    return out_point


##### ENDOSCOPE SPECIFIC UTILS FUNCTIONS


def get_transform_mri_cam(MRI_T_PatRef , PatRef_T_Cam , Cam_T_EndRef , Hand_T_Eye):
    transform = MRI_T_PatRef @ PatRef_T_Cam @ Cam_T_EndRef @ Hand_T_Eye
    return transform

def camera_to_mri(MRI_T_PatRef, PatRef_T_Cam, Cam_T_EndRef, Hand_T_Eye, point_in_camera_coords=[0, 0, 0]):
    """

    Args:
        Cam_T_Endo: 4x4 transform from endoscope reference coords to IR camera coords 
        Cam_T_PatRef: 4x4 transform from patient reference coords to IR camera coords
        PatRef_T_MRI: 4x4 transform from MRI to patient reference coords
        Hand_T_Eye: 4x4 transform from endoscope image coords to endoscope ref coords
        point_in_camera_coords: 3x1/1x3 list representing 3D point we are transforming from camera to MRI space

    Returns:

    """
    """
    Converts a point in camera space to MRI space.
    """
    transform = get_transform_mri_cam(MRI_T_PatRef , PatRef_T_Cam , Cam_T_EndRef , Hand_T_Eye)
    #transform = Hand_T_Eye

    out_point = multiply_point_by_matrix(transform, point_in_camera_coords)
    return out_point


def mri_to_camera(MRI_T_PatRef, PatRef_T_Cam, Cam_T_EndRef, Hand_T_Eye,  point_in_mri_coords=[0, 0, 0]):
    """

    Args:
        Cam_T_Endo:
        Cam_T_PatRef:
        PatRef_T_MRI:
        Eye_T_Hand:
        point_in_mri_coords:

    Returns:

    """
    """
    Converts a point in camera space to MRI space.
    """
    transform = get_transform_mri_cam(MRI_T_PatRef , PatRef_T_Cam , Cam_T_EndRef , Hand_T_Eye)
    transform = np.linalg.inv(transform)
    #transform = Hand_T_Eye @ EndRef_T_Cam @ Cam_T_PatRef @ PatRef_T_MRI

    # transform = np.linalg.inv(transform)
    out_point = multiply_point_by_matrix(transform, point_in_mri_coords)
    return out_point


#############
def extract_rigid_body_parameters(matrix):
    """
    extract_rigid_body_parameters(matrix)
    extracts parameters from transformation matrix

    Args:
        matrix: 4x4 transformation matrix

    Returns:
        list of all extracted parameters from matrix

    """
    t = matrix[0:3, 3]
    r = matrix[0:3, 0:3]
    rot = spr.from_matrix(r)
    euler = rot.as_euler('zyx', degrees=True)
    return [euler[0], euler[1], euler[2], t[0], t[1], t[2]]


def rigid_body_parameters_to_matrix(params):
    """
    rigid_body_parameters_to_matrix(params)
    converts a list of rigid body parameters to transformation matrix

    Args:
        params: list of rigid body parameters

    Returns:
        4x4 transformation matrix of these parameters

    """
    matrix = np.eye(4)
    r = (spr.from_euler('zyx', [params[0], params[1], params[2]], degrees=True)).as_matrix()
    matrix[0:3, 0:3] = r
    matrix[0][3] = params[3]
    matrix[1][3] = params[4]
    matrix[2][3] = params[5]
    return matrix


def convert_4x1_to_1x1x3(p_41):
    p_113 = np.zeros((1,1,3))
    p_113[0][0][0] = p_41[0][0]
    p_113[0][0][1] = p_41[1][0]
    p_113[0][0][2] = p_41[2][0]
    return p_113


def project_camera_point_to_image(point, intrinsics, distortion):
    """
    project_camera_point_to_image(point, intrinsics, distortion)
    projects a 3D point into 2D space

    Args:
        point: 4x1 np array of homogeneous 3D point we are projecting
        intrinsics: intrinsics matrix of camera
        distortion: distortion coeffs of camera

    Returns:
        2D projected point
    """
    rvec = np.zeros((1,3))
    tvec = np.zeros((1,3))
    image_points, jacobian = cv2.projectPoints(convert_4x1_to_1x1x3(point), rvec, tvec, intrinsics, distortion)
    return image_points[0][0] # returns a list


def project_camera_points_to_image(points, intrinsics, distortion):
    """
    TO ADD
    """
    rvec = np.zeros((1,3))
    tvec = np.zeros((1,3))
    # convert_4x1_to_1x1x3(point)

    #for idx in range(points.shape[0]):
        
    #    points[idx,:] = convert_4x1_to_1x1x3(np.array(points[idx,:]).T)

    image_points, jacobian = cv2.projectPoints(points, rvec, tvec, intrinsics, distortion)

    return image_points# returns a list


def get_ref_T_tip(pointer_length, dimension):
    """
    function to calculate transform between pointer's tip to pointer's reference

    Parameters
    ----------
    pointer_length- length of pointer in mm
    dimension- dimension along which to add offset (depends which coordinate system we're using- i.e camera or pointer)

    Returns
    -------
    ref_T_tip: The transformation matrix to go from the pointer's tip to the reference of the pointer.

    """
    if dimension == 'z':
        ref_T_tip = create_transform([0, 0, pointer_length, 0, 0, 0]) # create transform of all points depending on pointer's length
    elif dimension == 'x':
        ref_T_tip = create_transform([pointer_length, 0, 0, 0, 0, 0])
    return ref_T_tip


####################### REFERENCE DATA
def create_pnt_ref():

    # Creating pointer reference (from datasheet). Using homogenous (4 numbers, x,y,z,1) as row vectors.
    pnt_ref =  np.zeros((4, 4))

    # marker b
    pnt_ref[1][2] = 50 # z

    # marker c
    pnt_ref[2][1] = 25  # y
    pnt_ref[2][2] = 100 # z

    # marker d
    pnt_ref[3][1] = -25 # y
    pnt_ref[3][2] = 135 # z

    # adding 1 to 3rd dimension to turn to homogeneous coordinates
    pnt_ref[0][3] = 1
    pnt_ref[1][3] = 1
    pnt_ref[2][3] = 1
    pnt_ref[3][3] = 1

    return pnt_ref


def create_pnt_ref_in_camera_space():

    pnt_ref_in_camera_space =  np.zeros((4, 4))

    # marker b
    pnt_ref_in_camera_space[1][0] = 50 # x
    # marker c
    pnt_ref_in_camera_space[2][0] = 100 # x
    pnt_ref_in_camera_space[2][1] = 25 # should this be -y?------- to check
    # marker d
    pnt_ref_in_camera_space[3][0] = 135 # x
    pnt_ref_in_camera_space[3][1] = -25 # should this be -y?----- to check

    # adding 1 to third dimension to make homogeneous coords
    pnt_ref_in_camera_space[0][3] = 1
    pnt_ref_in_camera_space[1][3] = 1
    pnt_ref_in_camera_space[2][3] = 1
    pnt_ref_in_camera_space[3][3] = 1
    return pnt_ref_in_camera_space


def create_pat_ref():
    # Defining reference coordibates in ref coords (from datasheet)
    #A: x=0.00, y= 0.00, z=0.00
    #B: x=0.00, y= 28.59, z=41.02
    #C: x=0.00, y= 00.00, z=88.00
    #D: x=0.00, y=-44.32, z=40.45

    # Encoding the reference marker points into a numpy matrix
    pat_ref = np.zeros((4, 4))

    # marker b
    pat_ref[1][1] = 28.59 # y
    pat_ref[1][2] = 41.02 # z

    # marker c
    pat_ref[2][2] = 88 # z

    # marker d
    pat_ref[3][1] = -44.32 # y
    pat_ref[3][2] = 40.45 # z

    # adding 1 to last row to make coordinates homogenous
    pat_ref[0][3] = 1.0
    pat_ref[1][3] = 1.0
    pat_ref[2][3] = 1.0
    pat_ref[3][3] = 1.0
    return pat_ref


def create_pat_ref_in_camera_space():
    # Encoding the reference marker points into a numpy matrix, in camera space.
    pat_ref_in_camera_space = np.zeros((4, 4))

    #  TO ASK MATT-------> Which direction????
    pat_ref_in_camera_space[1][1] = 41.02 #
    pat_ref_in_camera_space[1][2] = 28.59

    pat_ref_in_camera_space[2][1] = 88

    pat_ref_in_camera_space[3][1] = 40.45
    pat_ref_in_camera_space[3][2] = -44.32

    '''
    
    # marker b
    pat_ref[1][1] = 28.59 # y
    pat_ref[1][2] = 41.02 # z
    
    # marker c
    pat_ref[2][2] = 88 # z
    
    # marker d
    pat_ref[3][1] = -44.32 # y
    pat_ref[3][2] = 40.45 # z
    '''

    print(pat_ref_in_camera_space)

    # converting to homogenous coords
    pat_ref_in_camera_space[0][3] = 1.0
    pat_ref_in_camera_space[1][3] = 1.0
    pat_ref_in_camera_space[2][3] = 1.0
    pat_ref_in_camera_space[3][3] = 1.0

    return pat_ref_in_camera_space


###### NOISE
def add_noise_to_points(points_in, sigma):
    """
    add_noise_to_points(points_in, sigma)
    adding noise to 3D points

    Args:
        points_in: 3xN matrix of points we are adding noise to
        sigma: standard deviation of normal distribution defining noise distribution

    Returns:
        same points but with random noise added to them
    """
    
    points_out = np.zeros((points_in.shape))
    for r in range(points_in.shape[0]):
        for c in range(points_in.shape[1]):
            points_out[r][c] = points_in[r][c] + random.normalvariate(0, sigma)
    return points_out


def add_noise_to_params(params, sigma):
    """
    add_noise_to_params(params, sigma)
    adds noise to parameters (translation, rotation etc)

    Args:
        params: list of all parameters (translations in xyz, rotations in etc.)
        sigma: standard deviation of normal noise to add

    Returns:
        parameters with noise added on them
    """
    params_out = copy.deepcopy(params)
    for i, p in enumerate(params):
        params_out[i] = params[i] + random.normalvariate(0, sigma)
    return params_out


def calculate_euclid_dist(pointer_tip_in_mri_space,tumour_in_mri_space ):
    """
    calculate_euclid_dist(pointer_tip_in_mri_space,tumour_in_mri_space )
    calculates euclidean distance between two 3D points (euclid_dist is x^2+y^2+x^2)

    Args:
        pointer_tip_in_mri_space: 4x1 np array of where we think the tumor is
        tumour_in_mri_space: 4x1 np array of where the tumor actually is

    Returns:
        euclidean distance between the two points
    """

    dist =  (pointer_tip_in_mri_space[0] - tumour_in_mri_space[0]) \
            * (pointer_tip_in_mri_space[0] - tumour_in_mri_space[0]) \
            + (pointer_tip_in_mri_space[1] - tumour_in_mri_space[1]) \
            * (pointer_tip_in_mri_space[1] - tumour_in_mri_space[1]) \
            + (pointer_tip_in_mri_space[2] - tumour_in_mri_space[2]) \
            * (pointer_tip_in_mri_space[2] - tumour_in_mri_space[2])

    return dist


###### converting shapes
def convert_points_1x3_to_1x4x3(p_13):
    p_113 = np.zeros((1,1,3))
    p_113[0][0][0] = p_13[0]
    p_113[0][0][1] = p_13[1]
    p_113[0][0][2] = p_13[2]
    return p_113


def convert_points_nx3_to_1xnx3(pn3):
    n = len(pn3)
    p_1n3 = np.zeros((1,n,3))

    for idx, point in enumerate(pn3):
        p_1n3[:,idx ,:] = convert_points_1x3_to_1x4x3(point)
    return p_1n3


