
import copy
import numpy as np
import cv2
import random
from scipy.spatial.transform import Rotation as spr
from matplotlib import pyplot as plt
import sksurgerycore.algorithms.procrustes as pro
import sksurgerycore.transforms.matrix as mu
from IPython.display import Image
import csv


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
                                  create_pat_ref_in_camera_space, get_ref_T_tip, calculate_euclid_dist

'''

def simulation_AR(end_ref_cam, end_ref_marker, pat_ref_cam, pat_ref_marker, PatRef_T_MRI, EndRef_T_EndP, target_location_in_mri_space, intrinsics, distortion, target_location_in_image_coordinates, Cam_T_PatRef, Cam_T_EndRef):
    x_values = []
    y_values = []
    for sigma_counter in range(0, 50, 1):
        sigma = float(sigma_counter) / float(100)
        rms = 0
        for i in range(number_samples):
            
            noisy_end_ref_cam = add_noise_to_points(end_ref_cam[:,0:3], sigma)  
            # recompute transforms with the noisy reference points (from references to camera)
            R, t, FRE = pro.orthogonal_procrustes(noisy_end_ref_cam, end_ref_marker[:,0:3])
            noisy_Cam_T_EndRef = mu.construct_rigid_transformation(R, t)

            # same to patient ref
            noisy_pat_ref_cam = add_noise_to_points(pat_ref_cam[:,0:3], sigma)
            R, t, FRE = pro.orthogonal_procrustes(noisy_pat_ref_cam, pat_ref_marker[:, 0:3])
            noisy_Cam_T_PatRef = mu.construct_rigid_transformation(R, t)

            # Here we add noise onto the PatRef_T_MRI_parameters, and reconstruct a new registration
            PatRef_T_MRI_parameters =  extract_rigid_body_parameters(PatRef_T_MRI)
            tmp_params = add_noise_to_params(PatRef_T_MRI_parameters, sigma)
            noisy_PatRef_T_MRI = rigid_body_parameters_to_matrix(tmp_params)
            
            # adding noise to hand-eye
            # Here we add noise onto the Hand_T_Eye_parameters, and reconstruct a new registration
            EndRef_T_EndP_parameters =  extract_rigid_body_parameters(EndRef_T_EndP)
            noisy_EndRef_T_EndP = add_noise_to_params(EndRef_T_EndP_parameters, sigma)
            noisy_hand_eye = rigid_body_parameters_to_matrix(noisy_EndRef_T_EndP)

            # convert noisy target point from MRI space to camera space
            #transformed_point_in_camera_space = mri_to_camera(np.linalg.inv(noisy_PatRef_T_MRI), np.linalg.inv(noisy_Cam_T_PatRef), noisy_Cam_T_EndRef, np.linalg.inv(noisy_hand_eye), target_location_in_marker_space)
            
            # use noisy transforms to get a target location from camera space to MRI space.            
            MRI_T_EndP_noisy = np.linalg.inv(noisy_PatRef_T_MRI) @ np.linalg.inv(noisy_Cam_T_PatRef) @ noisy_Cam_T_EndRef @  noisy_hand_eye
            MRI_T_EndP = np.linalg.inv(PatRef_T_MRI) @ np.linalg.inv(Cam_T_PatRef) @ Cam_T_EndRef @  EndRef_T_EndP

            transformed_point_in_marker_space_noisy = np.linalg.inv(MRI_T_EndP_noisy)@target_location_in_mri_space
            transformed_point_in_marker_space = np.linalg.inv(MRI_T_EndP)@target_location_in_mri_space

            Cam_T_EndRef_parameters = extract_rigid_body_parameters(np.linalg.inv(Cam_T_EndRef @  EndRef_T_EndP))
            # project noisy target point from camera space to image space
            rvec = np.zeros((1,3))
            tvec = np.zeros((1,3))
            rvec[0,0]=Cam_T_EndRef_parameters[0]
            rvec[0,1]=Cam_T_EndRef_parameters[1]
            rvec[0,2]=Cam_T_EndRef_parameters[2]

            tvec[0,0]=Cam_T_EndRef_parameters[3]
            tvec[0,1]=Cam_T_EndRef_parameters[4]
            tvec[0,2]=Cam_T_EndRef_parameters[5]

            #rvec = np.array([Cam_T_EndRef_parameters[:3]], dtype=float)
            #tvec = np.array([Cam_T_EndRef_parameters[4:]], dtype=float)
            transformed_point_in_image_space_noisy, _ = cv2.projectPoints(convert_4x1_to_1x1x3(transformed_point_in_marker_space_noisy), rvec, tvec, intrinsics, distortion)
            transformed_point_in_image_space_noisy = transformed_point_in_image_space_noisy.squeeze()
            #transformed_point_in_image_space = project_camera_point_to_image(transformed_point_in_camera_space, intrinsics, distortion, rvec=rvec, tvec=tvec)
            target_location_in_image_coordinates, _ = cv2.projectPoints(convert_4x1_to_1x1x3(transformed_point_in_marker_space), rvec, tvec, intrinsics, distortion)
            target_location_in_image_coordinates = target_location_in_image_coordinates.squeeze()
            #rvec = np.zeros((1,3))
            #tvec = np.zeros((1,3))
            #transformed_point_in_image_space, _ = cv2.projectPoints(convert_4x1_to_1x1x3(transformed_point_in_camera_space), rvec, tvec, intrinsics, distortion)
            #print(f'noisy_point_in_image_space: {transformed_point_in_image_space}')

            # compute euclidean distance between true target point and noisy in image space
            euclid_dist =  (transformed_point_in_image_space_noisy[0] - target_location_in_image_coordinates[0]) \
                        * (transformed_point_in_image_space_noisy[0] - target_location_in_image_coordinates[0]) \
                        + (transformed_point_in_image_space_noisy[1] - target_location_in_image_coordinates[1]) \
                        * (transformed_point_in_image_space_noisy[1] - target_location_in_image_coordinates[1]) 
            
            rms = rms + euclid_dist
        
        rms = rms / float(number_samples)
        rms = np.sqrt(rms)
        
        if sigma == 0.25:
            AR_pxls_25 = rms
            print(f'number of pxls off at sigma {sigma} is {AR_pxls_25}')
        if sigma == 0.12:
            AR_pxls_12 = rms
            print(f'number of pxls off at sigma {sigma} is {AR_pxls_12}')
        if sigma == 0.15:
            AR_pxls_15 = rms
            print(f'number of pxls off at sigma {sigma} is {AR_pxls_15}')
            
        x_values.append(sigma)
        y_values.append(rms)
        
    plt.plot(x_values, y_values, 'r', label='TRE (mm)')
    plt.legend(loc='upper left')
    plt.xlabel('sigma (rotations(degrees)/translations(mm))')
    plt.ylabel('RMS (Pixels)')
    plt.show()

    return AR_pxls_12, AR_pxls_15, AR_pxls_25


'''

def plot_ref(ref_coords, i, j):

    plt.plot(ref_coords[:,i],ref_coords[:,j], marker='*',linestyle = 'None',)
    plt.xlabel(i)
    plt.ylabel(j)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

    return


def create_pnt_ref():

    # Creating pointer reference (from datasheet). Using homogenous (4 numbers, x,y,z,1) as row vectors.
    pnt_ref =  np.zeros((4, 4))

    # marker A (0) -> 0,0,0

    # marker B (1) -> 0,50,0
    pnt_ref[1][1] = 50 # y

    # marker c (2) -> 25,100,0
    pnt_ref[2][0] = 25  # x
    pnt_ref[2][1] = 100 # y

    # marker d (3) -> -25, 135, 0
    pnt_ref[3][0] = -25 # x
    pnt_ref[3][1] = 135 # y

    # adding 1 to 3rd dimension to turn to homogeneous coordinates
    pnt_ref[0][3] = 1
    pnt_ref[1][3] = 1
    pnt_ref[2][3] = 1
    pnt_ref[3][3] = 1

    return pnt_ref



def create_pat_ref():
    # Defining reference coordibates in ref coords (from datasheet)
    
    # Encoding the reference marker points into a numpy matrix
    pat_ref = np.zeros((4, 4))
    # marker A (0) -> (0,0,0)

    # marker B (1) -> (41.02 ,0,28.59)
    pat_ref[1][0] = 41.02 # x
    pat_ref[1][2] = 28.59 # z

    # marker C (2) -> C = (88.00 ,0, 0)
    pat_ref[2][0] = 88 # x

    # marker D (3) -> (40.45,0,-44.32)
    pat_ref[3][0] = 40.45 # x
    pat_ref[3][2] = -44.32 # z
    
    # adding 1 to last row to make coordinates homogenous
    pat_ref[0][3] = 1.0
    pat_ref[1][3] = 1.0
    pat_ref[2][3] = 1.0
    pat_ref[3][3] = 1.0
    return pat_ref


def recompute_noisy_transform(pnts, sigma):
    noisy_pnts = add_noise_to_points(pnts[:,0:3], sigma) 
    R, t, FRE = pro.orthogonal_procrustes(noisy_pnts, pnts)
    noisy_Cam_T_EndRef = mu.construct_rigid_transformation(R, t)



def project_points(points, T, intrinsics, distortion): # target_location_in_marker_space, target_location_in_mri_space,
    params = extract_rigid_body_parameters(T)
    # project noisy target point from camera space to image space
    rvec = np.zeros((1,3))
    tvec = np.zeros((1,3))
    rvec[0,0]=params[0]
    rvec[0,1]=params[1]
    rvec[0,2]=params[2]

    tvec[0,0]=params[3]
    tvec[0,1]=params[4]
    tvec[0,2]=params[5]

    #rvec = np.array([Cam_T_EndRef_parameters[:3]], dtype=float)
    #tvec = np.array([Cam_T_EndRef_parameters[4:]], dtype=float)
    transformed_point, _ = cv2.projectPoints(convert_4x1_to_1x1x3(points), rvec, tvec, intrinsics, distortion)
    transformed_point = transformed_point.squeeze()
    return transformed_point


def simulation(Cam_T_PatRef=0, Cam_T_EndRef=0, EndRef_T_EndP=0, PatRef_T_MRI=0,end_ref_marker=0, end_ref_cam=0,pat_ref_marker=0, pat_ref_cam=0, add_reg_error=False, add_hand_eye_error=False, add_tracking_noise=False, AR=False, intrinsics=[], distortion=[]):
    x_values = []
    y_values = []

    # point at tip of pointer in cam space
    target_location_in_marker_space = np.zeros((4,1))
    target_location_in_marker_space[2,0] = 0
    target_location_in_marker_space[3,0] = 1 # homogeous

    MRI_T_EndP = np.linalg.inv(PatRef_T_MRI) @ np.linalg.inv(Cam_T_PatRef) @ Cam_T_EndRef @  EndRef_T_EndP

    target_location_in_mri_space = MRI_T_EndP @ target_location_in_marker_space
    if AR:
        T = Cam_T_EndRef @  EndRef_T_EndP
        target_location_in_image_coordinates = project_points(target_location_in_mri_space, T, intrinsics, distortion)
                
    PatRef_T_MRI_original = copy.deepcopy(PatRef_T_MRI)
    EndRef_T_EndP_original = copy.deepcopy(EndRef_T_EndP)

    for sigma_counter in range(0, 50, 1):
        sigma = float(sigma_counter) / float(100)

        rms = 0
        for i in range(number_samples):
            
            if add_tracking_noise:
                #print('adding tracking error')
                # Here we add tracking noise to the ref points in cam space
                noisy_end_ref_cam = add_noise_to_points(end_ref_cam[:,0:3], sigma)  
                # recompute transforms with the noisy reference points (from references to camera)
                R, t, FRE = pro.orthogonal_procrustes(noisy_end_ref_cam, end_ref_marker[:,0:3])
                Cam_T_EndRef = mu.construct_rigid_transformation(R, t)

                # same to patient ref
                noisy_pat_ref_cam = add_noise_to_points(pat_ref_cam[:,0:3], sigma)
                R, t, FRE = pro.orthogonal_procrustes(noisy_pat_ref_cam, pat_ref_marker[:, 0:3])
                Cam_T_PatRef = mu.construct_rigid_transformation(R, t)

            if add_reg_error:
                #print('adding registration error')
                # Here we add noise onto the PatRef_T_MRI_parameters, and reconstruct a new registration
                PatRef_T_MRI_parameters = extract_rigid_body_parameters(PatRef_T_MRI_original)
                noisy_params = add_noise_to_params(PatRef_T_MRI_parameters, sigma)
                PatRef_T_MRI = rigid_body_parameters_to_matrix(noisy_params)
            
            if add_hand_eye_error:
                Hand_T_Eye_parameters = extract_rigid_body_parameters(EndRef_T_EndP_original)
                noisy_hand_eye_params = add_noise_to_params(Hand_T_Eye_parameters, sigma)
                EndRef_T_EndP = rigid_body_parameters_to_matrix(noisy_hand_eye_params)

            # use noisy transforms to get a target location from camera space to MRI space.            
            MRI_T_EndP_noisy = np.linalg.inv(PatRef_T_MRI) @ np.linalg.inv(Cam_T_PatRef) @ Cam_T_EndRef @ EndRef_T_EndP
            transformed_point_noisy = MRI_T_EndP_noisy@target_location_in_marker_space
            
            if AR:
                T = Cam_T_EndRef @  EndRef_T_EndP#  Cam_T_EndRef @  EndRef_T_EndP
                transformed_point_in_image_space_noisy = project_points(transformed_point_noisy, T, intrinsics, distortion)
                
                euclid_dist =  (transformed_point_in_image_space_noisy[0] - target_location_in_image_coordinates[0]) \
                        * (transformed_point_in_image_space_noisy[0] - target_location_in_image_coordinates[0]) \
                        + (transformed_point_in_image_space_noisy[1] - target_location_in_image_coordinates[1]) \
                        * (transformed_point_in_image_space_noisy[1] - target_location_in_image_coordinates[1]) 
            else:
                # calculate euclid dist
                euclid_dist = calculate_euclid_dist(transformed_point_noisy, target_location_in_mri_space)
            rms = rms + euclid_dist


        rms = rms / float(number_samples)
        rms = np.sqrt(rms)

        if sigma == 0.25:
            error_25 = rms
            print(f' error at {sigma} is {error_25}mm')
        if sigma == 0.12:
            error_12 = rms
            print(f' error at {sigma} is {error_12}mm')
        if sigma == 0.15:
            error_15 = rms
            print(f' error at {sigma} is {error_15}mm')

        x_values.append(sigma)
        y_values.append(rms)

    return x_values, y_values, error_12, error_15, error_25




def main():
    # markers on endoscope in endref coordinates
    end_ref =  create_pnt_ref() # marker coords
    
    # EndP to EndRef (pointer length translation in y)
    EndRef_T_EndP = create_transform([0, length_of_endoscope, 0, 0, 0, 0]) # create transform of all points depending on pointer's length
    end_ref_marker = multiply_points_by_matrix(EndRef_T_EndP, end_ref, do_transpose=True) # transform all pointer points by this reference
    #hand_eye = get_ref_T_tip(length_of_endoscope, 'z')

    # EndRef to Cam (add distance to cam in z, rotate by x degrees)
    #end_ref_c =  create_pnt_ref_in_camera_space()     # endoscope reference in cam coords
    # once again creating offset to all the points to move origin to the endoscope's tip
    #offset_marker_transform_cam = create_transform([length_of_endoscope, 0, 0, 0, 0, 0])
    #end_ref_c = multiply_points_by_matrix(offset_marker_transform_cam, end_ref_c, do_transpose=True)
    
    rotate_about_z = create_transform([0, 0, 0, 0, 0, pointer_angle]) 
    translate_away_from_camera = create_transform([0, 0, distance_from_cam, 0, 0, 0])
    Cam_T_EndRef = translate_away_from_camera @ rotate_about_z
    end_ref_cam = multiply_points_by_matrix(Cam_T_EndRef, end_ref_marker, do_transpose=True)
    # getting transformation between coordinate systems
    #Cam_T_EndRef_r, Cam_T_EndRef_t, Cam_T_Endo_FRE = pro.orthogonal_procrustes(end_ref_cam[:,:3], end_ref_marker[:,:3])
    # reconstructing translation and rotation to one matrix
    #Cam_T_EndRef = mu.construct_rigid_transformation(Cam_T_EndRef_r, Cam_T_EndRef_t)

    # PatRef to Cam (add dist to cam to z plus x translation to right)
    pat_ref_marker = create_pat_ref()
    #pat_ref_c = create_pat_ref_in_camera_space()
    # translating to correct location 
    translate_along_x = create_transform([tumour_patref_y, 0, 0, 0, 0, 0])
    Cam_T_PatRef = translate_along_x @ translate_away_from_camera
    pat_ref_cam = multiply_points_by_matrix(Cam_T_PatRef, pat_ref_marker, do_transpose=True)
    # getting transform
    #Cam_T_PatRef_r, Cam_T_PatRef_t, Cam_T_PatRef_FRE = pro.orthogonal_procrustes(pat_ref_cam[:,0:3], pat_ref_marker[:,0:3])
    #Cam_T_PatRef = mu.construct_rigid_transformation(Cam_T_PatRef_r, Cam_T_PatRef_t)

    # PatRef to MRI
    PatRef_T_MRI = create_transform([x_t, y_t, z_t, 0, 0, 0])
    PatRef_T_MRI_parameters = extract_rigid_body_parameters(PatRef_T_MRI)
    print('here')
    print(np.rint(EndRef_T_EndP))
    print(np.rint(Cam_T_EndRef))
    print(np.rint(Cam_T_PatRef))
    print(np.rint(PatRef_T_MRI))

    print('EndP -> EndRef')
    print(np.rint(extract_rigid_body_parameters(EndRef_T_EndP)))

    print('EndRef -> Cam')
    print(np.rint(extract_rigid_body_parameters(Cam_T_EndRef)))

    print('PatRef -> Cam')
    print(np.rint(extract_rigid_body_parameters(Cam_T_PatRef)))

    print('MRI -> PatRef')
    print(np.rint(extract_rigid_body_parameters(PatRef_T_MRI)))

    '''
    # point at tip of pointer in cam space
    target_location_in_marker_space = np.zeros((4,1))
    target_location_in_marker_space[2,0] = 0
    target_location_in_marker_space[3,0] = 1 # homogeous

    #target_location_in_mri_space = camera_to_mri(np.linalg.inv(PatRef_T_MRI), np.linalg.inv(Cam_T_PatRef), Cam_T_EndRef, np.linalg.inv(EndRef_T_EndP), )
    
    MRI_T_EndP = np.linalg.inv(PatRef_T_MRI) @ np.linalg.inv(Cam_T_PatRef) @ Cam_T_EndRef @  EndRef_T_EndP
    
    target_location_in_mri_space = MRI_T_EndP @ target_location_in_marker_space
    #back_in_cam = np.linalg.inv(MRI_T_Cam) @ target_location_in_mri_space 
    '''
    
    # TRACKER ERROR SIMULATION
    tracking_x_values, tracking_y_values, tracking_error_12, tracking_error_15, tracking_error_25 \
        = simulation(Cam_T_PatRef=Cam_T_PatRef, \
            Cam_T_EndRef=Cam_T_EndRef, EndRef_T_EndP=EndRef_T_EndP, PatRef_T_MRI=PatRef_T_MRI, \
                end_ref_marker=end_ref_marker, end_ref_cam=end_ref_cam,\
                pat_ref_marker=pat_ref_marker, pat_ref_cam=pat_ref_cam,add_tracking_noise=True)

    plt.plot(tracking_x_values, tracking_y_values, 'r', label='TRE (mm)')
    plt.legend(loc='upper left')
    plt.xlabel('sigma ')
    plt.ylabel('TRE (mm)')
    plt.show()
    


    # REG
    tracking_reg_x_values, tracking_reg_y_values, tracking_reg_error_12, \
        tracking_reg_error_15, tracking_reg_error_25 = \
            simulation(\
                Cam_T_PatRef=Cam_T_PatRef, Cam_T_EndRef=Cam_T_EndRef, EndRef_T_EndP=EndRef_T_EndP, \
                PatRef_T_MRI=PatRef_T_MRI, end_ref_marker=end_ref_marker, \
                end_ref_cam=end_ref_cam,pat_ref_marker=pat_ref_marker, pat_ref_cam=pat_ref_cam, \
                    add_tracking_noise=False,add_reg_error=True)

    plt.plot(tracking_reg_x_values, tracking_reg_y_values, 'r', label='TRE (mm)')
    plt.legend(loc='upper left')
    plt.xlabel('sigma ')
    plt.ylabel('TRE (mm)')
    plt.show()

    # HAND-EYE
    tracking_hand_eye_x_values, tracking_hand_eye_y_values, tracking_hand_eye_error_12, \
        tracking_hand_eye_error_15, tracking_hand_eye_error_25 = \
            simulation( \
                Cam_T_PatRef=Cam_T_PatRef, Cam_T_EndRef=Cam_T_EndRef, EndRef_T_EndP=EndRef_T_EndP, \
                PatRef_T_MRI=PatRef_T_MRI, end_ref_marker=end_ref_marker, \
                end_ref_cam=end_ref_cam,pat_ref_marker=pat_ref_marker, pat_ref_cam=pat_ref_cam, \
                    add_tracking_noise=False,add_reg_error=False, add_hand_eye_error=True)

    plt.plot(tracking_hand_eye_x_values, tracking_hand_eye_y_values, 'r', label='TRE (mm)')
    plt.legend(loc='upper left')
    plt.xlabel('sigma ')
    plt.ylabel('TRE (mm)')
    plt.show()


    # Camera calibration
    intrinsics = np.loadtxt('calibration/intrinsics.txt')
    distortion = np.loadtxt('calibration/distortion.txt')

    print("Intrinsics are:" + str(intrinsics))
    print("Distortion coefficients are:" + str(distortion))


    # AR
    AR_x_values, AR_y_values, AR_pxls_12, AR_pxls_15, AR_pxls_25 = \
            simulation( Cam_T_PatRef=Cam_T_PatRef, Cam_T_EndRef=Cam_T_EndRef, EndRef_T_EndP=EndRef_T_EndP, \
                PatRef_T_MRI=PatRef_T_MRI, end_ref_marker=end_ref_marker, \
                end_ref_cam=end_ref_cam,pat_ref_marker=pat_ref_marker, pat_ref_cam=pat_ref_cam, \
                    add_tracking_noise=True,add_reg_error=True, add_hand_eye_error=True, \
                    AR=True, intrinsics=intrinsics, distortion=distortion)

    # SAVE RESULTS
    header = ['error', 'sigma_12', 'sigma_15', 'sigma_25']

    data_endoscope = [
        ['tracking', tracking_error_12[0], tracking_error_15[0], tracking_error_25[0]],
        ['endo_T_R', tracking_reg_error_12[0], tracking_reg_error_15[0], tracking_reg_error_25[0]],
        ['hand_eye', tracking_hand_eye_error_12[0], tracking_hand_eye_error_15[0], tracking_hand_eye_error_25[0]],
        ['AR_px', AR_pxls_12, AR_pxls_15, AR_pxls_25],
        
    ]

    with open('notebooks/results/endoscope.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        # Use writerows() not writerow()
        writer.writerows(data_endoscope)
    
        '''
    extrinsics_parameters = extract_rigid_body_parameters(Cam_T_EndRef@EndRef_T_EndP)
    # Need to project target point to image plane. Incidentally, the target point 
    # was straight along the optical axis, so should project to the middle of the image.
    rvec = np.zeros((1,3))
    tvec = np.zeros((1,3))
    #rvec=np.array([extrinsics_parameters[:3]])
    #tvec=np.array([extrinsics_parameters[4:]])
    target_location_in_image_coordinates, _ = cv2.projectPoints(convert_4x1_to_1x1x3(target_location_in_marker_space), rvec, tvec, intrinsics, distortion)
    #target_location_in_image_coordinates = project_camera_point_to_image(target_location_in_marker_space, intrinsics, distortion, )
    print("target_location_in_image_coordinates=" + str(target_location_in_image_coordinates))

    target_location_in_image_coordinates = target_location_in_image_coordinates.squeeze()
    AR_pxls_12, AR_pxls_15, AR_pxls_25 = \
        simulation_AR(end_ref_cam, end_ref_marker, pat_ref_cam, pat_ref_marker, \
            PatRef_T_MRI, EndRef_T_EndP, target_location_in_mri_space, intrinsics, \
            distortion, target_location_in_image_coordinates, Cam_T_PatRef, Cam_T_EndRef)

    '''
    return


if __name__=='__main__':
    # ALL MEASUREMENTS IN MM

    # P - , the length of the endoscope.
    length_of_endoscope = 100 # use 300 after merging

    # D - z distance from camera to plane where everything is located
    distance_from_cam = 1000 # since the camera and patient reference are aligned in the x and y directions, only distance is in z

    # 0 - angle of pointer
    pointer_angle = 45

    # Yc - distances from tumour to patient reference
    tumour_patref_y = 50  

    # NDI quotes 0.25mm for Polaris Spectra, some papers estimate it at 0.17mm
    typical_tracking_sigma = 0.25

    # For Model 2 and 3, using an endoscope, this determines the distance of a target of interest from the endoscope.
    working_distance = 50

    # for simulation to be reproducible
    number_samples = 10

    x_t = 100 # head length (about 20cm)
    y_t = 130 # menton to top of head (about 25cm)
    z_t = 80 # head bredth (about 15cm)
    main()