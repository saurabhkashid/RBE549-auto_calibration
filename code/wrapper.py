import cv2
import numpy as np
import glob
import argparse
import matplotlib.pyplot as plt 
from scipy.optimize import least_squares 


# Helper to build v_ij row
def v_ij(h, i, j):
    return np.array([
        h[0,i]*h[0,j],
        h[0,i]*h[1,j] + h[1,i]*h[0,j],
        h[1,i]*h[1,j],
        h[2,i]*h[0,j] + h[0,i]*h[2,j],
        h[2,i]*h[1,j] + h[1,i]*h[2,j],
        h[2,i]*h[2,j]
    ])
def get_intrensic_param(b):
    # 5) Recover intrinsic parameters from B
    v0 = (b[1]*b[3] - b[0]*b[4]) / (b[0]*b[2] - b[1]**2)
    lambda_ = b[5] - (b[3]**2 + v0*(b[1]*b[3] - b[0]*b[4])) / b[0]
    alpha = np.sqrt(lambda_ / b[0])
    beta  = np.sqrt(lambda_ * b[0] / (b[0]*b[2] - b[1]**2))
    gamma = -b[1] * alpha**2 * beta / lambda_
    u0    = gamma * v0 / alpha - b[3] * alpha**2 / lambda_
    A = np.array([[alpha, gamma, u0],
                [0,     beta,  v0],
                [0,     0,     1]])
    return A

def get_extrensic_param(A, Hs):
    Rotation = []
    translation = []
    K_inv = np.linalg.inv(A)
    for H in Hs:
        h1,h2,h3 = H[:,0], H[:,1], H[:,2]
        lam = 1.0/np.linalg.norm(K_inv.dot(h1))
        r1 = lam * K_inv.dot(h1)
        r2 = lam * K_inv.dot(h2)
        r3 = np.cross(r1, r2)
        # Orthogonalize
        U,_,Vt = np.linalg.svd(np.column_stack((r1,r2,r3)))
        R = U.dot(Vt)
        t = lam * K_inv.dot(h3)
        Rotation.append(R)
        translation.append(t.reshape(3,1))
    return Rotation, translation

def objective(params, world_pts, obs):
    res = residuals(params, world_pts, obs)   # from above
    return np.sum(res**2)

def residuals(params, world_pts, obs):
    fx, fy, cx, cy, k1, k2 = params[:6]
    K = np.array([[fx,0,cx],
                  [0,fy,cy],
                  [0,  0, 1]], dtype=float)
    out = []
    idx = 6
    for img_obs in obs:
        rvec = params[idx:idx+3];   idx+=3
        tvec = params[idx:idx+3];   idx+=3
        proj, _ = cv2.projectPoints(
            world_pts, rvec, tvec, K,
            np.array([k1, k2, 0, 0, 0], dtype=float)
        )
        out.append((proj.reshape(-1,2) - img_obs).ravel())
    return np.hstack(out)

def from_param_to_vect(A_init, R_init, t_init):
    vec0 = []
    fx, fy = A_init[0,0], A_init[1,1]
    cx, cy = A_init[0,2], A_init[1,2]
    vec0 += [fx, fy, cx, cy, 0.0, 0.0]   # start k1=k2=0
    for R, t in zip(R_init, t_init):
        rvec,_ = cv2.Rodrigues(R)
        vec0 += rvec.ravel().tolist()
        vec0 += t.ravel().tolist()
    vec0 = np.array(vec0)
    return vec0

def from_vect_to_param(vec, num_views):
    # Extract intrinsics and distortion coefficients
    fx, fy, cx, cy, k1, k2 = vec[:6]
    A = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0,  0,  1]])
    dist = np.array([k1, k2, 0.0, 0.0, 0.0])  # Assuming 5 distortion coeffs

    R_list = []
    t_list = []
    idx = 6
    for _ in range(num_views):
        rvec = np.array(vec[idx:idx+3])
        tvec = np.array(vec[idx+3:idx+6])
        R, _ = cv2.Rodrigues(rvec)
        R_list.append(R)
        t_list.append(tvec.reshape(3, 1))
        idx += 6

    return A, dist, R_list, t_list

def evaluate_params(A_optimized, dist_optimized, R_list_optimized, t_list_optimized, world_pts, obs):
    mean_reprojection_errors = []
    total_error = 0
    num_points = 0

    print("\n--- Reprojection Error Analysis ---")
    for i, (img_obs, R, t) in enumerate(zip(obs, R_list_optimized, t_list_optimized)):
        # Project world points using optimized params
        proj_points, _ = cv2.projectPoints(
            world_pts, R, t, A_optimized, 
            np.array([dist_optimized[0], dist_optimized[1], 0, 0, 0], dtype=float) # Ensure 5 elements for projectPoints, even if p1,p2,k3 are zero
        )
        proj_points = proj_points.reshape(-1, 2)

        # Calculate Euclidean distance for each point
        error_per_point = np.linalg.norm(proj_points - img_obs, axis=1)
        mean_image_error = np.mean(error_per_point)
        mean_reprojection_errors.append(mean_image_error)
        
        total_error += np.sum(error_per_point)
        num_points += len(error_per_point)

        print(f"Image {i+1}: Mean Reprojection Error = {mean_image_error:.4f} pixels")

    overall_mean_reprojection_error = total_error / num_points
    print(f"\nOverall Mean Reprojection Error: {overall_mean_reprojection_error:.4f} pixels")
    print(f"Optimized K (Intrinsic Matrix):\n{A_optimized}")
    print(f"Optimized Distortion Coefficients (k1, k2, p1, p2, k3):\n{dist_optimized}")

def undistort_images(img_file_names, A, dist, output_file):
    for img_file in img_file_names:
        img = cv2.imread(img_file)
        h,w = img.shape[:2]
        new_A, _ = cv2.getOptimalNewCameraMatrix(A, dist, (w, h), 1, (w, h))
        undistorted = cv2.undistort(img, A, dist, None, new_A)
        img_name = img_file.split("/")[-1]
        # save the undistorted img
        cv2.imwrite(f"{output_file}/{img_name}", undistorted)

def main():
    # --- User settings ---
    # Path to your calibration images (adjust pattern as needed)
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--calibrationImagePath', default="/mnt/d/courses/computer_vision/Home_Work/skashid@wpi_hw3/P3Data/Calibration", help='provide calibration image folder path')
    Parser.add_argument('--outputPath',default="/mnt/d/courses/computer_vision/Home_Work/skashid@wpi_hw3/P3Data/output", help= "Path to save images")
    Arg = Parser.parse_args()
    image_path = Arg.calibrationImagePath

    image_files = glob.glob(f'{image_path}/*.png')
    # image_files = glob.glob(f'{image_path}/*.jpg')
    # Number of inner corners per chessboard row and column
    cb_rows, cb_cols = 6, 9 # TODO remove 2 outer col and row

    # Termination criteria for corner refinement
    criteria = (cv2.TERM_CRITERIA_EPS +
                cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare object points (0,0,0), (1,0,0), ... in the pattern frame
    objp = np.zeros((cb_rows * cb_cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cb_cols, 0:cb_rows].T.reshape(-1, 2)

    # Storage for corresponding points
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane

    # 1) Detect corners in all images
    for fname in image_files:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (cb_cols, cb_rows), None)
        if ret:
            # Refine corner locations
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)

            # draw them in-place on img
            cv2.drawChessboardCorners(img, (cb_cols, cb_rows), corners, ret)
            # cv2.imshow('Corners', img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            imgpoints.append(corners2.reshape(-1, 2))
            objpoints.append(objp)

    # 2) Compute homographies for each view
    Hs = []
    for objp_i, imgp_i in zip(objpoints, imgpoints):
        H, _ = cv2.findHomography(objp_i[:, :2], imgp_i, 0)
        Hs.append(H)



    # 3) Build the V matrix from homographies
    V = []
    for H in Hs:
        V.append(v_ij(H, 0, 1))               # h1^T B h2 = 0
        V.append(v_ij(H, 0, 0) - v_ij(H, 1, 1))  # h1^T B h1 = h2^T B h2
    V = np.vstack(V)

    # 4) Solve V b = 0 via SVD
    _, _, Vt = np.linalg.svd(V)
    b = Vt[-1, :]

    A_init = get_intrensic_param(b)

    # 6) Extract extrinsic parameters for each view
    R_init, t_init = get_extrensic_param(A_init,Hs)

    # 7) Display results
    print("Intrinsic matrix A:")
    print(A_init)
    # for idx, (R, t) in enumerate(extrinsics):
    #     print(f"View {idx+1} rotation R:", R)
    #     print(f"View {idx+1} translation t:", t)

    # 4) Pack all parameters into a single vector
    #    [fx, fy, cx, cy, k1, k2, rvec1(3), t1(3), ..., rvecN(3), tN(3)]
    vec0 = from_param_to_vect(A_init, R_init, t_init)

    # 5) Define reprojection residuals
    world_pts = objp  # M×3
    obs = imgpoints   # list of N arrays, each M×2

    # 6) Run nonlinear optimization
    ret = least_squares(residuals, vec0, args=(world_pts, obs))
    
    # 7) Unpack optimized parameters
    A, dist, R_list, t_list = from_vect_to_param(ret.x, len(obs))

    # 8) evaluate the parameters 
    evaluate_params(A, dist, R_list, t_list, world_pts, obs)
    undistort_images(image_files, A, dist, Arg.outputPath)

    print("Optimized K:\n", dist)


if __name__ == "__main__":
    main()