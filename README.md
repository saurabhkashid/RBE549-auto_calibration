# RBE549-auto_calibration

## Table of Contents

* [Introduction](#introduction)
* [Features](#features)
* [Theory & Methodology](#theory--methodology)
* [Getting Started](#getting-started)
    * [Prerequisites](#prerequisites)
    * [Installation](#installation)
    * [Usage](#usage)
* [Results](#results)


---

## Introduction

Implementation of Camera calibration method by Zhengyou Zhang from Microsoft in his paper, [A flexible new technique for camera calibration.](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr98-71.pdf)

## Features

* **Zhang's Calibration Method:** Implements the core steps of Zhang's classical camera calibration algorithm.
* **Homography Estimation:** Calculates homographies between 3D world points and 2D image points for multiple views.
* **Initial Intrinsic Parameter Estimation:** Derives initial camera intrinsic matrix using linear least squares from homographies.
* **Extrinsic Parameter Estimation:** Computes rotation and translation vectors for each calibration image.
* **Non-linear Optimization (Bundle Adjustment):** Refines intrinsic (focal lengths, principal point, radial distortion $k_1, k_2$) and extrinsic parameters using `scipy.optimize.least_squares` to minimize reprojection error.
* **Distortion Correction:** Applies estimated distortion coefficients to undistort images.
* **Command-Line Interface:** Easy-to-use argument parsing for input/output paths.
* **Reprojection Error Analysis:** Provides quantitative assessment of calibration accuracy.
* **Support for Radial Distortion:** Specifically targets optimization for $k_1$ and $k_2$ radial distortion coefficients.

## Theory & Methodology

The proposed camera calibration technique offers a flexible new approach, suitable for users without specialized knowledge in 3D geometry or computer vision. It requires observing a planar pattern at a minimum of two different orientations, allowing either the camera or the pattern to be moved freely without needing to know the motion. The method accounts for radial lens distortion and combines a closed-form solution with a nonlinear refinement based on the maximum likelihood criterion.

The approach lies between traditional photogrammetric calibration and self-calibration methods by utilizing 2D metric information. This provides a balance, offering more flexibility than classical techniques and greater robustness compared to self-calibration.

### 1. Basic Equations

The methodology begins by examining the constraints on the camera's intrinsic parameters through the observation of a single plane.

* **Notation**: A 2D point is denoted by $m=[u,v]^{T}$, and a 3D point by $M=[X,Y,Z]^{T}$. Augmented vectors are represented as $\tilde{m}=[u,v,1]^{T}$ and $\tilde{M}=[X,Y,Z,1]^{T}$. The camera is modeled using the pinhole model, where the relationship between a 3D point $M$ and its image projection $m$ is given by:
    $s\tilde{m}=A[\begin{matrix}R&t\end{matrix}]\tilde{M}$ (1)
    Here, $s$ is an arbitrary scale factor, $(R, t)$ are the extrinsic parameters (rotation and translation relating the world coordinate system to the camera coordinate system), and $A$ is the camera intrinsic matrix:
    $A=[\begin{matrix}\alpha&\gamma&u_{0}\\ 0&\beta&v_{0}\\ 0&0&1\end{matrix}]$
    where $(u_{0},v_{0})$ are the principal point coordinates, $\alpha$ and $\beta$ are scale factors for the image $u$ and $v$ axes, respectively, and $\gamma$ describes the skewness of the two image axes.

* **Homography between the model plane and its image**: Assuming the model plane lies on $Z=0$ in the world coordinate system, a model point $M=[X,Y]^{T}$ and its image $m$ are related by a homography $H$:
    $s\tilde{m}=H\tilde{M}$ with $H=A[\begin{matrix}r_{1}&r_{2}&t\end{matrix}]$ (2)
    where $r_{1}$ and $r_{2}$ are the first two columns of the rotation matrix $R$, and $H$ is a $3\times3$ matrix defined up to a scale factor.

* **Constraints on the intrinsic parameters**: Given an image of the model plane, a homography $H=[\begin{matrix}h_{1}&h_{2}&h_{3}\end{matrix}]$ can be estimated. Due to the orthonormal nature of $r_{1}$ and $r_{2}$, two fundamental constraints on the intrinsic parameters arise:
    $h_{1}^{T}A^{-T}A^{-1}h_{2}=0$ (3)
    $h_{1}^{T}A^{-T}A^{-1}h_{1}=h_{2}^{T}A^{-T}A^{-1}h_{2}$ (4)
    These two constraints are derived from a single homography. The matrix $A^{-T}A^{-1}$ represents the image of the absolute conic.

* **Geometric Interpretation**: The constraints (3) and (4) relate to the absolute conic. The projection of the circular points on the model plane (points at infinity satisfying $x_{\infty}^{T}x_{\infty}=0$) onto the image plane results in $\tilde{m}_{\infty}=A(r_{1}\pm ir_{2})=h_{1}\pm ih_{2}$. Since $\tilde{m}_{\infty}$ lies on the image of the absolute conic (described by $A^{-T}A^{-1}$), requiring its real and imaginary parts to be zero yields equations (3) and (4).

### 2. Solving Camera Calibration

The calibration problem is solved through an analytical solution, followed by nonlinear optimization, and finally by addressing lens distortion.

* **Closed-form Solution**:
    Let $B=A^{-T}A^{-1}$, which is a symmetric matrix defined by a 6D vector $b=[B_{11},B_{12},B_{22},B_{13},B_{23},B_{33}]^{T}$. The two constraints (3) and (4) from a given homography can be rewritten as two homogeneous equations in $b$:
    $[\begin{matrix}v_{12}^{T}\\ (v_{11}-v_{22})^{T}\end{matrix}]b=0$ (8)
    If $n$ images of the model plane are observed, stacking $n$ such equations results in $Vb=0$, where $V$ is a $2n\times6$ matrix. If $n\ge3$, a unique solution for $b$ (up to a scale factor) can generally be found. For $n=2$, a skewless constraint ($\gamma=0$) can be added. The solution for $b$ is the eigenvector of $V^{T}V$ corresponding to the smallest eigenvalue. Once $b$ is estimated, the camera intrinsic matrix $A$ can be computed (details in Appendix B). Subsequently, extrinsic parameters for each image ($r_1, r_2, r_3, t$) are calculated from $A$ and the homography $H$. A method to estimate the best rotation matrix from the computed $R$ is provided in Appendix C to account for noise.

* **Maximum Likelihood Estimation**:
    The closed-form solution minimizes an algebraic distance, which is refined through maximum likelihood inference. Assuming image points are corrupted by independent and identically distributed noise, the maximum likelihood estimate minimizes the functional:
    $\sum_{i=1}^{n}\sum_{j=1}^{m}||m_{ij}-\hat{m}(A,R_{i},t_{i},M_{j})||^{2}$ (10)
    where $\hat{m}(A,R_{i},t_{i},M_{j})$ is the projection of point $M_{j}$ in image $i$. This nonlinear minimization problem is solved using the Levenberg-Marquardt Algorithm, requiring an initial guess for $A$ and $\{R_{i},t_{i}\}$ obtained from the closed-form solution.

* **Dealing with Radial Distortion**:
    Desktop cameras often exhibit significant radial distortion. The model incorporates the first two terms of radial distortion coefficients, $k_1$ and $k_2$. The distorted image coordinates $(\tilde{u},\tilde{v})$ are related to the ideal (distortion-free) coordinates $(u,v)$ by:
    $\tilde{u}=u+(u-u_{0})[k_{1}(x^{2}+y^{2})+k_{2}(x^{2}+y^{2})^{2}]$ (11)
    $\tilde{v}=v+(v-v_{0})[k_{1}(x^{2}+y^{2})+k_{2}(x^{2}+y^{2})^{2}]$ (12)
    where $(x,y)$ are normalized image coordinates.
    * **Estimating Radial Distortion by Alternation**: An initial estimation of $k_1$ and $k_2$ can be done by solving a linear least-squares problem (Equation 13) after the other intrinsic parameters are estimated. The process can then alternate between estimating $k_1, k_2$ and refining other parameters by re-solving Equation (10) with the distortion model applied, until convergence.
    * **Complete Maximum Likelihood Estimation**: For faster convergence, a complete set of parameters (including $k_1$ and $k_2$) can be estimated by minimizing an extended functional (Equation 14) that incorporates the distortion model directly:
        $\sum_{i=1}^{n}\sum_{j=1}^{m}||m_{ij}-\tilde{m}(A,k_{1},k_{2},R_{i},t_{i},M_{j})||^{2}$ (14)
        This is a nonlinear minimization solved using the Levenberg-Marquardt Algorithm, using initial guesses from the closed-form solution or by setting $k_1, k_2$ to zero.

* **Summary of Calibration Procedure**: The recommended steps are:
    1.  Print a planar pattern and attach it to a surface.
    2.  Capture multiple images of the pattern from different orientations.
    3.  Detect feature points within the images.
    4.  Estimate the five intrinsic parameters and all extrinsic parameters using the closed-form solution (Section 3.1).
    5.  Estimate the radial distortion coefficients ($k_1, k_2$) by solving the linear least-squares equation (13).
    6.  Refine all parameters by minimizing the full maximum likelihood functional (14).

## Getting Started

### Prerequisites

* Python 3.x
* NumPy
* OpenCV (`opencv-python`)
* SciPy
* Matplotlib (optional, for visualization if implemented)

You can install these using pip:

```bash
pip install numpy opencv-python scipy matplotlib
```

### usage
provide the aurgument calibrationImagePath and outputPath 
```python
python Wrapper.py --calibrationImagePath ../P3Data/Calibration --outputPath ../P3Data/output
```
## Result 
```
--- Reprojection Error Analysis --- \
Image 1: Mean Reprojection Error = 2.8572 pixels \
Image 2: Mean Reprojection Error = 0.1853 pixels \
Image 3: Mean Reprojection Error = 0.2274 pixels \
Image 4: Mean Reprojection Error = 0.3382 pixels \
Image 5: Mean Reprojection Error = 2.1208 pixels \
Image 6: Mean Reprojection Error = 2.0340 pixels \
Image 7: Mean Reprojection Error = 0.3538 pixels \
Image 8: Mean Reprojection Error = 0.2590 pixels \
Image 9: Mean Reprojection Error = 3.8496 pixels \
Image 10: Mean Reprojection Error = 0.2022 pixels \
Image 11: Mean Reprojection Error = 0.1910 pixels \
Image 12: Mean Reprojection Error = 3.8739 pixels \
Image 13: Mean Reprojection Error = 0.2952 pixels \
Image 14: Mean Reprojection Error = 0.8074 pixels 
```