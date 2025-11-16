// zhang_real_images_ceres.cpp
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <cassert>
#include <filesystem>
namespace fs = std::filesystem;

#include <opencv2/opencv.hpp>
#include <ceres/ceres.h>
#include <Eigen/Dense>
#include <ceres/rotation.h>

// ====== Basic structs ======
struct Intrinsics {
  double fx, fy, cx, cy;
};

struct Distortion {
  double k1, k2, k3, p1, p2; // Brown–Conrady
};

struct Pose {
  Eigen::Matrix3d R;
  Eigen::Vector3d t; // world -> camera
};

struct Observation {
  int view_id;
  Eigen::Vector3d Pw; // 3D point on chessboard (Z=0)
  double u, v;        // observed pixel
};

// ===================== 1. Homography via DLT (Zhang) =====================
Eigen::Matrix3d computeHomography(
    const std::vector<Eigen::Vector2d>& world_pts,  // (X, Y)
    const std::vector<Eigen::Vector2d>& img_pts)    // (u, v)
{
  assert(world_pts.size() == img_pts.size());
  const int N = (int)world_pts.size();
  Eigen::MatrixXd A(2*N, 9);
  A.setZero();

  for (int i = 0; i < N; ++i) {
    double X = world_pts[i][0];
    double Y = world_pts[i][1];
    double u = img_pts[i][0];
    double v = img_pts[i][1];

    A(2*i, 0) = -X;
    A(2*i, 1) = -Y;
    A(2*i, 2) = -1;
    A(2*i, 6) =  u*X;
    A(2*i, 7) =  u*Y;
    A(2*i, 8) =  u;

    A(2*i+1, 3) = -X;
    A(2*i+1, 4) = -Y;
    A(2*i+1, 5) = -1;
    A(2*i+1, 6) =  v*X;
    A(2*i+1, 7) =  v*Y;
    A(2*i+1, 8) =  v;
  }

  Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
  Eigen::VectorXd h = svd.matrixV().col(8); // smallest singular value
  Eigen::Matrix3d H;
  H << h(0), h(1), h(2),
       h(3), h(4), h(5),
       h(6), h(7), h(8);
  return H / H(2,2); // normalize so H(2,2)=1
}

// v_ij helper for Zhang
static Eigen::Matrix<double,6,1> v_ij(const Eigen::Matrix3d& H, int i, int j) {
  // i,j are col indices 0 or 1
  double h1i = H(0, i), h2i = H(1, i), h3i = H(2, i);
  double h1j = H(0, j), h2j = H(1, j), h3j = H(2, j);

  Eigen::Matrix<double,6,1> v;
  v(0) = h1i * h1j;
  v(1) = h1i * h2j + h2i * h1j;
  v(2) = h2i * h2j;
  v(3) = h3i * h1j + h1i * h3j;
  v(4) = h3i * h2j + h2i * h3j;
  v(5) = h3i * h3j;
  return v;
}

// ================= 2. Intrinsics from homographies (Zhang) ===============
Intrinsics solveIntrinsicsFromHomographies(const std::vector<Eigen::Matrix3d>& Hs) {
  const int n = (int)Hs.size();
  Eigen::MatrixXd V(2*n, 6);
  for (int k = 0; k < n; ++k) {
    const auto& H = Hs[k];
    Eigen::Matrix<double,6,1> v01 = v_ij(H, 0, 1);
    Eigen::Matrix<double,6,1> v00 = v_ij(H, 0, 0);
    Eigen::Matrix<double,6,1> v11 = v_ij(H, 1, 1);
    V.row(2*k)     = v01.transpose();
    V.row(2*k + 1) = (v00 - v11).transpose();
  }

  Eigen::JacobiSVD<Eigen::MatrixXd> svd(V, Eigen::ComputeFullV);
  Eigen::VectorXd b = svd.matrixV().col(5); // smallest singular vector

  double B11 = b(0);
  double B12 = b(1);
  double B22 = b(2);
  double B13 = b(3);
  double B23 = b(4);
  double B33 = b(5);

  double v0 = (B12*B13 - B11*B23) / (B11*B22 - B12*B12);
  double lambda = B33 - (B13*B13 + v0*(B12*B13 - B11*B23)) / B11;
  double alpha = std::sqrt(lambda / B11);
  double beta  = std::sqrt(lambda*B11 / (B11*B22 - B12*B12));
  double gamma = -B12 * alpha*alpha * beta / lambda;
  double u0 = gamma * v0 / beta - B13 * alpha*alpha / lambda;

  Intrinsics K;
  K.fx = alpha;
  K.fy = beta;
  K.cx = u0;
  K.cy = v0;
  return K;
}

// ================= 3. Extrinsics from K and H (Zhang) ====================
Pose extrinsicsFromHomography(const Eigen::Matrix3d& H, const Intrinsics& K) {
  Eigen::Matrix3d Kmat;
  Kmat << K.fx, 0,    K.cx,
          0,    K.fy, K.cy,
          0,    0,    1.0;

  Eigen::Matrix3d Kinv = Kmat.inverse();
  Eigen::Vector3d h1 = H.col(0);
  Eigen::Vector3d h2 = H.col(1);
  Eigen::Vector3d h3 = H.col(2);

  Eigen::Vector3d r1 = Kinv * h1;
  Eigen::Vector3d r2 = Kinv * h2;
  Eigen::Vector3d t  = Kinv * h3;

  double lambda = 1.0 / ((r1.norm() + r2.norm()) * 0.5);
  r1 *= lambda;
  r2 *= lambda;
  t  *= lambda;
  Eigen::Vector3d r3 = r1.cross(r2);

  Eigen::Matrix3d R;
  R.col(0) = r1;
  R.col(1) = r2;
  R.col(2) = r3;

  // Orthonormalize R
  Eigen::JacobiSVD<Eigen::Matrix3d> svdR(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
  R = svdR.matrixU() * svdR.matrixV().transpose();

  Pose pose;
  pose.R = R;
  pose.t = t;
  return pose;
}

// ============ 4. Ceres residual for Brown–Conrady distortion =============
struct DistortionResidual {
  DistortionResidual(const Intrinsics& K,
                     const Pose& pose,
                     const Eigen::Vector3d& Pw,
                     double u_obs, double v_obs)
      : K_(K), R_(pose.R), t_(pose.t),
        Pw_(Pw), uo_(u_obs), vo_(v_obs) {}

  template <typename T>
  bool operator()(const T* const d, T* residuals) const {
    // d = [k1,k2,k3,p1,p2]
    const T k1 = d[0];
    const T k2 = d[1];
    const T k3 = d[2];
    const T p1 = d[3];
    const T p2 = d[4];

    Eigen::Matrix<T,3,1> Xw = Pw_.cast<T>();
    Eigen::Matrix<T,3,1> Xc = R_.cast<T>() * Xw + t_.cast<T>();

    T X = Xc(0), Y = Xc(1), Z = Xc(2);
    T xn = X / Z;
    T yn = Y / Z;
    T r2 = xn*xn + yn*yn;
    T r4 = r2*r2;
    T r6 = r2*r4;

    T s  = T(1) + k1*r2 + k2*r4 + k3*r6;

    T dx_t = T(2)*p1*xn*yn + p2*(r2 + T(2)*xn*xn);
    T dy_t = p1*(r2 + T(2)*yn*yn) + T(2)*p2*xn*yn;

    T xd = xn*s + dx_t;
    T yd = yn*s + dy_t;

    T u = T(K_.fx)*xd + T(K_.cx);
    T v = T(K_.fy)*yd + T(K_.cy);

    residuals[0] = u - T(uo_);
    residuals[1] = v - T(vo_);
    return true;
  }

  Intrinsics K_;
  Eigen::Matrix3d R_;
  Eigen::Vector3d t_;
  Eigen::Vector3d Pw_;
  double uo_, vo_;
};


// bundle adjustment 

struct FullReprojResidual {
  FullReprojResidual(const Eigen::Vector3d& Pw,
                     double u_obs, double v_obs)
      : Pw_(Pw), uo_(u_obs), vo_(v_obs) {}

  // intrinsics: [fx, fy, cx, cy]
  // distortion: [k1, k2, k3, p1, p2]
  // pose:       [ax, ay, az, tx, ty, tz]  (angle-axis + translation)
  template <typename T>
  bool operator()(const T* const intrinsics,
                  const T* const distortion,
                  const T* const pose,
                  T* residuals) const {
    const T fx = intrinsics[0];
    const T fy = intrinsics[1];
    const T cx = intrinsics[2];
    const T cy = intrinsics[3];

    const T k1 = distortion[0];
    const T k2 = distortion[1];
    const T k3 = distortion[2];
    const T p1 = distortion[3];
    const T p2 = distortion[4];

    const T* angle_axis = pose;      // pose[0..2]
    const T* t = pose + 3;           // pose[3..5]

    // World point
    T Pw[3] = { T(Pw_(0)), T(Pw_(1)), T(Pw_(2)) };

    // Rotate
    T Pc[3];
    ceres::AngleAxisRotatePoint(angle_axis, Pw, Pc);

    // Translate
    Pc[0] += t[0];
    Pc[1] += t[1];
    Pc[2] += t[2];

    // Normalized coordinates
    T xn = Pc[0] / Pc[2];
    T yn = Pc[1] / Pc[2];
    T r2 = xn*xn + yn*yn;
    T r4 = r2*r2;
    T r6 = r2*r4;

    // Radial
    T s  = T(1) + k1*r2 + k2*r4 + k3*r6;

    // Tangential
    T dx_t = T(2)*p1*xn*yn + p2*(r2 + T(2)*xn*xn);
    T dy_t = p1*(r2 + T(2)*yn*yn) + T(2)*p2*xn*yn;

    T xd = xn*s + dx_t;
    T yd = yn*s + dy_t;

    T u = fx * xd + cx;
    T v = fy * yd + cy;

    residuals[0] = u - T(uo_);
    residuals[1] = v - T(vo_);
    return true;
  }

  Eigen::Vector3d Pw_;
  double uo_, vo_;
};


std::vector<std::string> listImagesInFolder(std::string &folder_path){
    std::vector<std::string> files;
    for(const auto &entry: fs::directory_iterator(folder_path)){
        if (!entry.is_regular_file()) continue;
        auto ext = entry.path().extension().string();
        for (auto &c:ext) c=(char)std::tolower(c);

        if (ext == ".png" || ext == ".jpg" || ext == ".jpeg"){
            files.push_back(entry.path().string());
        }
    }
    std::sort(files.begin(), files.end());
    return files;
}

Eigen::Vector2d projectPointWithDistortion(
    const Intrinsics& K,
    const double d[5],          // [k1,k2,k3,p1,p2]
    const Pose& pose,
    const Eigen::Vector3d& Pw)
{
  // world -> camera
  Eigen::Vector3d Xc = pose.R * Pw + pose.t;
  double X = Xc(0), Y = Xc(1), Z = Xc(2);

  double xn = X / Z;
  double yn = Y / Z;
  double r2 = xn*xn + yn*yn;
  double r4 = r2*r2;
  double r6 = r2*r4;

  double k1 = d[0];
  double k2 = d[1];
  double k3 = d[2];
  double p1 = d[3];
  double p2 = d[4];

  double s = 1.0 + k1*r2 + k2*r4 + k3*r6;

  double dx_t = 2.0*p1*xn*yn + p2*(r2 + 2.0*xn*xn);
  double dy_t = p1*(r2 + 2.0*yn*yn) + 2.0*p2*xn*yn;

  double xd = xn*s + dx_t;
  double yd = yn*s + dy_t;

  double u = K.fx * xd + K.cx;
  double v = K.fy * yd + K.cy;

  return Eigen::Vector2d(u, v);
}

Eigen::Vector2d projectWithParams(
    const double intrinsics[4],
    const double distortion[5],
    const std::array<double,6>& pose,
    const Eigen::Vector3d& Pw){
    
        double fx = intrinsics[0];
        double fy = intrinsics[1];
        double cx = intrinsics[2];
        double cy = intrinsics[3];

        double k1 = distortion[0];
        double k2 = distortion[1];
        double k3 = distortion[2];
        double p1 = distortion[3];
        double p2 = distortion[4];

        const double* angle_axis = pose.data();
        const double* t = pose.data() + 3;

        double Pw_arr[3] = { Pw(0), Pw(1), Pw(2) };
        double Pc[3];
        ceres::AngleAxisRotatePoint(angle_axis, Pw_arr, Pc);
        Pc[0] += t[0];
        Pc[1] += t[1];
        Pc[2] += t[2];

        double xn = Pc[0] / Pc[2];
        double yn = Pc[1] / Pc[2];
        double r2 = xn*xn + yn*yn;
        double r4 = r2*r2;
        double r6 = r2*r4;

        double s  = 1.0 + k1*r2 + k2*r4 + k3*r6;

        double dx_t = 2.0*p1*xn*yn + p2*(r2 + 2.0*xn*xn);
        double dy_t = p1*(r2 + 2.0*yn*yn) + 2.0*p2*xn*yn;

        double xd = xn*s + dx_t;
        double yd = yn*s + dy_t;

        double u = fx * xd + cx;
        double v = fy * yd + cy;

        return Eigen::Vector2d(u, v);
}

// ==================== 5. Main: using real chessboard images ==============
int main(int argc, char** argv) {
  // ---- User-configurable parameters ----
  // 1) List of image paths (fill this with your images)
  std::string folder_path = "/mnt/d/courses/computer_vision/Home_Work/skashid@wpi_hw3/Calibration_Imgs/Calibration_Imgs";
  auto image_files = listImagesInFolder(folder_path);

  // 2) Chessboard pattern
  cv::Size board_size(9, 6);     // inner corners (columns, rows)
  double square_size = 0.04;     // in meters (or any unit)

  if (image_files.empty()) {
    std::cerr << "Please fill image_files with your calibration images.\n";
    return 1;
  }

  // ---- Step A: Detect corners in each image using OpenCV ----
  std::vector<std::vector<cv::Point2f>> img_points_cv;
  std::vector<std::vector<cv::Point3f>> obj_points_cv;

  // Prepare single-view object points (same for all views, Z=0)
  std::vector<cv::Point3f> objp;
  for (int y = 0; y < board_size.height; ++y) {
    for (int x = 0; x < board_size.width; ++x) {
      objp.emplace_back(
          (x - (board_size.width-1)/2.0f) * (float)square_size,
          (y - (board_size.height-1)/2.0f) * (float)square_size,
          0.0f);
    }
  }

  for (const auto& fname : image_files) {
    cv::Mat img = cv::imread(fname, cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
      std::cerr << "Failed to load image: " << fname << "\n";
      return 1;
    }

    std::vector<cv::Point2f> corners;
    bool found = cv::findChessboardCorners(
        img, board_size, corners,
        cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);

    if (!found) {
      std::cerr << "Chessboard not found in: " << fname << "\n";
      continue;
    }

    // refine corners
    cv::cornerSubPix(
        img, corners, cv::Size(11,11), cv::Size(-1,-1),
        cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.001));

    // draw the corners 
    cv::Mat img_color;
    cv::cvtColor(img, img_color, cv::COLOR_GRAY2BGR);

    // Draw the detected corners
    cv::drawChessboardCorners(img_color, board_size, corners, found);

    // Show window (optional)
    // std::cout<<"-------detecting the corneres"<<"\n";
    // cv::Mat display;
    // cv::resize(img_color, display, cv::Size(), 0.5, 0.5); // 50% scale
    // cv::imshow("Detected corners", display);
    // // Wait for key: press any key to go to next image, or ESC to break
    // int key = cv::waitKey(0);
    // if (key == 27) { // ESC
    //     std::cout << "ESC pressed, stopping visualization.\n";
    //     cv::destroyAllWindows();
    //     // you can break here if you want to stop processing images
    //     // break;
    // }

    img_points_cv.push_back(corners);
    obj_points_cv.push_back(objp);
  }

  int num_views = (int)img_points_cv.size();
  if (num_views < 3) {
    std::cerr << "Need at least 3 valid views for Zhang. Found: " << num_views << "\n";
    return 1;
  }

  std::cout << "Using " << num_views << " valid views for calibration.\n";

  // ---- Convert OpenCV points to Eigen ----
  const int N = board_size.width * board_size.height;
  std::vector<std::vector<Eigen::Vector2d>> world_pts_2d(num_views);
  std::vector<std::vector<Eigen::Vector2d>> img_pts_2d(num_views);
  std::vector<std::vector<Eigen::Vector3d>> world_pts_3d(num_views);

  for (int v = 0; v < num_views; ++v) {
    world_pts_2d[v].resize(N);
    img_pts_2d[v].resize(N);
    world_pts_3d[v].resize(N);
    for (int i = 0; i < N; ++i) {
      const cv::Point3f& P = obj_points_cv[v][i];
      const cv::Point2f& C = img_points_cv[v][i];
      world_pts_2d[v][i] = Eigen::Vector2d(P.x, P.y);
      img_pts_2d[v][i]   = Eigen::Vector2d(C.x, C.y);
      world_pts_3d[v][i] = Eigen::Vector3d(P.x, P.y, P.z); // Z=0
    }
  }

  // ---- Step B: Zhang – compute homographies ----
  std::vector<Eigen::Matrix3d> Hs(num_views);
  for (int v = 0; v < num_views; ++v) {
    Hs[v] = computeHomography(world_pts_2d[v], img_pts_2d[v]);
  }

  // ---- Step C: Zhang – solve intrinsics from homographies ----
  Intrinsics K_init = solveIntrinsicsFromHomographies(Hs);
  std::cout << "Initial K from Zhang:\n";
  std::cout << " fx = " << K_init.fx << " fy = " << K_init.fy
            << " cx = " << K_init.cx << " cy = " << K_init.cy << "\n";

  // ---- Step D: Extrinsics for each view ----
  std::vector<Pose> poses_init(num_views);
  for (int v = 0; v < num_views; ++v) {
    poses_init[v] = extrinsicsFromHomography(Hs[v], K_init);
  }

  // ---- Step E: build observation list (for Ceres) ----
  std::vector<Observation> observations;
  observations.reserve(num_views * N);
  for (int v = 0; v < num_views; ++v) {
    for (int i = 0; i < N; ++i) {
      Observation o;
      o.view_id = v;
      o.Pw = world_pts_3d[v][i];
      o.u = img_pts_2d[v][i][0];
      o.v = img_pts_2d[v][i][1];
      observations.push_back(o);
    }
  }

  // ---- Step F: Nonlinear solve for distortion with Ceres ----
//   Distortion D_est;
//   D_est.k1 = 0.0;
//   D_est.k2 = 0.0;
//   D_est.k3 = 0.0;
//   D_est.p1 = 0.0;
//   D_est.p2 = 0.0;
//   double d[5] = {D_est.k1, D_est.k2, D_est.k3, D_est.p1, D_est.p2};

//   ceres::Problem problem;
//   ceres::LossFunction* loss = new ceres::HuberLoss(1.0);

//   for (const auto& o : observations) {
//     const Pose& P = poses_init[o.view_id];
//     auto* cost = new ceres::AutoDiffCostFunction<DistortionResidual, 2, 5>(
//         new DistortionResidual(K_init, P, o.Pw, o.u, o.v));
//     problem.AddResidualBlock(cost, loss, d);
//   }

//   ceres::Solver::Options opts;
//   opts.linear_solver_type = ceres::DENSE_QR;
//   opts.max_num_iterations = 100;
//   opts.minimizer_progress_to_stdout = true;

//   ceres::Solver::Summary summary;
//   ceres::Solve(opts, &problem, &summary);
//   std::cout << summary.BriefReport() << "\n";

//   std::cout << "Estimated distortion:\n";
//   std::cout << " k1 = " << d[0]
//             << " k2 = " << d[1]
//             << " k3 = " << d[2]
//             << " p1 = " << d[3]
//             << " p2 = " << d[4] << "\n";

    // ---- Step F: Nonlinear solve for K, distortion, and poses with Ceres ----

    // Intrinsics: [fx, fy, cx, cy]
    double intrinsics[4] = { K_init.fx, K_init.fy, K_init.cx, K_init.cy };

    // Distortion: [k1, k2, k3, p1, p2]
    double distortion[5] = { 0.0, 0.0, 0.0, 0.0, 0.0 };  // start from zero, or some guess

    // Poses: one 6D vector per view: [ax, ay, az, tx, ty, tz]
    std::vector<std::array<double, 6>> pose_params(num_views);

    for (int v = 0; v < num_views; ++v) {
        // Convert R (Eigen) to angle-axis
        // Convert R (Eigen) to angle-axis safely using Eigen
        Eigen::AngleAxisd aa(poses_init[v].R);
        Eigen::Vector3d aa_vec = aa.angle() * aa.axis();

        pose_params[v][0] = aa_vec(0);
        pose_params[v][1] = aa_vec(1);
        pose_params[v][2] = aa_vec(2);

        pose_params[v][3] = poses_init[v].t(0);
        pose_params[v][4] = poses_init[v].t(1);
        pose_params[v][5] = poses_init[v].t(2);
    }

    // create the paramer blocks
    ceres::Problem problem;
    ceres::LossFunction* loss = new ceres::HuberLoss(1.0);

    problem.AddParameterBlock(intrinsics, 4);
    problem.AddParameterBlock(distortion, 5);
    for(int v=0; v<num_views; v++){
        problem.AddParameterBlock(pose_params[v].data(), 6);
    }
    
    // add residual block
    for(const auto &o : observations){
        int v = o.view_id;
        auto * cost = new ceres::AutoDiffCostFunction<FullReprojResidual, 2, 4, 5, 6>(
            new  FullReprojResidual(o.Pw, o.u, o.v));

        problem.AddResidualBlock(
            cost,
            loss,
            intrinsics,
            distortion,
            pose_params[v].data()
        );
    }

    //solve 
    ceres::Solver::Options opts;
    opts.linear_solver_type = ceres::DENSE_SCHUR;   // TODO: use the SPARSE_SCHUR insted
    opts.max_num_iterations = 100;
    opts.minimizer_progress_to_stdout = false;

    ceres::Solver::Summary summary;
    ceres::Solve(opts, &problem, &summary);
    std::cout<< summary.BriefReport()<<"\n";

    std::cout << "Optimized intrinsics:\n";
    std::cout << " fx = " << intrinsics[0]
            << " fy = " << intrinsics[1]
            << " cx = " << intrinsics[2]
            << " cy = " << intrinsics[3] << "\n";

    std::cout << "Optimized distortion:\n";
    std::cout << " k1 = " << distortion[0]
            << " k2 = " << distortion[1]
            << " k3 = " << distortion[2]
            << " p1 = " << distortion[3]
            << " p2 = " << distortion[4] << "\n";

    // ===== Reprojection error per image (no OpenCV) =====
    // double total_err_sq = 0.0;
    // size_t total_points = 0;

    // std::cout << "\nReprojection error per image:\n";

    // for (int v = 0; v < num_views; ++v) {
    // double err_sq = 0.0;
    // size_t Npts = (size_t)N;

    // for (int i = 0; i < N; ++i) {
    //     const Eigen::Vector3d& Pw = world_pts_3d[v][i];
    //     const Eigen::Vector2d& uv_obs = img_pts_2d[v][i];

    //     Eigen::Vector2d uv_pred =
    //         projectPointWithDistortion(K_init, d, poses_init[v], Pw);

    //     Eigen::Vector2d diff = uv_pred - uv_obs;
    //     double e = diff.norm();          // Euclidean error in pixels
    //     err_sq += e * e;
    // }

    // double rms = std::sqrt(err_sq / (double)Npts);
    // std::cout << "  View " << v << ": RMS reprojection error = "
    //             << rms << " pixels\n";

    // total_err_sq += err_sq;
    // total_points += Npts;
    // }

    // double total_rms = std::sqrt(total_err_sq / (double)total_points);
    // std::cout << "Overall RMS reprojection error = "
    //         << total_rms << " pixels\n";

    // print the reprojection erro 
    double total_err_sq = 0.0;
    size_t total_points = 0;

    std::cout << "\nReprojection error per image (full BA):\n";

    for (int v = 0; v < num_views; ++v) {
    double err_sq = 0.0;
    for (int i = 0; i < N; ++i) {
        const Eigen::Vector3d& Pw = world_pts_3d[v][i];
        const Eigen::Vector2d& uv_obs = img_pts_2d[v][i];

        Eigen::Vector2d uv_pred = projectWithParams(
            intrinsics, distortion, pose_params[v], Pw);

        Eigen::Vector2d diff = uv_pred - uv_obs;
        double e = diff.norm();
        err_sq += e*e;
    }
    double rms = std::sqrt(err_sq / (double)N);
    std::cout << "  View " << v << ": RMS = " << rms << " px\n";

    total_err_sq += err_sq;
    total_points += N;
    }

    double total_rms = std::sqrt(total_err_sq / (double)total_points);
    std::cout << "Overall RMS reprojection error (full BA) = "
            << total_rms << " px\n";



  return 0;
}
