// Newton RGB pair (ZMQ 5557) -> M3T RGB-only tracker -> pose (ZMQ 5559).
// M3T has its own camera-0-centered world; emitted poses use Newton world.
// No Newton ground-truth input is used by this process.
#include <m3t/static_detector.h>
#include <m3t/body.h>
#include <m3t/camera.h>
#include <m3t/common.h>
#include <m3t/link.h>
#include <m3t/detector.h>
#include <m3t/normal_viewer.h>
#include <m3t/optimizer.h>
#include <m3t/publisher.h>
#include <m3t/region_modality.h>
#include <m3t/region_model.h>
#include <m3t/renderer_geometry.h>
#include <m3t/tracker.h>
#include <Eigen/Geometry>
#include <nlohmann/json.hpp>
#include <opencv2/imgproc.hpp>
#include <zmq.hpp>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <set>
#include <utility>

namespace fs = std::filesystem;
using Json = nlohmann::json;

struct FrameInfo {
  int64_t frame_id = -1;
  double timestamp_s = 0;
  bool valid = false;
};

m3t::Transform3fA ReadPose(const Json& matrix) {
  if (!matrix.is_array() || matrix.size() != 4) {
    throw std::runtime_error("Expected 4x4 camera transform");
  }
  m3t::Transform3fA result = m3t::Transform3fA::Identity();
  for (int r = 0; r < 4; ++r) {
    if (!matrix.at(r).is_array() || matrix.at(r).size() != 4)
      throw std::runtime_error("Invalid transform row");
    for (int c = 0; c < 4; ++c)
      result.matrix()(r, c) = matrix.at(r).at(c).get<float>();
  }
  return result;
}

struct Calibration {
  int width = 0, height = 0;
  m3t::Intrinsics intr{};
  m3t::Transform3fA newton_from_camera = m3t::Transform3fA::Identity();
};

std::array<Calibration, 2> LoadCalibration(const fs::path& path) {
  std::ifstream file(path);
  if (!file) throw std::runtime_error("Calibration not found: " + path.string());
  const Json data = Json::parse(file);
  if (data.at("cameras").size() != 2)
    throw std::runtime_error("Expected exactly two cameras in calibration");
  std::array<Calibration, 2> cameras;
  for (int i = 0; i < 2; ++i) {
    const auto& c = data.at("cameras").at(i);
    if (c.at("id").get<int>() != i)
      throw std::runtime_error("Unexpected camera IDs in calibration");
    auto& out = cameras.at(i);
    out.width = c.at("width").get<int>();
    out.height = c.at("height").get<int>();
    const auto& K = c.at("K_opencv");
    out.intr = {K.at(0).at(0).get<float>(), K.at(1).at(1).get<float>(),
                K.at(0).at(2).get<float>(), K.at(1).at(2).get<float>(),
                out.width, out.height};
    out.newton_from_camera = ReadPose(c.at("T_world_from_camera_opencv"));
    if (out.width <= 0 || out.height <= 0 ||
        out.width > 4096 || out.height > 4096)
      throw std::runtime_error("Invalid camera resolution");
  }
  return cameras;
}

class SharedPairSource {
 public:
  SharedPairSource(const std::string& endpoint, int width, int height)
      : ctx_(1), subscriber_(ctx_, zmq::socket_type::sub),
        width_(width), height_(height) {
    subscriber_.set(zmq::sockopt::subscribe, "");
    subscriber_.set(zmq::sockopt::rcvhwm, 2);
    subscriber_.set(zmq::sockopt::rcvtimeo, 20000);
    subscriber_.connect(endpoint);
    std::cout << "[m3t] Receiving RGB pairs from " << endpoint << std::endl;
  }

  // M3T updates two cameras sequentially. Serve the SAME image pair to both,
  // regardless of which camera M3T asks for first. Advance only after both.
  bool NextImage(int camera_index, cv::Mat* image) {
    if (camera_index < 0 || camera_index > 1) return false;
    if (!info_.valid || consumed_mask_ == 3) {
      if (!ReceiveLatest()) return false;
      consumed_mask_ = 0;
    }
    const int bit = 1 << camera_index;
    if (consumed_mask_ & bit) {
      std::cerr << "[m3t] Camera called twice in the same cycle" << std::endl;
      return false;
    }
    *image = bgr_[camera_index];
    consumed_mask_ |= bit;
    return true;
  }

  const FrameInfo& current_info() const { return info_; }

 private:
  bool ReceivePacket(zmq::recv_flags first_flag) {
    zmq::message_t parts[3];
    for (int i = 0; i < 3; ++i) {
      const auto result = subscriber_.recv(parts[i],
          i == 0 ? first_flag : zmq::recv_flags::none);
      if (!result) return false;
      const bool more = subscriber_.get(zmq::sockopt::rcvmore);
      if (more != (i < 2)) {
        std::cerr << "[m3t] Unexpected multipart packet length" << std::endl;
        return false;
      }
    }
    Json h;
    try {
      h = Json::parse(std::string(static_cast<const char*>(parts[0].data()),
                                  parts[0].size()));
      if (h.at("width").get<int>() != width_ ||
          h.at("height").get<int>() != height_ ||
          h.at("camera_count").get<int>() != 2 ||
          h.at("pixel_format").get<std::string>() != "RGB8") {
        std::cerr << "[m3t] Frame format does not match camera configuration" << std::endl;
        return false;
      }
      const size_t nbytes = size_t(width_) * size_t(height_) * 3;
      for (int i = 0; i < 2; ++i) {
        if (parts[i + 1].size() != nbytes) {
          std::cerr << "[m3t] Incorrect RGB image size" << std::endl;
          return false;
        }
        cv::Mat rgb(height_, width_, CV_8UC3, parts[i + 1].data());
        cv::cvtColor(rgb, bgr_[i], cv::COLOR_RGB2BGR);
      }
      info_ = {h.at("frame_id").get<int64_t>(),
               h.at("timestamp_s").get<double>(), true};
    } catch (const std::exception& e) {
      std::cerr << "[m3t] Bad frame packet: " << e.what() << std::endl;
      return false;
    }
    return true;
  }

  bool ReceiveLatest() {
    try {
      if (!ReceivePacket(zmq::recv_flags::none)) {
        std::cerr << "[m3t] Frame timeout/bad packet; is Newton running?" << std::endl;
        return false;
      }
      // Keep the newest COMPLETE pair already queued, avoiding extra latency.
      // Receiver high-water mark is 2; multipart frames are atomic in ZeroMQ.
      while (subscriber_.get(zmq::sockopt::events) & ZMQ_POLLIN) {
        if (!ReceivePacket(zmq::recv_flags::dontwait)) break;
      }
      return true;
    } catch (const zmq::error_t& e) {
      std::cerr << "[m3t] ZMQ receive error: " << e.what() << std::endl;
      return false;
    }
  }

  zmq::context_t ctx_;
  zmq::socket_t subscriber_;
  int width_, height_;
  cv::Mat bgr_[2];
  FrameInfo info_;
  int consumed_mask_ = 3;
};

class NewtonColorCamera final : public m3t::ColorCamera {
 public:
  NewtonColorCamera(const std::string& name, int index,
                    std::shared_ptr<SharedPairSource> source,
                    const Calibration& calibration,
                    const m3t::Transform3fA& camera2m3t_world)
      : m3t::ColorCamera(name), index_(index), source_(std::move(source)) {
    intrinsics_ = calibration.intr;
    set_camera2world_pose(camera2m3t_world);
  }
  bool SetUp() override {
    set_up_ = true;
    return true;
  }
  bool UpdateImage(bool /*synchronized*/) override {
    if (!set_up_) return false;
    return source_->NextImage(index_, &image_);
  }
 private:
  int index_;
  std::shared_ptr<SharedPairSource> source_;
};

// RGB-only automatic initializer for the distinctively blue Newton cube.
// M3T has no built-in image-based auto-detector. This intentionally uses camera
// pixels, camera calibration, known tabletop height and known initial upright
// orientation; it NEVER reads Newton object state or ground-truth poses.
class AutoBlueCubeDetector final : public m3t::Detector {
 public:
  AutoBlueCubeDetector(const std::shared_ptr<m3t::Optimizer>& optimizer,
                       const std::shared_ptr<m3t::ColorCamera>& cam0,
                       const std::shared_ptr<m3t::ColorCamera>& cam1,
                       const Calibration& calibration0,
                       const Calibration& calibration1)
      : m3t::Detector("automatic_blue_cube_detector", true),
        optimizer_(optimizer), cam0_(cam0), cam1_(cam1),
        calibration0_(calibration0), calibration1_(calibration1) {}

  bool SetUp() override {
    set_up_ = optimizer_ && optimizer_->set_up() &&
              cam0_ && cam0_->set_up() && cam1_ && cam1_->set_up();
    return set_up_;
  }

  bool DetectPoses(const std::set<std::string>& names,
                   std::set<std::string>* detected_names = nullptr) override {
    if (!set_up_) return false;
    if (!names.count(optimizer_->name())) return true;
    ++attempts_;
    cv::Point2f uv0, uv1;
    if (!FindBlueCubeCenter(cam0_->image(), &uv0) ||
        !FindBlueCubeCenter(cam1_->image(), &uv1)) {
      if (attempts_ % 30 == 1)
        std::cout << "[m3t] Auto-init: waiting for blue cube masks in BOTH cameras\n";
      return true;  // No detection yet; M3T will retry next camera pair.
    }

    // The upright 3-inch cube center is 1.5 inches above Newton's z=0 table.
    // Use both camera observations, not Newton's body position.
    constexpr float kCubeCenterZ = 0.0381f;
    Eigen::Vector3f w0, w1;
    if (!PointOnKnownHeightPlane(uv0, calibration0_, kCubeCenterZ, &w0) ||
        !PointOnKnownHeightPlane(uv1, calibration1_, kCubeCenterZ, &w1)) {
      if (attempts_ % 30 == 1)
        std::cout << "[m3t] Auto-init: invalid camera ray / calibration\n";
      return true;
    }
    const float disagreement_m = (w0.head<2>() - w1.head<2>()).norm();
    if (disagreement_m > 0.10f) {
      if (attempts_ % 30 == 1)
        std::cout << "[m3t] Auto-init: two-camera centers disagree by "
                  << disagreement_m << " m; check blue mask/calibration\n";
      return true;
    }
    Eigen::Vector3f world_center = 0.5f * (w0 + w1);
    world_center.z() = kCubeCenterZ;

    // Initial Newton cube orientation is upright and axis-aligned. The cube's
    // rotation is NOT read from Newton; M3T will refine it while tracking.
    m3t::Transform3fA newton_from_cube = m3t::Transform3fA::Identity();
    newton_from_cube.translation() = world_center;

    // This M3T instance uses camera 0 as its internal world reference frame.
    const m3t::Transform3fA m3t_from_cube =
        calibration0_.newton_from_camera.inverse() * newton_from_cube;
    UpdatePoses(m3t_from_cube, optimizer_);
    if (detected_names) detected_names->insert(optimizer_->name());
    std::cout << "[m3t] Auto-initialized from two RGB views: Newton XYZ=["
              << world_center.x() << ", " << world_center.y() << ", "
              << world_center.z() << "] m (camera disagreement "
              << disagreement_m * 1000.0f << " mm)\n";
    return true;
  }

  std::vector<std::shared_ptr<m3t::Optimizer>> optimizer_ptrs() const override {
    return {optimizer_};
  }
  std::shared_ptr<m3t::Camera> camera_ptr() const override { return cam0_; }

 private:
  static bool FindBlueCubeCenter(const cv::Mat& bgr, cv::Point2f* center) {
    if (bgr.empty() || bgr.type() != CV_8UC3) return false;
    cv::Mat hsv, mask;
    cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);
    // For the Newton default --cube-color-rgb 35 100 230. OpenCV hue is 0..179.
    // Widen/narrow these thresholds if lighting changes the rendered pixels.
    cv::inRange(hsv, cv::Scalar(90, 65, 35), cv::Scalar(135, 255, 255), mask);
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN,
                     cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));
    cv::Mat labels, stats, centroids;
    const int n = cv::connectedComponentsWithStats(
        mask, labels, stats, centroids, 8, CV_32S);
    int selected = -1, best_area = 70;
    for (int i = 1; i < n; ++i) {
      int area = stats.at<int>(i, cv::CC_STAT_AREA);
      if (area > best_area && area < bgr.rows * bgr.cols / 3) {
        best_area = area;
        selected = i;
      }
    }
    if (selected < 0) return false;
    *center = cv::Point2f(float(centroids.at<double>(selected, 0)),
                          float(centroids.at<double>(selected, 1)));
    return true;
  }

  static bool PointOnKnownHeightPlane(const cv::Point2f& pixel,
                                      const Calibration& cal, float z,
                                      Eigen::Vector3f* world_point) {
    const auto& K = cal.intr;
    const Eigen::Vector3f direction_camera(
        (pixel.x - K.ppu) / K.fu, (pixel.y - K.ppv) / K.fv, 1.0f);
    const Eigen::Vector3f direction_world =
        cal.newton_from_camera.linear() * direction_camera;
    const Eigen::Vector3f camera_origin =
        cal.newton_from_camera.translation();
    if (std::abs(direction_world.z()) < 1e-5f) return false;
    const float distance_along_ray =
        (z - camera_origin.z()) / direction_world.z();
    if (!(distance_along_ray > 0) || !std::isfinite(distance_along_ray))
      return false;
    *world_point = camera_origin + distance_along_ray * direction_world;
    return world_point->allFinite();
  }

  std::shared_ptr<m3t::Optimizer> optimizer_;
  std::shared_ptr<m3t::ColorCamera> cam0_, cam1_;
  Calibration calibration0_, calibration1_;
  int attempts_ = 0;
};

class JsonPosePublisher final : public m3t::Publisher {
 public:
  JsonPosePublisher(std::shared_ptr<m3t::Body> body,
                    std::shared_ptr<SharedPairSource> source,
                    const m3t::Transform3fA& newton_from_m3t,
                    const std::string& endpoint)
      : m3t::Publisher("newton_pose_publisher"), body_(std::move(body)),
        source_(std::move(source)), newton_from_m3t_(newton_from_m3t),
        ctx_(1), publisher_(ctx_, zmq::socket_type::pub) {
    publisher_.set(zmq::sockopt::sndhwm, 2);
    publisher_.set(zmq::sockopt::linger, 0);
    publisher_.bind(endpoint);
    std::cout << "[m3t] Publishing poses at " << endpoint << std::endl;
  }
  bool SetUp() override {
    set_up_ = body_->set_up();
    return set_up_;
  }
  bool UpdatePublisher(int /*iteration*/) override {
    const auto& info = source_->current_info();
    if (!info.valid) return true;
    const auto& local_pose = body_->body2world_pose();
    // M3T's undetected body defaults to z=-10 m. Do not draw that as a pose.
    if (local_pose.translation().z() < -2.0f) return true;
    const m3t::Transform3fA pose = newton_from_m3t_ * local_pose;
    Eigen::Quaternionf q(pose.linear());
    q.normalize();
    const Eigen::Vector3f p = pose.translation();
    if (!p.allFinite() || !q.coeffs().allFinite()) return true;
    Json result = {
      {"frame_id", info.frame_id},
      {"timestamp_s", info.timestamp_s},
      {"position_m", {p.x(), p.y(), p.z()}},
      {"quat_xyzw", {q.x(), q.y(), q.z(), q.w()}},
      {"valid", true}
    };
    const std::string payload = result.dump();
    try {
      (void)publisher_.send(zmq::buffer(payload), zmq::send_flags::dontwait);
    } catch (const zmq::error_t& e) {
      std::cerr << "[m3t] Pose send error: " << e.what() << std::endl;
    }
    return true;
  }
 private:
  std::shared_ptr<m3t::Body> body_;
  std::shared_ptr<SharedPairSource> source_;
  m3t::Transform3fA newton_from_m3t_;
  zmq::context_t ctx_;
  zmq::socket_t publisher_;
};

int main(int argc, char** argv) {
  if (argc < 4 || argc > 6) {
    std::cerr << "Usage: run_newton_stream CAMERA_JSON CUBE_OBJ MODEL_CACHE_DIR "
                 "[FRAMES_ENDPOINT] [POSE_BIND_ENDPOINT]\n";
    return 2;
  }
  try {
    const fs::path calibration_path = argv[1];
    const fs::path mesh_path = argv[2];
    const fs::path cache_dir = argv[3];
    const std::string frame_endpoint = argc >= 5 ? argv[4] : "tcp://127.0.0.1:5557";
    const std::string pose_endpoint = argc >= 6 ? argv[5] : "tcp://127.0.0.1:5559";
    fs::create_directories(cache_dir);
    const auto c = LoadCalibration(calibration_path);
    if (c[0].width != c[1].width || c[0].height != c[1].height)
      throw std::runtime_error("Two streamed cameras must use matching resolutions");
    if (!fs::exists(mesh_path)) throw std::runtime_error("Cube mesh not found");

    // Keep M3T's internal world equal to camera 0 for compatibility with the
    // existing dashboard publisher. Relative camera 1 pose:
    // T_m3t_from_cam1 = inv(T_newton_from_cam0) * T_newton_from_cam1.
    const m3t::Transform3fA newton_from_m3t = c[0].newton_from_camera;
    const m3t::Transform3fA m3t_from_camera1 =
        newton_from_m3t.inverse() * c[1].newton_from_camera;
    auto source = std::make_shared<SharedPairSource>(frame_endpoint,
                                                      c[0].width, c[0].height);
    auto cam0 = std::make_shared<NewtonColorCamera>("newton_cam0", 0, source,
        c[0], m3t::Transform3fA::Identity());
    auto cam1 = std::make_shared<NewtonColorCamera>("newton_cam1", 1, source,
        c[1], m3t_from_camera1);

    auto tracker = std::make_shared<m3t::Tracker>("newton_m3t");
    tracker->set_start_tracking_after_detection(true);
    auto render_geometry = std::make_shared<m3t::RendererGeometry>("render_geometry");
    // Mesh vertices must be in meters and centered at the cube's body origin.
    auto cube = std::make_shared<m3t::Body>("cube", mesh_path,
        1.0f, true, true, m3t::Transform3fA::Identity());
    render_geometry->AddBody(cube);
    auto model = std::make_shared<m3t::RegionModel>(
        "cube_region_model", cube, cache_dir / "cube_region_model.bin");
    auto region0 = std::make_shared<m3t::RegionModality>(
        "region_cam0", cube, cam0, model);
    auto region1 = std::make_shared<m3t::RegionModality>(
        "region_cam1", cube, cam1, model);
    auto link = std::make_shared<m3t::Link>("cube_link", cube);
    link->AddModality(region0);
    link->AddModality(region1);
    auto optimizer = std::make_shared<m3t::Optimizer>("cube_optimizer", link);
    tracker->AddOptimizer(optimizer);

    // -----------------------------------------------------------
    // INITIALIZATION USING KNOWN STARTING POSE
    // -----------------------------------------------------------

    // Initial cube pose in Newton's world coordinates.
    // Position is in meters.
    //
    // Cube dimensions: 0.0762 x 0.0762 x 0.0762 m
    // Table surface: z = 0
    // Cube center height: 0.0381 m

    m3t::Transform3fA initial_cube_newton =
        m3t::Transform3fA::Identity();

    initial_cube_newton.translation() =
        Eigen::Vector3f(0.0f, 0.0f, 0.0381f);

    // Initial orientation is identity.
    // The cube starts aligned with Newton's world axes.

    // Convert Newton's world coordinates into M3T's
    // camera-0-centered coordinate system.

    m3t::Transform3fA initial_cube_m3t =
        newton_from_m3t.inverse() * initial_cube_newton;

    // Initialize M3T using the known starting pose.

    auto detector = std::make_shared<m3t::StaticDetector>(
        "known_initial_cube_pose",
        optimizer,
        initial_cube_m3t
    );

    tracker->AddDetector(detector);

std::cout
    << "[m3t] Using known initial cube pose." << std::endl
    << "[m3t] Position: (0, 0, 0.0381) m" << std::endl
    << "[m3t] Orientation: identity" << std::endl;
    tracker->AddPublisher(std::make_shared<JsonPosePublisher>(
        cube, source, newton_from_m3t, pose_endpoint));
    tracker->AddViewer(std::make_shared<m3t::NormalColorViewer>(
        "m3t_cam0_overlay", cam0, render_geometry));
    tracker->AddViewer(std::make_shared<m3t::NormalColorViewer>(
        "m3t_cam1_overlay", cam1, render_geometry));

    std::cout
    << "[m3t] Configured RGB-only multi-camera tracking.\n"
    << "[m3t] Initializing from known starting pose.\n"
    << "[m3t] Press 'q' in a tracking viewer to quit.\n";
    if (!tracker->SetUp()) {
      std::cerr << "[m3t] Tracker setup failed" << std::endl;
      return 1;
    }
    // Attempt auto-detection immediately and retry until both masks succeed.
    // Start tracking automatically AFTER successful detection.
    if (!tracker->RunTrackerProcess(true, false)) {
      std::cerr << "[m3t] Tracker loop ended with failure" << std::endl;
      return 1;
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[m3t] Error: " << e.what() << std::endl;
    return 1;
  }
}
