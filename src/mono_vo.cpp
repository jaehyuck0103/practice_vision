#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/video/tracking.hpp>

#include <fmt/core.h>
#include <fmt/printf.h>
#include <range/v3/all.hpp>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

namespace fs = std::filesystem;
using namespace ranges;

const fs::path KITTI_ROOT{"../KITTI_ODOM"};

std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>>
featureDetTrack(const cv::Mat &img_prev, const cv::Mat &img_next) {

    auto detector = cv::FastFeatureDetector::create(20);

    std::vector<cv::KeyPoint> keypoints;
    detector->detect(img_prev, keypoints);

    auto pts_prev =
        keypoints | views::transform([](const cv::KeyPoint &x) { return x.pt; }) | to<std::vector>;

    std::vector<uint8_t> status;
    std::vector<float> err;
    std::vector<cv::Point2f> pts_next;
    cv::calcOpticalFlowPyrLK(img_prev, img_next, pts_prev, pts_next, status, err);

    // get rid of points for which the KLT tracking failed or those who have gone outside the frame
    std::vector<cv::Point2f> points_prev_valid;
    std::vector<cv::Point2f> points_next_valid;
    for (size_t i = 0; i < status.size(); i++) {
        const auto &pt = pts_next.at(i);
        if (status.at(i) && pt.x >= 0 && pt.y >= 0 && pt.x < img_prev.cols &&
            pt.y < img_prev.rows) {

            points_prev_valid.emplace_back(pts_prev.at(i));
            points_next_valid.emplace_back(pt);
        }
    }
    return {points_prev_valid, points_next_valid};
}

std::vector<std::array<double, 3>> getGtPoses(int sequence_id) {

    std::vector<std::array<double, 3>> gt_poses;
    std::ifstream myfile{KITTI_ROOT / fmt::format("poses/{:02d}.txt", sequence_id)};
    if (myfile.is_open()) {
        for (std::string line; getline(myfile, line);) {
            auto nums = line | views::split(' ') | to<std::vector<std::string>>;
            gt_poses.push_back(
                {std::stod(nums.at(3)), std::stod(nums.at(7)), std::stod(nums.at(11))});
        }
    } else {
        std::cout << "Unable to open file\n";
        abort();
    }

    return gt_poses;
}

cv::Matx33d readIntrinsic(int sequence_id) {

    std::ifstream myfile{KITTI_ROOT / fmt::format("sequences/{:02d}/calib.txt", sequence_id)};
    if (myfile.is_open()) {
        std::string line;
        getline(myfile, line);
        auto nums = line | views::split(' ') | to<std::vector<std::string>>;
        return {
            std::stod(nums.at(1)),
            std::stod(nums.at(2)),
            std::stod(nums.at(3)),
            std::stod(nums.at(5)),
            std::stod(nums.at(6)),
            std::stod(nums.at(7)),
            std::stod(nums.at(9)),
            std::stod(nums.at(10)),
            std::stod(nums.at(11))};
    } else {
        std::cout << "Unable to open file\n";
        abort();
    }
}

int main() {
    int seq_id = 0;
    fs::path seq_dir = KITTI_ROOT / fmt::format("sequences/{:02d}", seq_id);

    const auto dir_iter = fs::directory_iterator(seq_dir / "image_0");
    const auto img_paths =
        dir_iter | views::filter([](const auto &x) { return x.path().extension() == ".png"; }) |
        to<std::vector<fs::path>> | actions::sort;

    auto gt_poses = getGtPoses(seq_id);
    if (gt_poses.size() != img_paths.size()) {
        fmt::print("len_scales({}) != len_img_paths({})\n", gt_poses.size(), img_paths.size());
        abort();
    }
    cv::Mat traj = cv::Mat::zeros(1000, 1000, CV_8UC3);
    cv::Mat ego_pose = cv::Mat::eye(4, 4, CV_64F);

    for (const auto &pos : gt_poses) {
        cv::circle(traj, {int(pos.at(0)) + 500, -int(pos.at(2)) + 500}, 1, CV_RGB(0, 0, 255), 2);
    }

    // Read Camera Matrix
    const auto K = readIntrinsic(seq_id);

    for (size_t i = 0; i < img_paths.size() - 1; ++i) {
        auto img_prev = cv::imread(img_paths[i], cv::IMREAD_GRAYSCALE);
        auto img_next = cv::imread(img_paths[i + 1], cv::IMREAD_GRAYSCALE);

        auto [pts_prev, pts_next] = featureDetTrack(img_prev, img_next);

        // Recover the essential matrix and the pose
        cv::Mat E, R, t, mask;
        E = cv::findEssentialMat(pts_next, pts_prev, K, cv::RANSAC, 0.999, 1.0, 1000, mask);
        cv::recoverPose(E, pts_next, pts_prev, K, R, t, mask);

        // Update pose
        const double scale = sqrt(
            pow(gt_poses.at(i + 1).at(0) - gt_poses.at(i).at(0), 2) +
            pow(gt_poses.at(i + 1).at(1) - gt_poses.at(i).at(1), 2) +
            pow(gt_poses.at(i + 1).at(2) - gt_poses.at(i).at(2), 2));

        if (scale > 0.1 && t.at<double>(2) > t.at<double>(0) &&
            t.at<double>(2) > t.at<double>(1)) {

            t *= scale;

            cv::Mat ego_motion = cv::Mat::eye({4, 4}, CV_64F);
            R.copyTo(ego_motion(cv::Rect(0, 0, 3, 3)));
            t.copyTo(ego_motion(cv::Rect(3, 0, 1, 3)));
            ego_pose = ego_pose * ego_motion;
        } else {
            std::cout << "scale below 0.1, or incorrect translation\n";
        }

        // Display
        fmt::print(
            "Coordinates: x = {:02f}m y = {:02f}m z = {:02f}m\n",
            ego_pose.at<double>(0, 3),
            ego_pose.at<double>(1, 3),
            ego_pose.at<double>(2, 3));

        const auto x = int(ego_pose.at<double>(0, 3)) + 500;
        const auto y = -int(ego_pose.at<double>(2, 3)) + 500;
        cv::circle(traj, {x, y}, 1, CV_RGB(255, 0, 0), 2);

        cv::imshow("Camera", img_next);
        cv::imshow("Trajectory", traj);
        cv::waitKey(1);
    }

    return 0;
}
