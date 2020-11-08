#include <algorithm>
#include <chrono>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/tracking.hpp>

#define WINDOW_NAME "Live"
#define MODEL_PATH                                                             \
  "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"
#define BOX_THICKNESS 2

static double calculateIou(const cv::Rect2d &rect1, const cv::Rect2d &rect2) {
  auto intersectionArea = (rect1 & rect2).area();
  return intersectionArea / (rect1.area() + rect2.area() - intersectionArea);
}

struct MatchResult {
  std::vector<std::tuple<int, int>> matches;
  std::vector<int> unmatchedFaces;
  std::vector<int> unmatchedTracks;
};

class FaceDetector {
public:
  FaceDetector() : faceClassifier{MODEL_PATH} {}

  std::vector<cv::Rect> detectFaces(cv::Mat bgrFrame) {
    auto t1 = std::chrono::high_resolution_clock::now();

    cv::Mat greyFrame;
    cv::cvtColor(bgrFrame, greyFrame, cv::COLOR_BGR2GRAY);
    cv::equalizeHist(greyFrame, greyFrame);

    std::vector<cv::Rect> faces;
    faceClassifier.detectMultiScale(greyFrame, faces);
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "Detect faces time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1)
                     .count()
              << " ms\n";
    return faces;
  };

  // std::vector<cv::Rect> detectFacesWithRoi(cv::Mat bgrFrame) {
  //   try to only run detection in those regions where we last saw
  //   a face.
  // };
  // std::vector<cv::Rect> lastSeenFaces;

private:
  cv::CascadeClassifier faceClassifier;
};

/*
  How tracking should work:
  - Detect face
  - Use face to init tracker
  - For the next X frames, use the tracker->update
  - After X frames, run face detection in an ROI around the current box
    and reinitialize the tracker or kill it
  - If the tracker->update fails, run detection again in an ROI of the track

  Adding new tracks:
  - Every so often, I should run the face detector to try and find new tracks to
  add

  Dealing with overlaps:
  - If another track is within X of this track, be careful!
  - Stop the CSRT trackers in both tracks and instead run detection.

  Track interface:
  - Track(initialBoundingBox)
  - update(frame)  // keeps track of whether the update succeeded
  - reInit(faceDetector, frame)
  - framesSinceLastSuccessfulUpdate or isDead
*/
class FaceTrack {
public:
  FaceTrack(int i, cv::Mat frame, const cv::Rect2d &boundingBox, int freq)
      : id{i}, framesSinceLastSuccessfulUpdate{0}, framesSinceInit{0},
        detectionHitCount{1}, detectionFrequency{freq}, roi{boundingBox},
        tracker{cv::TrackerCSRT::create()} {
    if (!tracker->init(frame, roi)) {
      throw std::runtime_error("Failed to init face track");
    };
  }

  void updateWithoutBoundingBox(cv::Mat frame) {
    framesSinceInit++;
    if (tracker->update(frame, roi)) {
      // std::cout << "Track " << id << " successful update\n";
      framesSinceLastSuccessfulUpdate = 0;
    } else {
      std::cout << "Track " << id << " failed to update\n";
      framesSinceLastSuccessfulUpdate++;
    }
  }

  void updateWithBoundingBox(cv::Mat frame, const cv::Rect2d &boundingBox) {
    roi = boundingBox;
    framesSinceLastSuccessfulUpdate = 0;
    // framesSinceInit = 0;
    detectionHitCount++;
    tracker = cv::TrackerCSRT::create();
    if (!tracker->init(frame, roi)) {
      throw std::runtime_error("Failed to init face track");
    }
  }

  const cv::Rect2d &getBoundingBox() const { return roi; }

  const int id;

  bool isExpired() const {
    // This says something like: keep track around for 1 more pass of the
    // detector
    return framesSinceLastSuccessfulUpdate > 0 ||
           framesSinceInit > ((detectionHitCount + 1) * detectionFrequency);
  }

private:
  int framesSinceLastSuccessfulUpdate;
  int framesSinceInit;
  int detectionHitCount;
  const int detectionFrequency;
  cv::Rect2d roi;
  cv::Ptr<cv::Tracker> tracker;
};

class FaceTrackManager {
public:
  FaceTrackManager(int freq)
      : trackCount{0}, detectionFrequency{freq}, framesSinceDetection{
                                                     detectionFrequency} {}

  void processFrame(FaceDetector &faceDetector, cv::Mat frame) {
    // Think about the case when tracks are close together
    if (framesSinceDetection >= detectionFrequency) {
      framesSinceDetection = 0;
      auto faces = faceDetector.detectFaces(frame);
      // Match
      auto matchResult = match(faces);
      // Update tracks with matched detections
      for (auto &[trackIdx, faceIdx] : matchResult.matches) {
        std::cout << "Matched track " << trackIdx << " with face " << faceIdx
                  << "\n";
        tracks.at(trackIdx).updateWithBoundingBox(frame, faces.at(faceIdx));
      }
      // Create new tracks
      for (auto &faceIdx : matchResult.unmatchedFaces) {
        int trackId = trackCount;
        trackCount++;
        std::cout << "Creating track: " << trackId
                  << " from face idx: " << faceIdx << "\n";
        FaceTrack newTrack{trackId, frame, faces[faceIdx], detectionFrequency};
        tracks.insert(std::make_pair(trackId, newTrack));
      }
      // Update unmatched tracks
      for (auto &trackIdx : matchResult.unmatchedTracks) {
        std::cout << "Updating unmatched track: " << trackIdx << "\n";
        tracks.at(trackIdx).updateWithoutBoundingBox(frame);
      }
    } else {
      framesSinceDetection++;
      // Update tracks
      for (auto &trackIter : tracks) {
        trackIter.second.updateWithoutBoundingBox(frame);
      }
    }
    // Remove old tracks
    for (auto it = tracks.cbegin(); it != tracks.cend();) {
      if (it->second.isExpired()) {
        std::cout << "Removing track " << it->second.id << "\n";
        tracks.erase(it++);
      } else {
        it++;
      }
    }
  }

  void drawTracks(cv::Mat frame) const {
    std::for_each(tracks.begin(), tracks.end(), [&frame](auto &track) {
      cv::rectangle(frame, track.second.getBoundingBox(), cv::Scalar(0, 255, 0),
                    BOX_THICKNESS);
    });
  }

private:
  MatchResult match(const std::vector<cv::Rect> &faces) {
    // XXX: This needs a proper cost matrix implementation
    std::vector<std::tuple<int, int>> matches;
    std::vector<int> unmatchedTracks;

    std::vector<bool> matchedFaces(faces.size());
    std::fill(matchedFaces.begin(), matchedFaces.end(), false);

    for (auto &trackIter : tracks) {
      bool trackWasMatched = false;
      for (auto i = 0; i < faces.size(); i++) {
        if (!matchedFaces.at(i) &&
            calculateIou(trackIter.second.getBoundingBox(), faces[i]) > 0) {
          matchedFaces[i] = true;
          trackWasMatched = true;
          matches.push_back(std::make_tuple(trackIter.first, i));
          break;
        }
      }
      if (!trackWasMatched) {
        unmatchedTracks.push_back(trackIter.first);
      }
    }

    std::vector<int> unmatchedFaces;
    for (int i = 0; i < matchedFaces.size(); i++) {
      if (!matchedFaces.at(i)) {
        unmatchedFaces.push_back(i);
      }
    }
    return {matches, unmatchedFaces, unmatchedTracks};
  }

  int trackCount;
  int detectionFrequency;
  int framesSinceDetection;
  std::unordered_map<int, FaceTrack> tracks;
};

int main(int argc, char **argv) {
  std::string videoFilepath{"/home/seb/Downloads/1601429081287000000.ts"};
  // std::string videoFilepath{"/home/seb/Downloads/output.mp4"};

  auto cap = cv::VideoCapture(videoFilepath);
  if (!cap.isOpened()) {
    std::cerr << "Failed to open video capture\n";
    return 1;
  }

  auto faceDetector = FaceDetector();
  auto trackManager = FaceTrackManager(20);
  // std::optional<FaceTrack> faceTrack{};
  // auto faceTracker = cv::TrackerCSRT::create();

  cv::Mat frame;
  while (true) {
    cap.read(frame);
    if (frame.empty()) {
      break;
    }
    cv::resize(frame, frame, cv::Size(), 1280.0 / frame.cols,
               720.0 / frame.rows);

    trackManager.processFrame(faceDetector, frame);
    trackManager.drawTracks(frame);

    // if (!faceTrack.has_value()) {
    //   auto faces = faceDetector.detectFaces(frame);

    //   if (faces.size() > 0) {
    //     faceTrack.emplace(0, frame, faces[0]);
    //   }
    // } else {
    //   faceTrack.value().updateWithoutBoundingBox(frame);
    // }

    // if (faceTrack.has_value()) {
    //   cv::rectangle(frame, faceTrack.value().getBoundingBox(),
    //                 cv::Scalar(0, 255, 0), BOX_THICKNESS);
    // }

    // for (auto &face : faces) {
    //   cv::rectangle(frame, {face.x, face.y},
    //                 {face.x + face.width, face.y + face.height},
    //                 cv::Scalar(0, 255, 0), BOX_THICKNESS);
    // }

    // if (faces.size() > 0 && !isTrackerInitialized) {
    //   faceTracker->init(frame, faces[0]);
    // }

    cv::imshow(WINDOW_NAME, frame);
    int k = cv::waitKey(25);
    if ((k & 0xff) == 'q') {
      break;
    }
  }
}
