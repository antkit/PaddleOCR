#pragma once

#include <include/paddleocr.h>
#include <include/paddlestructure.h>

struct Task {
  std::string id;
  std::string command;
  //std::string created_at;
  std::string content;
};

struct OcrTask {
  OcrTask();

  std::string lang;
  std::string image; // img first then region on screen
  std::vector<int> region; // screen region
  std::vector<std::string> actions; // (flip, grayscale, resize)
  bool det;
  bool rec;
  bool cls;
};

// locate images on screen or locate image(captured on screen) in images
struct LocateTask {
  LocateTask();

  std::vector<std::string> images;
  std::vector<int> region; // screen's region or images's region
  float confidence;

  std::vector<std::string> actions; // (flip, grayscale, resize)
  std::string mode; // images_on_screen | screen_in_images | images_in_image | image_in_images
  std::string mask;
  int method;

  std::string image;
};

struct PixelTask {
  int x;
  int y;
};

struct ScreenshotTask {
  std::vector<int> region; // screen region
  std::string path; // path to save screenshot
};

struct LocateResult {
  LocateResult();

  int located; // index of the image located
  cv::Rect region; // x, y, w, h
  double score; // maxLoc value, 0~1
};

class Worker {
public:
  Worker();
  ~Worker();

  void execute(const Task& task);

protected:
  std::shared_ptr<PaddleOCR::PPOCR> ppocr_by_lang(const std::string& lang);

  void do_execute(const std::string& id, const OcrTask& task);
  void do_execute(const std::string& id, const LocateTask& task);
  void do_execute(const std::string& id, const PixelTask& task);
  void do_execute(const std::string& id, const ScreenshotTask& task);

  std::pair<bool, std::shared_ptr<cv::Mat>> apply_actions(const std::string& id, const cv::Mat& image, const std::vector<std::string>& actions);

  std::shared_ptr<LocateResult> do_locate(const std::string& id, const cv::Mat& image, const std::vector<std::string>& images,
    const std::string& mask, const std::string& mode, int method, float confidence);

  void print_result(const std::string& id, bool success, const std::string& content);
  void print_result(const std::string& id, bool success, const std::vector<PaddleOCR::OCRPredictResult>& ocr_result);

private:
  std::map<std::string, std::shared_ptr<PaddleOCR::PPOCR>> ppocrs_;
  std::map<std::string, std::shared_ptr<PaddleOCR::PaddleStructure>> ps_engines_;

  static std::mutex output_mutex_;
};
