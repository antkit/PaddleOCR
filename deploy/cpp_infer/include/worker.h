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
  std::string img; // img first then region on screen
  std::vector<int> region; // screen region
  float scale;
  bool det;
  bool rec;
  bool cls;
};

struct LocateTask {
  LocateTask();

  std::vector<std::string> images;
  std::vector<int> region; // screen region
  float confidence;

  std::vector<std::string> actions; // (flip, grayscale,)
  std::string mode; // screen | images, locate on/in
  std::string mask;
};

struct PixelTask {
  int x;
  int y;
};

struct ScreenshotTask {
  std::vector<int> region; // screen region
  std::string path; // path to save screenshot
};

class Worker {
public:
  Worker();
  ~Worker();

  bool busy();
  void execute(const Task& task);

protected:
  std::shared_ptr<PaddleOCR::PPOCR> ppocr_by_lang(const std::string& lang);

  void do_execute(const std::string& id, const OcrTask& task);
  void do_execute(const std::string& id, const LocateTask& task);
  void do_execute(const std::string& id, const PixelTask& task);
  void do_execute(const std::string& id, const ScreenshotTask& task);

  void print_result(const std::string& id, bool success, const std::string& content);
  void print_result(const std::string& id, bool success, const std::vector<PaddleOCR::OCRPredictResult>& ocr_result);

private:
  std::map<std::string, std::shared_ptr<PaddleOCR::PPOCR>> ppocrs_;
  std::map<std::string, std::shared_ptr<PaddleOCR::PaddleStructure>> ps_engines_;

  static std::mutex output_mutex_;
};
