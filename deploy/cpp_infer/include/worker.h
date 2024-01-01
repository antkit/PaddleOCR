#pragma once

#include <include/paddleocr.h>
#include <include/paddlestructure.h>

struct Task {
  std::string id;
  std::string command;
  std::string created_at;
  std::string content;
};

struct OcrTask {
  OcrTask();

  std::string lang;
  std::vector<int> region; // screen region
  float scale;
  bool det;
  bool rec;
  bool cls;
  std::string img;
};

class Worker {
public:
  Worker();
  ~Worker();

  bool busy();
  void execute(const Task& task);
  //void ocr(const std::string& task);

protected:
  void do_execute(const std::string& id, const OcrTask& task);

  //void do_ocr(std::shared_ptr<PaddleOCR::PPOCR> ppocr, int x, int y, int w, int h);
  //void do_ocr(std::shared_ptr<PaddleOCR::PPOCR> ppocr, std::vector<cv::String>& cv_all_img_names);
  //void do_structure(std::shared_ptr<PaddleOCR::PaddleStructure> engine, std::vector<cv::String>& cv_all_img_names);

  void print_result(const std::vector<PaddleOCR::OCRPredictResult>& ocr_result);
  void print_result(const std::string& id, const std::vector<PaddleOCR::OCRPredictResult>& ocr_result);

private:
  std::map<std::string, std::shared_ptr<PaddleOCR::PPOCR>> ppocrs_;
  std::map<std::string, std::shared_ptr<PaddleOCR::PaddleStructure>> ps_engines_;
};
