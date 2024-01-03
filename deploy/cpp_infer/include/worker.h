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

struct LocateTask {
  LocateTask();

  std::vector<std::string> images;
  std::vector<int> region; // screen region
  float scale; // scale the image on screen region
  float confidence;
};

class Worker {
public:
  Worker();
  ~Worker();

  bool busy();
  void execute(const Task& task);

protected:
  void do_execute(const std::string& id, const OcrTask& task);
  void do_execute(const std::string& id, const LocateTask& task);

  void print_result(const std::string& id, bool success, const std::vector<PaddleOCR::OCRPredictResult>& ocr_result);

private:
  std::map<std::string, std::shared_ptr<PaddleOCR::PPOCR>> ppocrs_;
  std::map<std::string, std::shared_ptr<PaddleOCR::PaddleStructure>> ps_engines_;
};
