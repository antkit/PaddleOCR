#pragma once

#include <include/paddleocr.h>
#include <include/paddlestructure.h>

class Worker {
public:
  Worker();
  ~Worker();

  bool busy();
  void execute(const std::string& task);

protected:
  void do_ocr(std::shared_ptr<PaddleOCR::PPOCR> ppocr, int x, int y, int w, int h);
  void do_ocr(std::shared_ptr<PaddleOCR::PPOCR> ppocr, std::vector<cv::String>& cv_all_img_names);
  void do_structure(std::shared_ptr<PaddleOCR::PaddleStructure> engine, std::vector<cv::String>& cv_all_img_names);

private:
  std::map<std::string, std::shared_ptr<PaddleOCR::PPOCR>> ppocrs_;
  std::map<std::string, std::shared_ptr<PaddleOCR::PaddleStructure>> ps_engines_;
};
