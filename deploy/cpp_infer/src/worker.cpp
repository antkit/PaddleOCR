#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <Windows.h>
#endif

#include <include/json.hpp>
#include <include/args.h>
#include <include/worker.h>

using json = nlohmann::json;
using namespace PaddleOCR;

void check_worker_params() {
  if (FLAGS_det_model_dir.empty()) {
    std::cerr << "missing flag det_model_dir" << std::endl;
    exit(1);
  }
  if (FLAGS_rec_model_dir.empty()) {
    std::cerr << "missing flag rec_model_dir" << std::endl;
    exit(1);
  }
  if (FLAGS_rec_char_dict_path.empty()) {
    std::cerr << "missing flag rec_char_dict_path" << std::endl;
    exit(1);
  }
}

void split(const std::string& s, std::vector<std::string>& sv, const char* delim = " ") {
  sv.clear();
  char* buffer = new char[s.size() + 1];
  buffer[s.size()] = '\0';
  std::copy(s.begin(), s.end(), buffer);
  char* p = std::strtok(buffer, delim);
  do {
    sv.push_back(p);
  } while ((p = std::strtok(NULL, delim)));
  delete[] buffer;
}

#ifdef _WIN32
cv::Mat captureScreenMat(int x, int y, int width, int height)
{
  // get handles to a device context (DC)
  HWND hwnd = GetDesktopWindow();
  HDC hwindowDC = GetDC(hwnd);
  HDC hwindowCompatibleDC = CreateCompatibleDC(hwindowDC);

  // define scale, height and width
  //int screenx = GetSystemMetrics(SM_XVIRTUALSCREEN);
  //int screeny = GetSystemMetrics(SM_YVIRTUALSCREEN);
  //int width = GetSystemMetrics(SM_CXVIRTUALSCREEN);
  //int height = GetSystemMetrics(SM_CYVIRTUALSCREEN);

  // create a bitmap
  HBITMAP hbwindow = CreateCompatibleBitmap(hwindowDC, width, height);
  BITMAPINFOHEADER bi;
  bi.biSize = sizeof(BITMAPINFOHEADER);
  bi.biWidth = width;
  bi.biHeight = -height;  //this is the line that makes it draw upside down or not
  bi.biPlanes = 1;
  bi.biBitCount = 32;
  bi.biCompression = BI_RGB;
  bi.biSizeImage = 0;
  bi.biXPelsPerMeter = 0;
  bi.biYPelsPerMeter = 0;
  bi.biClrUsed = 0;
  bi.biClrImportant = 0;

  // use the previously created device context with the bitmap
  SelectObject(hwindowCompatibleDC, hbwindow);

  // copy from the window device context to the bitmap device context
  //SetStretchBltMode(hwindowCompatibleDC, COLORONCOLOR);
  //StretchBlt(hwindowCompatibleDC, 0, 0, width, height, hwindowDC, x, y, width, height, SRCCOPY);  //change SRCCOPY to NOTSRCCOPY for wacky colors !
  BitBlt(hwindowCompatibleDC, 0, 0, width, height, hwindowDC, x, y, SRCCOPY);

  // create mat object
  cv::Mat mat(height, width, CV_8UC4);
  GetDIBits(hwindowCompatibleDC, hbwindow, 0, height, mat.data, (BITMAPINFO*)&bi, DIB_RGB_COLORS); //copy from hwindowCompatibleDC to hbwindow

  // avoid memory leak
  DeleteObject(hbwindow);
  DeleteDC(hwindowCompatibleDC);
  ReleaseDC(hwnd, hwindowDC);

  return mat;
}
#endif // _WIN32

OcrTask::OcrTask()
  : scale(1.0), det(true), rec(true), cls(false) {
}

void from_json(const json& j, Task& t) {
  j.at("id").get_to(t.id);
  j.at("command").get_to(t.command);
  j.at("content").get_to(t.content);
}

void from_json(const json& j, OcrTask& t) {
  j.at("lang").get_to(t.lang);
  j.at("region").get_to(t.region);
  j.at("scale").get_to(t.scale);
  j.at("det").get_to(t.det);
  j.at("rec").get_to(t.rec);
  j.at("cls").get_to(t.cls);
  if (j.contains("img")) {
    j.at("img").get_to(t.img);
  }
}

// FLAGS_* are used in:
//  - PPOCR::PPOCR, PPOCR::benchmark_log
//  - PaddleStructure::PaddleStructure, PaddleStructure::benchmark_log
Worker::Worker() {
  // construct default ppocr
  ppocrs_[FLAGS_lang] = std::shared_ptr<PPOCR>(new PPOCR());
}

Worker::~Worker() {
}

bool Worker::busy() {
  return false;
}

void Worker::execute(const Task& task) {
  auto j = json::parse(task.content, nullptr, false);
  if (j.is_discarded()) {
    std::cerr << "invalid task content" << std::endl;
    return;
  }
  std::cerr << "parsed content" << std::endl;
  if (task.command == "ocr") {
    auto ocr_task = j.template get<OcrTask>();
    do_execute(task.id, ocr_task);
  }
  else {
    std::cerr << "unknown task: " << task.command << std::endl;
  }
}

void Worker::do_execute(const std::string& id, const OcrTask& task) {
  if (ppocrs_.find(task.lang) == ppocrs_.end()) {
    ppocrs_[task.lang] = std::shared_ptr<PPOCR>(new PPOCR());
  }
  std::shared_ptr<PPOCR> ppocr = ppocrs_[task.lang];
  if (!ppocr) {
    ppocr = std::shared_ptr<PPOCR>(new PPOCR());
  }

  if (FLAGS_benchmark) {
    ppocr->reset_timer();
  }

  // TODO deal with w=0 and h=0
  cv::Mat img_with_alpha = captureScreenMat(task.region[0], task.region[1], task.region[2], task.region[3]);
  if (!img_with_alpha.data) {
    std::cerr << "[ERROR] screenshot invalid data" << std::endl;
    print_result(id, false, std::vector<OCRPredictResult>());
    return;
  }
  //cv::imwrite(FLAGS_output + "/screenshot.png", img_with_alpha);

  //std::cerr << "\t ocr running..." << std::endl;
  cv::Mat img = cv::Mat();
  if (std::abs(task.scale - 1.0) > 0.001) {
    auto new_size = cv::Size(int(img_with_alpha.cols * task.scale), int(img_with_alpha.rows * task.scale));
    cv::Mat scaled = cv::Mat();
    cv::resize(img_with_alpha, scaled, new_size);
    cv::cvtColor(scaled, img, cv::COLOR_BGRA2BGR);
  }
  else {
    cv::cvtColor(img_with_alpha, img, cv::COLOR_BGRA2BGR);
  }
  std::vector<OCRPredictResult> ocr_result = ppocr->ocr(img, FLAGS_det, FLAGS_rec, FLAGS_cls);
  //std::cerr << "\t ocr finished" << std::endl;

  std::unique_ptr<char[]> buf(new char[size_t(128)]);
  int size = std::snprintf(buf.get(), 128, "ss_%d_%d_%d_%d.png", task.region[0], task.region[1], task.region[2], task.region[3]);
  //std::cerr << "predict screenshot: " << buf.get() << std::endl;
  print_result(id, true, ocr_result);
  if (FLAGS_visualize && FLAGS_det) {
    Utility::VisualizeBboxes(img, ocr_result, FLAGS_output + "/" + buf.get());
  }
  if (FLAGS_benchmark) {
    ppocr->benchmark_log(1);
  }
}

void Worker::print_result(const std::string& id, bool success, const std::vector<PaddleOCR::OCRPredictResult>& ocr_result) {
  json ocr_texts = json::array();
  for (std::vector<OCRPredictResult>::const_iterator p = ocr_result.begin(); p != ocr_result.end(); p++) {
    // it can be many results with score 0
    if (p->score < 0.00001) {
      continue;
    }
    ocr_texts.push_back({
      {"bbox", p->box},
      {"text", p->text},
      {"confidence", p->score},
      });
  }
  json j = json();
  j["id"] = id;
  j["success"] = success;
  j["content"] = ocr_texts.dump();
  std::cout << j.dump() << std::endl;
}

int run_workers() {
  using namespace std::chrono_literals;

  // default ppocr
  FLAGS_det_model_dir = FLAGS_models_dir + "/ch_PP-OCRv4_det_infer";
  FLAGS_rec_model_dir = FLAGS_models_dir + "/ch_PP-OCRv4_rec_infer";
  FLAGS_rec_char_dict_path = FLAGS_models_dir + "/ppocr_keys_v1.txt";

  check_worker_params();

  if (!Utility::PathExists(FLAGS_output)) {
    Utility::CreateDir(FLAGS_output);
  }

  Worker worker;
  while (true) {
    std::string line;
    if (!std::getline(std::cin, line)) {
      //std::cerr << "FAILED to getline" << std::endl;
      std::this_thread::sleep_for(10ms);
      continue;
    }

    if (FLAGS_visualize) {
      std::cerr << "LINE: [" << line << "]" << std::endl;
    }
    auto j = json::parse(line, nullptr, false);
    if (j.is_discarded() || !j.contains("command")) {
      std::cerr << "unknown task" << std::endl;
      continue;
    }

    Task task;
    try {
      task = j.get<Task>();
    }
    catch (...) {
      std::cerr << "illegal task format" << std::endl;
      continue;
    }

    if (task.command == "DONE") {
      break;
    }
    worker.execute(task);
  }

  std::cerr << "quiting..." << std::endl;
  return 0;
}
