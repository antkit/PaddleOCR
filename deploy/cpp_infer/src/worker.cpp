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

LocateTask::LocateTask()
  : scale(1.0), confidence(0.999) {
}

void from_json(const json& j, Task& t) {
  j.at("id").get_to(t.id);
  j.at("command").get_to(t.command);
  j.at("content").get_to(t.content);
}

void from_json(const json& j, OcrTask& t) {
  j.at("lang").get_to(t.lang);
  j.at("region").get_to(t.region);
  if (j.contains("scale")) {
    j.at("scale").get_to(t.scale);
  }
  j.at("det").get_to(t.det);
  j.at("rec").get_to(t.rec);
  j.at("cls").get_to(t.cls);
  if (j.contains("img")) {
    j.at("img").get_to(t.img);
  }
}

void from_json(const json& j, LocateTask& t) {
  j.at("region").get_to(t.region);
  if (j.contains("scale")) {
    j.at("scale").get_to(t.scale);
  }
  if (j.contains("confidence")) {
    j.at("confidence").get_to(t.scale);
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
  //std::cerr << "parsed content" << std::endl;
  if (task.command == "ocr") {
    auto real_task = j.template get<OcrTask>();
    do_execute(task.id, real_task);
  }
  else if (task.command == "locate") {
    auto real_task = j.template get<LocateTask>();
    do_execute(task.id, real_task);
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
  if (std::abs(task.scale - 1.0) > 0.0001) {
    auto new_size = cv::Size(int(img_with_alpha.cols * task.scale), int(img_with_alpha.rows * task.scale));
    cv::Mat scaled = cv::Mat();
    cv::resize(img_with_alpha, scaled, new_size);
    cv::cvtColor(scaled, img, cv::COLOR_BGRA2BGR);
  }
  else {
    cv::cvtColor(img_with_alpha, img, cv::COLOR_BGRA2BGR);
  }
  std::vector<OCRPredictResult> ocr_result = ppocr->ocr(img, FLAGS_det, FLAGS_rec, FLAGS_cls);

  // revert scaling
  if (std::abs(task.scale - 1.0) > 0.0001) {
    for (std::vector<OCRPredictResult>::iterator p = ocr_result.begin(); p != ocr_result.end(); ++p) {
      for (std::vector<std::vector<int>>::iterator p2 = p->box.begin(); p2 != p->box.end(); ++p2) {
        std::transform(p2->cbegin(), p2->cend(), p2->begin(), [task](int v) {
          return int(std::round(v / task.scale));
        });
      }
    }
  }
  //std::cerr << "\t ocr finished" << std::endl;

  const std::vector<int>& rg = task.region;
  std::cerr << "predict screenshot: " << rg[0] << "," << rg[1] << "," << rg[2] << "," << rg[3] << std::endl;
  print_result(id, true, ocr_result);
  if (FLAGS_visualize && FLAGS_det) {
    std::unique_ptr<char[]> buf(new char[size_t(128)]);
    int size = std::snprintf(buf.get(), 128, "ss_%d_%d_%d_%d.png", rg[0], rg[1], rg[2], rg[3]);
    Utility::VisualizeBboxes(img_with_alpha, ocr_result, FLAGS_output + "/" + buf.get());
  }
  if (FLAGS_benchmark) {
    ppocr->benchmark_log(1);
  }
}

void Worker::do_execute(const std::string& id, const LocateTask& task) {
  // TODO deal with w=0 and h=0
  cv::Mat img_with_alpha = captureScreenMat(task.region[0], task.region[1], task.region[2], task.region[3]);
  if (!img_with_alpha.data) {
    std::cerr << "[ERROR] screenshot invalid data" << std::endl;
    json j = json();
    j["id"] = id;
    j["success"] = false;
    j["content"] = "{}";
    std::cout << j.dump() << std::endl;
    return;
  }

  cv::Mat img = cv::Mat();
  if (std::abs(task.scale - 1.0) > 0.0001) {
    auto new_size = cv::Size(int(img_with_alpha.cols * task.scale), int(img_with_alpha.rows * task.scale));
    cv::Mat scaled = cv::Mat();
    cv::resize(img_with_alpha, scaled, new_size);
    cv::cvtColor(scaled, img, cv::COLOR_BGRA2BGR);
  }
  else {
    cv::cvtColor(img_with_alpha, img, cv::COLOR_BGRA2BGR);
  }

  for (std::vector<std::string>::const_iterator p = task.images.begin(); p != task.images.end(); ++p) {
    auto templ = cv::imread(*p, cv::IMREAD_COLOR); // cv::IMREAD_GRAYSCALE
    auto match_method = cv::TemplateMatchModes::TM_CCORR_NORMED;
    cv::Mat result(cv::Size(img.cols - templ.cols + 1, img.rows - templ.rows + 1), CV_32FC1);
    cv::matchTemplate(img, templ, result, match_method);

    normalize(result, result, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());
    double minVal;
    double maxVal;
    cv::Point minLoc;
    cv::Point maxLoc;
    cv::Point matchLoc;
    cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, cv::Mat());
    if (match_method == cv::TemplateMatchModes::TM_SQDIFF || match_method == cv::TemplateMatchModes::TM_SQDIFF_NORMED) {
      matchLoc = minLoc;
    }
    else {
      matchLoc = maxLoc;
    }
    //rectangle(img_display, matchLoc, Point(matchLoc.x + templ.cols, matchLoc.y + templ.rows), Scalar::all(0), 2, 8, 0);
    //rectangle(result, matchLoc, Point(matchLoc.x + templ.cols, matchLoc.y + templ.rows), Scalar::all(0), 2, 8, 0);
    //imshow(image_window, img_display);
    //imshow(result_window, result);
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

class WorkerManager {
public:
  WorkerManager() : stop_(false) {}

  void add_task(const Task& task) {
    std::lock_guard<std::mutex> scoped_lock(task_locker_);
    std::shared_ptr<Task> new_task(new Task(task));
    task_deque_.push_back(new_task);
  }

  void run() {
    std::cout << "WorkerManager::run" << std::endl;
    stop_ = false;
    for (int i = 0; i < 3; i++) {
      std::thread t(&WorkerManager::do_run, this);
      t.detach();
    }
  }

  void stop() {
    std::lock_guard<std::mutex> scoped_lock(task_locker_);
    stop_ = true;
  }

protected:
  void do_run() {
    using namespace std::chrono_literals;
    std::cout << "WorkerManager::do_run" << std::endl;

    Worker worker;
    while (!stop_) {
      //std::cout << "WorkerManager::do_run checking task deque" << std::endl;
      std::shared_ptr<Task> task;
      {
        std::lock_guard<std::mutex> scoped_lock(task_locker_);
        if (!task_deque_.empty()) {
          task = task_deque_.front();
          task_deque_.pop_front();
        }
      }
      if (task) {
        worker.execute(*task);
      }
      else {
        std::this_thread::sleep_for(10ms);
      }
    }
  }

private:
  std::deque<std::shared_ptr<Task>> task_deque_;
  std::mutex task_locker_;
  volatile bool stop_;
};

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

  WorkerManager wm;
  wm.run();

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
      wm.stop();
      break;
    }
    wm.add_task(task);
  }

  //Worker worker;
  //while (true) {
  //  std::string line;
  //  if (!std::getline(std::cin, line)) {
  //    //std::cerr << "FAILED to getline" << std::endl;
  //    std::this_thread::sleep_for(10ms);
  //    continue;
  //  }

  //  if (FLAGS_visualize) {
  //    std::cerr << "LINE: [" << line << "]" << std::endl;
  //  }
  //  auto j = json::parse(line, nullptr, false);
  //  if (j.is_discarded() || !j.contains("command")) {
  //    std::cerr << "unknown task" << std::endl;
  //    continue;
  //  }

  //  Task task;
  //  try {
  //    task = j.get<Task>();
  //  }
  //  catch (...) {
  //    std::cerr << "illegal task format" << std::endl;
  //    continue;
  //  }

  //  if (task.command == "DONE") {
  //    break;
  //  }
  //  worker.execute(task);
  //}

  std::cerr << "quiting..." << std::endl;
  return 0;
}
