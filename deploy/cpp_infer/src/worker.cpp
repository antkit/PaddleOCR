#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <codecvt>
#include <iostream>
#include <locale>
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

struct PaddleResource {
  PaddleResource() {}
  PaddleResource(const std::string& m, const std::string& r, const std::string& d)
    : det_model(m), rec_model(r), rec_char_dict(d) {
  }

  std::string det_model;
  std::string rec_model;
  std::string rec_char_dict;
};

class ResourceManager {
public:
  static ResourceManager& instance() {
    // we do not need thread safe here
    static ResourceManager res_instance;
    return res_instance;
  }

  bool contains_ppocr_lang(const std::string& lang) {
    return resources_.find(lang) != resources_.end();
  }

  std::string det_model(const std::string& lang) {
    auto iter = resources_.find(lang);
    return iter != resources_.end() ? iter->second.det_model : "";
  }

  std::string rec_model(const std::string& lang) {
    auto iter = resources_.find(lang);
    return iter != resources_.end() ? iter->second.rec_model : "";
  }

  std::string rec_char_dict(const std::string& lang) {
    auto iter = resources_.find(lang);
    return iter != resources_.end() ? iter->second.rec_char_dict : "";
  }

protected:
  ResourceManager() {
    // See paddleocr.py
    resources_["ch"] = PaddleResource("models/ch_PP-OCRv4_det_infer", "models/ch_PP-OCRv4_rec_infer", "dicts/ppocr_keys_v1.txt");
    resources_["en"] = PaddleResource("models/en_PP-OCRv3_det_infer", "models/en_PP-OCRv4_rec_infer", "dicts/en_dict.txt");
    resources_["korean"] = PaddleResource("models/Multilingual_PP-OCRv3_det_infer", "models/korean_PP-OCRv4_rec_infer", "dicts/korean_dict.txt");
    resources_["japan"] = PaddleResource("models/Multilingual_PP-OCRv3_det_infer", "models/japan_PP-OCRv4_rec_infer", "dicts/japan_dict.txt");
    resources_["chinese_cht"] = PaddleResource("models/Multilingual_PP-OCRv3_det_infer", "models/chinese_cht_PP-OCRv3_rec_infer", "dicts/chinese_cht_dict.txt");
    resources_["te"] = PaddleResource("models/Multilingual_PP-OCRv3_det_infer", "models/te_PP-OCRv4_rec_infer", "dicts/te_dict.txt");
    resources_["ka"] = PaddleResource("models/Multilingual_PP-OCRv3_det_infer", "models/ka_PP-OCRv4_rec_infer", "dicts/ka_dict.txt");
    resources_["latin"] = PaddleResource("models/Multilingual_PP-OCRv3_det_infer", "models/latin_PP-OCRv3_rec_infer", "dicts/latin_dict.txt");
    resources_["arabic"] = PaddleResource("models/Multilingual_PP-OCRv3_det_infer", "models/arabic_PP-OCRv4_rec_infer", "dicts/ar_dict.txt");
    resources_["cyrillic"] = PaddleResource("models/Multilingual_PP-OCRv3_det_infer", "models/cyrillic_PP-OCRv3_rec_infer", "dicts/cyrillic_dict.txt");
    resources_["devanagari"] = PaddleResource("models/Multilingual_PP-OCRv3_det_infer", "models/devanagari_PP-OCRv4_rec_infer", "dicts/devanagari_dict.txt");
  }

private:
  std::map<std::string, PaddleResource> resources_;
};

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

bool read_image(const std::string& filepath, cv::Mat& dst) {
  std::wstring_convert<std::codecvt_utf8<wchar_t>> conv;
  std::wstring path = conv.from_bytes(filepath.c_str());
  std::ifstream file(path, std::iostream::binary);
  if (!file) {
    return false;
  }

  size_t length = file.rdbuf()->pubseekoff(0, file.end, file.in);
  file.rdbuf()->pubseekpos(0, file.in);
  std::vector<uchar> buf(length);
  file.rdbuf()->sgetn((char*)buf.data(), length);
  cv::imdecode(buf, cv::IMREAD_COLOR, &dst);
  return true;
}

bool write_image(const std::string& filepath, const cv::Mat& src) {
  std::wstring_convert<std::codecvt_utf8<wchar_t>> conv;
  std::wstring path = conv.from_bytes(filepath.c_str());
  std::ofstream file(path, std::iostream::binary);
  if (!file) {
    return false;
  }

  std::vector<uchar> buf;
  cv::imencode(".png", src, buf);
  file.write((char*)buf.data(), buf.size());
  return true;
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
  : scale(1.0), /*grayscale(false), */confidence(0) {
}

void from_json(const json& j, Task& t) {
  j.at("id").get_to(t.id);
  j.at("command").get_to(t.command);
  if (j.contains("content")) {
    j.at("content").get_to(t.content);
  }
}

void from_json(const json& j, OcrTask& t) {
  j.at("lang").get_to(t.lang);
  if (j.contains("img")) {
    j.at("img").get_to(t.img);
  }
  j.at("region").get_to(t.region);
  if (j.contains("scale")) {
    j.at("scale").get_to(t.scale);
  }
  if (j.contains("det")) {
    j.at("det").get_to(t.det);
  }
  if (j.contains("rec")) {
    j.at("rec").get_to(t.rec);
  }
  if (j.contains("cls")) {
    j.at("cls").get_to(t.cls);
  }
}

void from_json(const json& j, LocateTask& t) {
  j.at("images").get_to(t.images);
  j.at("region").get_to(t.region);
  if (j.contains("scale")) {
    j.at("scale").get_to(t.scale);
  }
  //if (j.contains("grayscale")) {
  //  j.at("grayscale").get_to(t.grayscale);
  //}
  if (j.contains("confidence")) {
    j.at("confidence").get_to(t.confidence);
  }
  if (j.contains("actions")) {
    j.at("actions").get_to(t.actions);
  }
  if (j.contains("mode")) {
    j.at("mode").get_to(t.mode);
  }
  if (j.contains("mask")) {
    j.at("mask").get_to(t.mask);
  }
}

void from_json(const json& j, PixelTask& t) {
  j.at("x").get_to(t.x);
  j.at("y").get_to(t.y);
}

void from_json(const json& j, ScreenshotTask& t) {
  j.at("region").get_to(t.region);
  j.at("path").get_to(t.path);
}

std::mutex Worker::output_mutex_;

// FLAGS_* are used in:
//  - PPOCR::PPOCR, PPOCR::benchmark_log
//  - PaddleStructure::PaddleStructure, PaddleStructure::benchmark_log
Worker::Worker() {
  // construct default ppocr
  std::cerr << "constructing default ppocr:" << std::endl;
  std::cerr << "FLAGS_det_model_dir: " << FLAGS_det_model_dir << std::endl;
  std::cerr << "FLAGS_rec_model_dir: " << FLAGS_rec_model_dir << std::endl;
  std::cerr << "FLAGS_rec_char_dict_path: " << FLAGS_rec_char_dict_path << std::endl;
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
    print_result(task.id, false, "invalid task content");
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
  else if (task.command == "pixel") {
    auto real_task = j.template get<PixelTask>();
    do_execute(task.id, real_task);
  }
  else if (task.command == "screenshot") {
    auto real_task = j.template get<ScreenshotTask>();
    do_execute(task.id, real_task);
  }
  else {
    std::cerr << "unknown task: " << task.command << std::endl;
  }
}

std::shared_ptr<PPOCR> Worker::ppocr_by_lang(const std::string& lang) {
  if (!ResourceManager::instance().contains_ppocr_lang(lang)) {
    // return the default one
    return ppocrs_[FLAGS_lang];
  }
  auto iter = ppocrs_.find(lang);
  if (iter != ppocrs_.end()) {
    return iter->second;
  }
  auto det_model = FLAGS_data_dir + "/" + ResourceManager::instance().det_model(lang);
  auto rec_model = FLAGS_data_dir + "/" + ResourceManager::instance().rec_model(lang);
  auto rec_char_dict = FLAGS_data_dir + "/" + ResourceManager::instance().rec_char_dict(lang);
  ppocrs_[lang] = std::shared_ptr<PPOCR>(new PPOCR(det_model, rec_model, rec_char_dict));
  return ppocrs_[lang];
}

void Worker::do_execute(const std::string& id, const OcrTask& task) {
  //if (ppocrs_.find(task.lang) == ppocrs_.end()) {
  //  // TODO initialize PPOCR with task.lang
  //  ppocrs_[task.lang] = std::shared_ptr<PPOCR>(new PPOCR());
  //}
  std::shared_ptr<PPOCR> ppocr = ppocr_by_lang(task.lang);

  if (FLAGS_benchmark) {
    ppocr->reset_timer();
  }

  // TODO deal with w=0 and h=0
  cv::Mat img_with_alpha = captureScreenMat(task.region[0], task.region[1], task.region[2], task.region[3]);
  if (!img_with_alpha.data) {
    print_result(id, false, "captured screen without data");
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
  std::cerr << "predict screen: " << rg[0] << "," << rg[1] << "," << rg[2] << "," << rg[3] << std::endl;
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

struct LocateAction {
  std::string action;
  std::string params;
};

struct LocateResult {
  int located; // index of the image located
  std::vector<int> region; // result region
  float score; // maxLoc value, 0~1
};

void Worker::do_execute(const std::string& id, const LocateTask& task) {
  // TODO support task.grayscale
  std::cerr << "executing locate task" << std::endl;

  // TODO deal with w=0 and h=0
  cv::Mat captured_with_alpha = captureScreenMat(task.region[0], task.region[1], task.region[2], task.region[3]);
  if (!captured_with_alpha.data) {
    print_result(id, false, "captured screen without data");
    return;
  }

  bool with_scale = std::abs(task.scale - 1.0) > 0.0001;
  cv::Mat captured = cv::Mat();
  if (with_scale) {
    auto new_size = cv::Size(int(captured_with_alpha.cols * task.scale), int(captured_with_alpha.rows * task.scale));
    cv::Mat scaled = cv::Mat();
    cv::resize(captured_with_alpha, scaled, new_size);
    cv::cvtColor(scaled, captured, cv::COLOR_BGRA2BGR);
  }
  else {
    cv::cvtColor(captured_with_alpha, captured, cv::COLOR_BGRA2BGR);
  }

  // TODO support flip
  bool locate_on_screen = task.mode.empty() || task.mode == "screen";
  std::cerr << "locate_on_screen: " << locate_on_screen << std::endl;

  // TODO use multithread for multiple images locating?
  for (size_t i = 0; i < task.images.size(); i++) {
    cv::Mat image;
    if (!read_image(task.images[i], image)) {
      print_result(id, false, "can't load the image");
      return;
    }

    // TODO load task.mask and use it
    // TM_CCOEFF_NORMED's range -1~1, and the best matching is maxLoc
    auto match_method = cv::TemplateMatchModes::TM_CCOEFF_NORMED; // TM_CCORR_NORMED;
    cv::Mat result;
    if (locate_on_screen) {
      cv::matchTemplate(captured, image, result, match_method);
    }
    else {
      cv::matchTemplate(image, captured, result, match_method);
    }
    //std::cerr << "matchTemplate finished " << std::endl;

    double minVal, maxVal;
    cv::Point minLoc, maxLoc;
    cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, cv::Mat());

    // (maxVal + 1) / 2 transform -1~1 to 0~1
    if ((maxVal + 1) / 2 >= task.confidence) {
      json result = {
        int(i),
        locate_on_screen ? (task.region[0] + (with_scale ? int(maxLoc.x / task.scale) : maxLoc.x)) : maxLoc.x,
        locate_on_screen ? (task.region[1] + (with_scale ? int(maxLoc.y / task.scale) : maxLoc.y)) : maxLoc.y,
        locate_on_screen ? (with_scale ? int(image.cols / task.scale) : image.cols) : captured.cols,
        locate_on_screen ? (with_scale ? int(image.rows / task.scale) : image.rows) : captured.rows
      };
      print_result(id, true, result.dump());
      return;
    }
  }

  //for (size_t i = 0; i < task.images.size(); i++) {
  //  cv::Mat templ;
  //  if (!read_image(task.images[i], templ)) {
  //    print_result(id, false, "can't load the image");
  //    return;
  //  }

  //  // TM_CCOEFF_NORMED's range(-1, 1), and the best matching is maxLoc
  //  auto match_method = cv::TemplateMatchModes::TM_CCOEFF_NORMED; // TM_CCORR_NORMED;
  //  cv::Mat result;// (cv::Size(captured.cols - templ.cols + 1, captured.rows - templ.rows + 1), CV_32FC1);
  //  cv::matchTemplate(captured, templ, result, match_method);
  //  //std::cerr << "matchTemplate finished " << std::endl;

  //  double minVal, maxVal;
  //  cv::Point minLoc, maxLoc;
  //  cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, cv::Mat());
  //  //std::cerr << "minVal: " << minVal << ", maxVal: " << maxVal << ", required: " << task.confidence << std::endl;

  //  //cv::rectangle(captured_with_alpha, maxLoc, cv::Point(maxLoc.x + templ.cols, maxLoc.y + templ.rows), cv::Scalar::all(0), 2, 8, 0);
  //  //cv::rectangle(captured, maxLoc, cv::Point(maxLoc.x + templ.cols, maxLoc.y + templ.rows), cv::Scalar::all(255), 2, 8, 0);
  //  //cv::rectangle(result, maxLoc, cv::Point(maxLoc.x + templ.cols, maxLoc.y + templ.rows), cv::Scalar::all(0), 2, 8, 0);
  //  //cv::imwrite(FLAGS_output + "/locate_screen.png", img_with_alpha);
  //  //cv::imwrite(FLAGS_output + "/locate_screen_scaled.png", captured);
  //  //cv::imwrite(FLAGS_output + "/locate_result.png", result);

  //  if ((maxVal + 1) / 2 >= task.confidence) {
  //    json result = {
  //      int(i),
  //      task.region[0] + (with_scale ? int(maxLoc.x / task.scale) : maxLoc.x),
  //      task.region[1] + (with_scale ? int(maxLoc.y / task.scale) : maxLoc.y),
  //      with_scale ? int(templ.cols / task.scale) : templ.cols,
  //      with_scale ? int(templ.rows / task.scale) : templ.rows
  //    };
  //    print_result(id, true, result.dump());
  //    return;
  //  }
  //}
  print_result(id, false, "locate failed");
}

void Worker::do_execute(const std::string& id, const PixelTask& task) {
  cv::Mat img_with_alpha = captureScreenMat(task.x, task.y, 1, 1);
  if (!img_with_alpha.data) {
    print_result(id, false, "captured screen without data");
    return;
  }
  auto bgra = img_with_alpha.at<cv::Vec4b>(0, 0);
  json result = { bgra[2], bgra[1], bgra[0], bgra[3] }; // to rgba
  print_result(id, true, result.dump());
}

void Worker::do_execute(const std::string& id, const ScreenshotTask& task) {
  cv::Mat img_with_alpha = captureScreenMat(task.region[0], task.region[1], task.region[2], task.region[3]);
  if (!img_with_alpha.data) {
    print_result(id, false, "captured screen without data");
    return;
  }
  //cv::imwrite(task.path, img_with_alpha);
  if (!write_image(task.path, img_with_alpha)) {
    print_result(id, false, "write image failed");
    return;
  }
  print_result(id, true, "{}");
}

void Worker::print_result(const std::string& id, bool success, const std::string& content) {
  json j = json();
  j["id"] = id;
  j["success"] = success;
  j["content"] = content;

  std::ostringstream stream;
  stream << '\n' << j.dump(); // enforce start a new line

  std::lock_guard<std::mutex> lock(output_mutex_);
  std::cout << stream.str() << std::endl; // flush with endl
}

void Worker::print_result(const std::string& id, bool success, const std::vector<PaddleOCR::OCRPredictResult>& ocr_result) {
  json ocr_texts = json::array();
  for (std::vector<OCRPredictResult>::const_iterator p = ocr_result.begin(); p != ocr_result.end(); p++) {
    // it can be many results with score 0
    if (p->score < 0.00001) {
      //std::cerr << "score: " << p->score << ", text: " << p->text << std::endl;
      continue;
    }
    ocr_texts.push_back({
      {"bbox", p->box},
      {"text", p->text},
      {"confidence", p->score},
      });
  }
  print_result(id, success, ocr_texts.dump());
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
    std::cerr << "Runing with " << FLAGS_workers_num << " workers" << std::endl;
    stop_ = false;
    for (int i = 0; i < FLAGS_workers_num; i++) {
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
    //std::cerr << "WorkerManager::do_run" << std::endl;

    Worker worker;
    while (!stop_) {
      //std::cerr << "WorkerManager::do_run checking task deque" << std::endl;
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

  if (!ResourceManager::instance().contains_ppocr_lang(FLAGS_lang)) {
    std::cerr << "unsupported lang: " << FLAGS_lang << std::endl;
    exit(1);
  }
  FLAGS_det_model_dir = FLAGS_data_dir + "/" + ResourceManager::instance().det_model(FLAGS_lang);
  FLAGS_rec_model_dir = FLAGS_data_dir + "/" + ResourceManager::instance().rec_model(FLAGS_lang);
  FLAGS_rec_char_dict_path = FLAGS_data_dir + "/" + ResourceManager::instance().rec_char_dict(FLAGS_lang);

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
      std::cerr << "LINE: " << line << std::endl;
    }
    auto j = json::parse(line, nullptr, false);
    if (j.is_discarded() || !j.contains("command")) {
      std::cerr << "unknown task: " << j.dump() << std::endl;
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

  std::cerr << "quiting..." << std::endl;
  return 0;
}
