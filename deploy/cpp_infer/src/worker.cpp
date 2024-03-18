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

static const char* AUTO_MASK = "auto";

struct Action {
  std::string action;
  std::string params;
};

static void from_json(const json& j, Action& a) {
  j.at("action").get_to(a.action);
  if (j.contains("params")) {
    j.at("params").get_to(a.params);
  }
}

// if factor > 0 then use factor to do resizing,
// else use width and height to do resizing
struct ActionResizeParams {
  float factor;
  int width;
  int height;
};

static void from_json(const json& j, ActionResizeParams& a) {
  if (j.contains("factor")) {
    j.at("factor").get_to(a.factor);
  }
  if (j.contains("width")) {
    j.at("width").get_to(a.width);
  }
  if (j.contains("height")) {
    j.at("height").get_to(a.height);
  }
}

// https://docs.opencv.org/4.x/d2/de8/group__core__array.html#gaca7be533e3dac7feb70fc60635adf441
struct ActionFlipParams {
  int code; // 0 - flipping around the x-axis; Positive value - flipping around y-axis; Negative value - flipping around both axes.
};

static void from_json(const json& j, ActionFlipParams& a) {
  if (j.contains("code")) {
    j.at("code").get_to(a.code);
  }
}

struct ActionCropParams {
  std::vector<int> region;
};

static void from_json(const json& j, ActionCropParams& a) {
  j.at("region").get_to(a.region);
}

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

static void check_worker_params() {
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

static bool read_image(const std::string& filepath, cv::Mat& dst, cv::ImreadModes mode=cv::IMREAD_COLOR) {
  // cv::imread has problem if filepath is in unicode, so use std fstream to load the image file
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
  cv::imdecode(buf, mode, &dst);
  return true;
}

static bool write_image(const std::string& filepath, const cv::Mat& src) {
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

static void split(const std::string& s, std::vector<std::string>& sv, const char* delim = " ") {
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
static cv::Mat captureScreenMat(int x, int y, int width, int height)
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

static void from_json(const json& j, Task& t) {
  j.at("id").get_to(t.id);
  j.at("command").get_to(t.command);
  if (j.contains("content")) {
    j.at("content").get_to(t.content);
  }
}

OcrTask::OcrTask()
  : det(true), rec(true), cls(false) {
}

static void from_json(const json& j, OcrTask& t) {
  j.at("lang").get_to(t.lang);
  if (j.contains("image")) {
    j.at("image").get_to(t.image);
  }
  if (j.contains("region")) {
    j.at("region").get_to(t.region);
  }
  if (j.contains("actions")) {
    j.at("actions").get_to(t.actions);
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

LocateTask::LocateTask()
  : confidence(0), method(5) {
}

static void from_json(const json& j, LocateTask& t) {
  j.at("images").get_to(t.images);
  if (j.contains("region")) {
    j.at("region").get_to(t.region);
  }
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
  if (j.contains("method")) {
    j.at("method").get_to(t.method);
  }
  if (j.contains("image")) {
    j.at("image").get_to(t.image);
  }
}

static void from_json(const json& j, PixelTask& t) {
  j.at("x").get_to(t.x);
  j.at("y").get_to(t.y);
}

static void from_json(const json& j, ScreenshotTask& t) {
  j.at("region").get_to(t.region);
  j.at("path").get_to(t.path);
}

LocateResult::LocateResult()
  : located(-1), score(0) {
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

std::pair<bool, std::shared_ptr<cv::Mat>> Worker::apply_actions(const std::string& id, const cv::Mat& image, const std::vector<std::string>& actions) {
  std::unique_ptr<cv::Mat> result_image;
  for (auto p = actions.begin(); p != actions.end(); ++p) {
    std::cerr << "action:" << *p << std::endl;
    auto j = json::parse(*p, nullptr, false);
    if (j.is_discarded()) {
      print_result(id, false, "invalid action content");
      return std::make_pair(false, nullptr);
    }
    auto action = j.template get<Action>();
    auto j2 = json::parse(action.params, nullptr, false);
    if (j2.is_discarded()) {
      print_result(id, false, "invalid action params");
      return std::make_pair(false, nullptr);
    }

    const cv::Mat& target_image = result_image ? *result_image : image;
    if (action.action == "resize") {
      auto size = std::make_unique<cv::Size>(target_image.cols, target_image.rows);
      auto ap = j2.template get<ActionResizeParams>();
      if (ap.factor > 0) {
        if (std::abs(ap.factor - 1.0) > 0.00001) {
          size->width = int(target_image.cols * ap.factor);
          size->height = int(target_image.rows * ap.factor);
        }
      }
      else {
        if (ap.width <= 0 || ap.height <= 0) {
          print_result(id, false, "invalid resize params");
          return std::make_pair(false, nullptr);
        }
        if (ap.width != target_image.cols || ap.height != target_image.rows) {
          size->width = ap.width;
          size->height = ap.height;
        }
      }

      if (size->width != target_image.cols || size->height != target_image.rows) {
        std::unique_ptr<cv::Mat> mat_output(new cv::Mat());
        cv::resize(target_image, *mat_output, *size);
        result_image = std::move(mat_output);
      }
    }
    else if (action.action == "flip") {
      auto ap = j2.template get<ActionFlipParams>();
      std::unique_ptr<cv::Mat> mat_output(new cv::Mat());
      cv::flip(target_image, *mat_output, ap.code);
      result_image = std::move(mat_output);
    }
    //else if (action.action == "crop") {
    //  auto ap = j2.template get<ActionCropParams>();
    //  if (ap.region.size() < 4 || ap.region[0] < 0 || ap.region[1] < 0 || ap.region[2] < 0 || ap.region[3] < 0) {
    //    print_result(id, false, "invalid crop params");
    //    return std::make_pair(false, nullptr);
    //  }
    //  if (ap.region[0] + ap.region[2] > target_image.cols || ap.region[1] + ap.region[3] > target_image.rows) {
    //    print_result(id, false, "crop region exceeded");
    //    return std::make_pair(false, nullptr);
    //  }
    //  cv::Rect crop(ap.region[0], ap.region[1], ap.region[2], ap.region[3]);
    //  if (!result_image) {
    //    result_image = std::unique_ptr<cv::Mat>(new cv::Mat());
    //  }
    //  *result_image = target_image(crop);
    //}
    else if (action.action == "grayscale") {
      // TODO support grayscale
      print_result(id, false, "not implemented grayscale");
      return std::make_pair(false, nullptr);
    }
  }
  return std::make_pair(true, std::move(result_image));
}

void Worker::do_execute(const std::string& id, const OcrTask& task) {
  std::shared_ptr<PPOCR> ppocr = ppocr_by_lang(task.lang);
  if (FLAGS_benchmark) {
    ppocr->reset_timer();
  }

  std::shared_ptr<cv::Mat> image(new cv::Mat());
  if (!task.image.empty()) {
    if (!read_image(task.image, *image)) {
      std::cerr << "[ERROR] can't load the image: " << task.image << std::endl;
      print_result(id, false, "failed load image");
      return;
    }
    if (!task.region.empty()) {
      if (task.region.size() != 4 || task.region[0] < 0 || task.region[1] < 0 || task.region[2] <= 0 || task.region[3] <= 0) {
        print_result(id, false, "region error");
        return;
      }
      if (task.region[0] + task.region[2] > image->cols || task.region[1] + task.region[3] > image->rows) {
        print_result(id, false, "region exceeded");
        return;
      }
      cv::Rect crop(task.region[0], task.region[1], task.region[2], task.region[3]);
      std::shared_ptr<cv::Mat> tmp_image = image;
      image = std::shared_ptr<cv::Mat>(new cv::Mat());
      *image = (*tmp_image)(crop);
    }
  }
  else {
    if (task.region.size() < 4 || task.region[2] <= 0 || task.region[3] <= 0) {
      print_result(id, false, "region error");
      return;
    }
    cv::Mat captured_image = captureScreenMat(task.region[0], task.region[1], task.region[2], task.region[3]);
    if (!captured_image.data) {
      print_result(id, false, "captured screen without data");
      return;
    }
    cv::cvtColor(captured_image, *image, cv::COLOR_BGRA2BGR);
  }

  auto mat_exchanger = apply_actions(id, *image, task.actions);
  if (!mat_exchanger.first) {
    return;
  }

  const cv::Mat& final_image = mat_exchanger.second ? *mat_exchanger.second : *image;
  std::vector<OCRPredictResult> ocr_result = ppocr->ocr(final_image, FLAGS_det, FLAGS_rec, FLAGS_cls);

  if (FLAGS_visualize && FLAGS_det) {
    std::unique_ptr<char[]> buf(new char[size_t(128)]);
    if (!task.region.empty()) {
      std::snprintf(buf.get(), 128, "/ocr_%d_%d_%d_%d.png", task.region[0], task.region[1], task.region[2], task.region[3]);
    }
    else {
      bool formatted = false;
      for (auto p = task.actions.begin(); p != task.actions.end(); ++p) {
        auto j = json::parse(*p, nullptr, false);
        auto action = j.template get<Action>();
        if (action.action == "crop") {
          auto j2 = json::parse(action.params, nullptr, false);
          auto ap = j2.template get<ActionCropParams>();
          std::snprintf(buf.get(), 128, "/ocr_%d_%d_%d_%d.png", ap.region[0], ap.region[1], ap.region[2], ap.region[3]);
          formatted = true;
          break;
        }
      }
      if (!formatted) {
        std::snprintf(buf.get(), 128, "/ocr_image.png");
      }
    }
    Utility::VisualizeBboxes(final_image, ocr_result, FLAGS_output + "/" + buf.get());
  }

  // revert scaling result to match original image
  if (final_image.cols != image->cols || final_image.rows != image->rows) {
    float scales[2] = { float(final_image.cols) / image->cols, float(final_image.rows) / image->rows };
    size_t index = 0;
    for (std::vector<OCRPredictResult>::iterator p = ocr_result.begin(); p != ocr_result.end(); ++p) {
      for (std::vector<std::vector<int>>::iterator p2 = p->box.begin(); p2 != p->box.end(); ++p2) {
        std::transform(p2->cbegin(), p2->cend(), p2->begin(), [&](int v) {
          return int(std::round(v / scales[index++ % 2]));
          });
      }
    }
  }

  print_result(id, true, ocr_result);

  if (FLAGS_benchmark) {
    ppocr->benchmark_log(1);
  }
}

void Worker::do_execute(const std::string& id, const LocateTask& task) {
  if (FLAGS_visualize) {
    std::cerr << "executing locate task" << std::endl;
  }
  if (task.method != cv::TemplateMatchModes::TM_CCOEFF_NORMED && task.method != cv::TemplateMatchModes::TM_CCORR_NORMED) {
    print_result(id, false, "method error");
    return;
  }

  std::shared_ptr<cv::Mat> image(new cv::Mat());
  std::string mode = task.mode;
  if (task.mode.empty() || task.mode == "images_on_screen" || task.mode == "screen_in_images") {
    if (task.region.size() < 4 || task.region[2] <= 0 || task.region[3] <= 0) {
      print_result(id, false, "region error");
      return;
    }
    cv::Mat captured_bgra = captureScreenMat(task.region[0], task.region[1], task.region[2], task.region[3]);
    if (!captured_bgra.data) {
      print_result(id, false, "captured screen without data");
      return;
    }

    mode = task.mode.empty() || task.mode == "images_on_screen" ? "images_in_image" : "image_in_images";
    cv::cvtColor(captured_bgra, *image, cv::COLOR_BGRA2BGR); // remove alpha channel in captured image
  }
  else {
    // keep alpha channel if it exists
    if (!read_image(task.image, *image, cv::IMREAD_UNCHANGED)) {
      std::cerr << "[ERROR] can't load the image: " << task.image << std::endl;
      print_result(id, false, "failed load image");
      return;
    }

    if (!task.region.empty()) {
      if (task.region.size() != 4 || task.region[0] < 0 || task.region[1] < 0 || task.region[2] <= 0 || task.region[3] <= 0) {
        print_result(id, false, "region error");
        return;
      }
      if (task.region[0] + task.region[2] > image->cols || task.region[1] + task.region[3] > image->rows) {
        print_result(id, false, "region exceeded");
        return;
      }
      cv::Rect crop(task.region[0], task.region[1], task.region[2], task.region[3]);
      std::shared_ptr<cv::Mat> tmp_image = image;
      image = std::shared_ptr<cv::Mat>(new cv::Mat());
      *image = (*tmp_image)(crop);
    }
  }

  auto mat_exchanger = apply_actions(id, *image, task.actions);
  if (!mat_exchanger.first) {
    return;
  }

  const cv::Mat& final_image = mat_exchanger.second ? *mat_exchanger.second : *image;
  if (FLAGS_visualize) {
    std::unique_ptr<char[]> buf(new char[size_t(128)]);
    if (!task.region.empty()) {
      std::snprintf(buf.get(), 128, "/loc_%d_%d_%d_%d.png", task.region[0], task.region[1], task.region[2], task.region[3]);
    }
    else {
      std::snprintf(buf.get(), 128, "/loc_image.png");
    }
    cv::imwrite(FLAGS_output + buf.get(), final_image);
  }

  std::shared_ptr<LocateResult> loc_result = do_locate(id, final_image, task.images, task.mask, mode, task.method, task.confidence);

  if (loc_result) {
    if (loc_result->located >= 0 && loc_result->score >= task.confidence) {
      if (mode == "images_in_image") {
        if (final_image.cols != image->cols) {
          float scale = float(image->cols) / final_image.cols;
          loc_result->region.x = std::lround(loc_result->region.x * scale);
          loc_result->region.width = std::lround(loc_result->region.width * scale);
        }
        if (final_image.rows != image->rows) {
          float scale = float(image->rows) / final_image.rows;
          loc_result->region.y = std::lround(loc_result->region.y * scale);
          loc_result->region.height = std::lround(loc_result->region.height * scale);
        }
        if (!task.region.empty()) {
          loc_result->region.x += task.region[0];
          loc_result->region.y += task.region[1];
        }
      }
      cv::Rect& rc = loc_result->region;
      json result = { loc_result->located, rc.x, rc.y, rc.width, rc.height, float(loc_result->score) };
      print_result(id, true, result.dump());
    }
    else {
      print_result(id, false, "locate failed");
    }
  }
}

// mode:
//  - images_in_image
//  - image_in_images
std::shared_ptr<LocateResult> Worker::do_locate(
  const std::string& id,
  const cv::Mat& image,
  const std::vector<std::string>& images,
  const std::string& mask,
  const std::string& mode,
  int method,
  float confidence
) {
  std::cerr << "calling do_locate" << std::endl;

  // TODO support task.grayscale

  // TODO determine thread number by param in task
  auto match_method = cv::TemplateMatchModes(method);
  std::shared_ptr<LocateResult> loc_result(new LocateResult());
  std::mutex mut;
  std::vector<std::shared_ptr<std::thread>> threads;
  std::size_t running_index = 0;

  cv::Mat final_image = image;
  std::shared_ptr<cv::Mat> mask_image;
  if (!mask.empty()) {
    if (mask == AUTO_MASK) {
      if (mode == "image_in_images" && image.channels() == 4) {
        std::cerr << "using alpha channel as mask" << std::endl;
        cv::Mat alpha_channel;
        cv::extractChannel(image, alpha_channel, 3);
        cv::Mat array[3] = { alpha_channel, alpha_channel , alpha_channel };
        mask_image = std::make_shared<cv::Mat>();
        cv::merge(array, 3, *mask_image);
        cv::cvtColor(image, final_image, cv::COLOR_BGRA2BGR);

        cv::imwrite(FLAGS_output + "/rgb.png", final_image);
        cv::imwrite(FLAGS_output + "/mask.png", *mask_image);
      }
    }
    else {
      mask_image = std::make_shared<cv::Mat>();
      if (!read_image(mask, *mask_image, cv::IMREAD_COLOR)) {
        std::cerr << "[ERROR] can't load mask: " << mask << std::endl;
        print_result(id, false, "can't load mask");
        return nullptr;
      }
      if (mask_image->cols != image.cols || mask_image->rows != image.rows) {
        auto resized_mask = std::make_shared<cv::Mat>();
        cv::resize(*mask_image, *resized_mask, cv::Size(image.cols, image.rows));
        mask_image = resized_mask;
      }
    }
  }

  if (final_image.channels() == 4) {
    cv::cvtColor(image, final_image, cv::COLOR_BGRA2BGR);
  }

  for (size_t ti = 0; ti < 4 && ti < images.size(); ti++) {
    std::shared_ptr<std::thread> t(new std::thread([&] {
      while (true) {
        size_t image_index;
        {
          std::lock_guard<std::mutex> locker(mut);
          if (running_index >= images.size()) {
            return nullptr;
          }
          image_index = running_index++;
        }

        cv::Mat indexed_image;

        cv::Mat result;
        if (mode == "images_in_image") {
          if (!read_image(images[image_index], indexed_image, mask == AUTO_MASK ? cv::IMREAD_UNCHANGED : cv::IMREAD_COLOR)) {
            std::cerr << "[ERROR] can't load the image: " << images[image_index] << std::endl;
            print_result(id, false, "can't load the image");
            return nullptr;
          }
          if (indexed_image.rows > final_image.rows || indexed_image.cols > final_image.cols) {
            print_result(id, false, "template's size out of range");
            return nullptr;
          }
          if (mask == AUTO_MASK && indexed_image.channels() == 4) {
            std::cerr << "using alpha chanel as mask" << std::endl;
            cv::Mat alpha_channel;
            cv::extractChannel(indexed_image, alpha_channel, 3);
            cv::Mat array[3] = { alpha_channel, alpha_channel , alpha_channel };
            mask_image = std::make_shared< cv::Mat>();
            cv::merge(array, 3, *mask_image);
            cv::cvtColor(indexed_image, indexed_image, cv::COLOR_BGRA2BGR);

            cv::imwrite(FLAGS_output + "/rgb.png", indexed_image);
            cv::imwrite(FLAGS_output + "/mask.png", *mask_image);
          }
          else if (mask_image && (mask_image->cols != indexed_image.cols || mask_image->rows != indexed_image.rows)) {
            auto resized_mask = std::make_shared<cv::Mat>();
            cv::resize(*mask_image, *resized_mask, cv::Size(indexed_image.cols, indexed_image.rows));
            mask_image = resized_mask;
          }

          cv::matchTemplate(final_image, indexed_image, result, match_method, mask_image ? *mask_image : cv::noArray());
        }
        else {
          if (!read_image(images[image_index], indexed_image)) {
            std::cerr << "[ERROR] can't load the image: " << images[image_index] << std::endl;
            print_result(id, false, "can't load the image");
            return nullptr;
          }
          if (final_image.rows > indexed_image.rows || final_image.cols > indexed_image.cols) {
            std::cerr << "[ERROR] template's size out of range" << std::endl;
            print_result(id, false, "template's size out of range");
            return nullptr;
          }
          cv::matchTemplate(indexed_image, final_image, result, match_method, mask_image ? *mask_image : cv::noArray());
        }

        double minVal, maxVal;
        cv::Point minLoc, maxLoc;
        cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, cv::Mat());
        if (FLAGS_visualize) {
          std::cerr << "x:" << maxLoc.x << ", y:" << maxLoc.y << ", score:" << (maxVal + 1) / 2
            << ", index:" << image_index << std::endl;
        }

        std::lock_guard<std::mutex> locker(mut);
        if ((maxVal + 1) / 2 > loc_result->score) {
          loc_result->score = (maxVal + 1) / 2;
          loc_result->located = int(image_index);
          if (mode == "images_in_image") {
            loc_result->region.x = maxLoc.x * image.cols / final_image.cols;
            loc_result->region.y = maxLoc.y * image.rows / final_image.rows;
            loc_result->region.width = indexed_image.cols * image.cols / final_image.cols;
            loc_result->region.height = indexed_image.rows * image.rows / final_image.rows;
          }
          else {
            loc_result->region.x = maxLoc.x;
            loc_result->region.y = maxLoc.y;
            loc_result->region.width = final_image.cols;
            loc_result->region.height = final_image.rows;
          }
        }
      }
      }));
    threads.push_back(t);
  }
  for (size_t i = 0; i < threads.size(); i++) {
    threads[i]->join();
  }
  return loc_result;
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
