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

struct Action {
  std::string action;
  std::string params;
};

void from_json(const json& j, Action& a) {
  j.at("action").get_to(a.action);
  if (j.contains("params")) {
    j.at("params").get_to(a.params);
  }
}

struct ActionResizeParams {
  float factor; // use factor first
  int width;
  int height;
};

void from_json(const json& j, ActionResizeParams& a) {
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

struct ActionFlipParams {
  int code;
};

void from_json(const json& j, ActionFlipParams& a) {
  if (j.contains("code")) {
    j.at("code").get_to(a.code);
  }
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

void from_json(const json& j, Task& t) {
  j.at("id").get_to(t.id);
  j.at("command").get_to(t.command);
  if (j.contains("content")) {
    j.at("content").get_to(t.content);
  }
}

OcrTask::OcrTask()
  : det(true), rec(true), cls(false) {
}

void from_json(const json& j, OcrTask& t) {
  j.at("lang").get_to(t.lang);
  if (j.contains("img")) {
    j.at("img").get_to(t.img);
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

void from_json(const json& j, LocateTask& t) {
  j.at("images").get_to(t.images);
  j.at("region").get_to(t.region);
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
  std::shared_ptr<PPOCR> ppocr = ppocr_by_lang(task.lang);
  if (FLAGS_benchmark) {
    ppocr->reset_timer();
  }

  std::shared_ptr<cv::Mat> image(new cv::Mat());
  if (!task.img.empty()) {
    cv::Mat file_image;
    if (!read_image(task.img, file_image)) {
      std::cerr << "[ERROR] can't load the image: " << task.img << std::endl;
      print_result(id, false, "failed load image");
      return;
    }
    if (!task.region.empty()) {
      if (task.region.size() != 4 || task.region[0] < 0 || task.region[1] < 0 || task.region[2] <= 0 || task.region[3] <= 0) {
        print_result(id, false, "region error");
        return;
      }
      if (task.region[0] + task.region[2] > file_image.cols || task.region[1] + task.region[3] > file_image.rows) {
        print_result(id, false, "region exceeded");
        return;
      }
      cv::Rect crop(task.region[0], task.region[1], task.region[2], task.region[3]);
      *image = file_image(crop);
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

  std::unique_ptr<cv::Size> resize = std::make_unique<cv::Size>(image->cols, image->rows);
  std::unique_ptr<cv::Mat> mat_exchanger;
  for (size_t i = 0; i < task.actions.size(); i++) {
    std::cerr << "action:" << task.actions[i] << std::endl;
    auto j = json::parse(task.actions[i], nullptr, false);
    if (j.is_discarded()) {
      print_result(id, false, "invalid action content in ocr");
      return;
    }
    auto action = j.template get<Action>();
    auto j2 = json::parse(action.params, nullptr, false);
    if (j2.is_discarded()) {
      print_result(id, false, "invalid action params in ocr");
      return;
    }
    if (action.action == "resize") {
      auto params = j2.template get<ActionResizeParams>();
      if (params.factor > 0) {
        if (std::abs(params.factor - 1.0) > 0.00001) {
          resize->width = int(image->cols * params.factor);
          resize->height = int(image->rows * params.factor);
        }
      }
      else {
        if (params.width <= 0 || params.height <= 0) {
          print_result(id, false, "invalid resize params in ocr");
          return;
        }
        if (params.width != image->cols || params.height != image->rows) {
          resize->width = params.width;
          resize->height = params.height;
        }
      }
      if (resize->width != image->cols || resize->height != image->rows) {
        std::unique_ptr<cv::Mat> mat_output(new cv::Mat());
        cv::resize(mat_exchanger ? *mat_exchanger : *image, *mat_output, *resize);
        mat_exchanger = std::move(mat_output);
      }
    }
    else if (action.action == "flip") {
      auto params = j2.template get<ActionFlipParams>();
      std::unique_ptr<cv::Mat> mat_output(new cv::Mat());
      cv::flip(mat_exchanger ? *mat_exchanger : *image, *mat_output, params.code);
      mat_exchanger = std::move(mat_output);
    }
    else if (action.action == "grayscale") {
      print_result(id, false, "not implemented grayscale in ocr");
    }
  }

  const cv::Mat& image_processed = mat_exchanger ? *mat_exchanger : *image;
  std::vector<OCRPredictResult> ocr_result = ppocr->ocr(image_processed, FLAGS_det, FLAGS_rec, FLAGS_cls);

  const std::vector<int>& rg = task.region;
  if (FLAGS_visualize && FLAGS_det) {
    std::unique_ptr<char[]> buf(new char[size_t(128)]);
    int size = std::snprintf(buf.get(), 128, "ocr_%d_%d_%d_%d.png", rg[0], rg[1], rg[2], rg[3]);
    Utility::VisualizeBboxes(image_processed, ocr_result, FLAGS_output + "/" + buf.get());
  }

  // revert scaling result to match original image
  if (resize->width != image->cols || resize->height != image->rows) {
    float scales[2] = { float(resize->width) / image->cols, float(resize->height) / image->rows };
    size_t index = 0;
    for (std::vector<OCRPredictResult>::iterator p = ocr_result.begin(); p != ocr_result.end(); ++p) {
      for (std::vector<std::vector<int>>::iterator p2 = p->box.begin(); p2 != p->box.end(); ++p2) {
        std::transform(p2->cbegin(), p2->cend(), p2->begin(), [&](int v) {
          return int(std::round(v / scales[index++ % 2]));
          });
      }
    }
  }
  //std::cerr << "\t ocr finished" << std::endl;

  std::cerr << "predict screen: " << rg[0] << "," << rg[1] << "," << rg[2] << "," << rg[3] << std::endl;
  print_result(id, true, ocr_result);

  if (FLAGS_benchmark) {
    ppocr->benchmark_log(1);
  }
}

LocateResult::LocateResult()
  : located(-1), score(0) {
}

void Worker::do_execute(const std::string& id, const LocateTask& task) {
  // TODO support task.grayscale
  if (FLAGS_visualize) {
    std::cerr << "executing locate task" << std::endl;
  }
  if (task.method != cv::TemplateMatchModes::TM_CCOEFF_NORMED && task.method != cv::TemplateMatchModes::TM_CCORR_NORMED) {
    print_result(id, false, "method error");
    return;
  }

  std::string mode = task.mode;
  std::shared_ptr<LocateResult> loc_result;
  if (task.mode.empty() || task.mode == "images_on_screen" || task.mode == "screen_in_images") {
    std::shared_ptr<cv::Mat> image(new cv::Mat());
    if (task.region.size() < 4 || task.region[2] <= 0 || task.region[3] <= 0) {
      print_result(id, false, "region error");
      return;
    }
    cv::Mat captured_bgra = captureScreenMat(task.region[0], task.region[1], task.region[2], task.region[3]);
    if (!captured_bgra.data) {
      print_result(id, false, "captured screen without data");
      return;
    }

    cv::cvtColor(captured_bgra, *image, cv::COLOR_BGRA2BGR);
    if (FLAGS_visualize) {
      std::unique_ptr<char[]> buf(new char[size_t(128)]);
      const std::vector<int>& rg = task.region;
      std::snprintf(buf.get(), 128, "/loc_%d_%d_%d_%d.png", rg[0], rg[1], rg[2], rg[3]);
      cv::imwrite(FLAGS_output + buf.get(), *image);
    }

    mode = task.mode.empty() || task.mode == "images_on_screen" ? "images_in_image" : "image_in_images";
    loc_result = do_locate(id, *image, task.images, task.mask, task.actions, mode, task.method, task.confidence);
  }
  else {
    if (!task.region.empty() && (task.region.size() != 4 || task.region[0] < 0 || task.region[1] < 0 || task.region[2] <= 0 || task.region[3] <= 0)) {
      print_result(id, false, "region error");
      return;
    }
    std::shared_ptr<cv::Mat> file_image(new cv::Mat());
    if (!read_image(task.image, *file_image)) {
      std::cerr << "[ERROR] can't load the image: " << task.image << std::endl;
      print_result(id, false, "failed load image");
      return;
    }
    cv::Mat image = task.region.empty() ? *file_image : cv::Mat();
    if (!task.region.empty()) {
      if (task.region.size() != 4 || task.region[0] < 0 || task.region[1] < 0 || task.region[2] <= 0 || task.region[3] <= 0) {
        print_result(id, false, "region error");
        return;
      }
      if (task.region[0] + task.region[2] > file_image->cols || task.region[1] + task.region[3] > file_image->rows) {
        print_result(id, false, "region exceeded");
        return;
      }
      cv::Rect crop(task.region[0], task.region[1], task.region[2], task.region[3]);
      image = (*file_image)(crop);
    }
    if (FLAGS_visualize) {
      std::unique_ptr<char[]> buf(new char[size_t(128)]);
      if (!task.region.empty()) {
        const std::vector<int>& rg = task.region;
        std::snprintf(buf.get(), 128, "/loc_%d_%d_%d_%d.png", rg[0], rg[1], rg[2], rg[3]);
      }
      else {
        std::snprintf(buf.get(), 128, "/loc_image.png");
      }
      cv::imwrite(FLAGS_output + buf.get(), image);
    }

    loc_result = do_locate(id, image, task.images, task.mask, task.actions, mode, task.method, task.confidence);
  }

  if (loc_result) {
    if (loc_result->located >= 0 && loc_result->score >= task.confidence) {
      if (mode == "images_in_image" && !task.region.empty()) {
        loc_result->region.x += task.region[0];
        loc_result->region.y += task.region[1];
      }
      cv::Rect& rc = loc_result->region;
      json result = { loc_result->located, rc.x, rc.y, rc.width, rc.height, float(loc_result->score) };
      print_result(id, true, result.dump());
      return;
    }
    print_result(id, false, "locate failed");
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
  const std::vector<std::string>& actions,
  const std::string& mode,
  int method,
  float confidence
) {
  // TODO support task.grayscale
  if (mode != "images_in_image" && mode != "image_in_images") {
    print_result(id, false, "mode error");
    return nullptr;
  }
  if (method != cv::TemplateMatchModes::TM_CCOEFF_NORMED && method != cv::TemplateMatchModes::TM_CCORR_NORMED) {
    print_result(id, false, "method error");
    return nullptr;
  }

  std::unique_ptr<cv::Size> resize = std::make_unique<cv::Size>(image.cols, image.rows);
  std::unique_ptr<cv::Mat> mat_exchanger;
  for (size_t i = 0; i < actions.size(); i++) {
    std::cerr << "action:" << actions[i] << std::endl;
    auto j = json::parse(actions[i], nullptr, false);
    if (j.is_discarded()) {
      print_result(id, false, "invalid action content in locate");
      return nullptr;
    }
    auto action = j.template get<Action>();
    auto j2 = json::parse(action.params, nullptr, false);
    if (j2.is_discarded()) {
      print_result(id, false, "invalid action params in locate");
      return nullptr;
    }
    if (action.action == "resize") {
      auto params = j2.template get<ActionResizeParams>();
      if (params.factor > 0) {
        if (std::abs(params.factor - 1.0) > 0.00001) {
          resize->width = int(image.cols * params.factor);
          resize->height = int(image.rows * params.factor);
        }
      }
      else {
        if (params.width <= 0 || params.height <= 0) {
          print_result(id, false, "invalid resize params in ocr");
          return nullptr;
        }
        resize->width = params.width;
        resize->height = params.height;
      }
      if (resize->width != image.cols || resize->height != image.rows) {
        std::unique_ptr<cv::Mat> mat_output(new cv::Mat());
        cv::resize(mat_exchanger ? *mat_exchanger : image, *mat_output, *resize);
        mat_exchanger = std::move(mat_output);
      }
    }
    else if (action.action == "flip") {
      auto params = j2.template get<ActionFlipParams>();
      std::unique_ptr<cv::Mat> mat_output(new cv::Mat());
      cv::flip(mat_exchanger ? *mat_exchanger : image, *mat_output, params.code);
      mat_exchanger = std::move(mat_output);
    }
    else if (action.action == "grayscale") {
      print_result(id, false, "not implemented grayscale in locate");
      return nullptr;
    }
  }

  const cv::Mat& image_processed = mat_exchanger ? *mat_exchanger : image;
  std::unique_ptr<cv::Mat> mask_image;
  if (!mask.empty()) {
    std::cerr << "loading mask:" << mask << std::endl;
    std::unique_ptr<cv::Mat> tmp = std::make_unique<cv::Mat>();
    if (!read_image(mask, *tmp)) {
      std::cerr << "[ERROR] can't load mask: " << mask << std::endl;
      print_result(id, false, "can't load mask");
      return nullptr;
    }
    if (mode == "images_in_image") {
      // mask's size determined by each image in task.images
      mask_image = std::move(tmp);
    }
    else {
      // mask's size is fixed
      mask_image = std::make_unique<cv::Mat>();
      cv::resize(*tmp, *mask_image, cv::Size(image_processed.cols, image_processed.rows));
    }
  }

  std::shared_ptr<LocateResult> loc_result(new LocateResult());
  std::mutex mut;
  std::vector<std::shared_ptr<std::thread>> threads;
  std::size_t running_index = 0;
  // TODO thread number by param in task
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

        //std::cerr << "locating " << task.images[image_index] << std::endl;
        cv::Mat indexed_image;
        if (!read_image(images[image_index], indexed_image)) {
          std::cerr << "[ERROR] can't load the image: " << images[image_index] << std::endl;
          print_result(id, false, "can't load the image");
          return nullptr;
        }

        // TM_CCORR_NORMED is faster than TM_CCOEFF_NORMED when mask is enabled,
        // but TM_CCORR_NORMED sometimes is not accurate enough.
        // TM_CCOEFF_NORMED and TM_CCORR_NORMED range in -1~1, and the best matching is maxLoc.
        auto match_method = cv::TemplateMatchModes(method);
        cv::Mat result;
        if (mode == "images_in_image") {
          if (indexed_image.rows > image_processed.rows || indexed_image.cols > image_processed.cols) {
            std::cerr << "[ERROR] template's size out of range" << std::endl;
            print_result(id, false, "template's size out of range");
            return nullptr;
          }
          if (!mask_image) {
            cv::matchTemplate(image_processed, indexed_image, result, match_method);
          }
          else if (mask_image->cols != indexed_image.cols || mask_image->rows != indexed_image.rows) {
            cv::Mat resized_mask;
            cv::resize(*mask_image, resized_mask, cv::Size(indexed_image.cols, indexed_image.rows));
            cv::matchTemplate(image_processed, indexed_image, result, match_method, resized_mask);
          }
          else {
            cv::matchTemplate(image_processed, indexed_image, result, match_method, *mask_image);
          }
        }
        else {
          if (image_processed.rows > indexed_image.rows || image_processed.cols > indexed_image.cols) {
            std::cerr << "[ERROR] template's size out of range" << std::endl;
            print_result(id, false, "template's size out of range");
            return nullptr;
          }
          cv::matchTemplate(indexed_image, image_processed, result, match_method, mask_image ? *mask_image : cv::noArray());
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
            loc_result->region.x = maxLoc.x * image.cols / resize->width;
            loc_result->region.y = maxLoc.y * image.rows / resize->height;
            loc_result->region.width = indexed_image.cols * image.cols / resize->width;
            loc_result->region.height = indexed_image.rows * image.rows / resize->height;
          }
          else {
            loc_result->region.x = maxLoc.x;
            loc_result->region.y = maxLoc.y;
            loc_result->region.width = resize->width;
            loc_result->region.height = resize->height;
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
