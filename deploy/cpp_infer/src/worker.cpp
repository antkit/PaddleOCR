#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <Windows.h>
#endif

#include <include/args.h>
#include <include/worker.h>

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

void Worker::execute(const std::string& task) {
  std::vector<std::string> args;
  split(task, args);
  std::vector<char*> argv;
  for (const auto& arg : args) {
    std::cout << "ARG: " << arg << std::endl;
    // TODO filter args, ignore det, rec, models_dir and cls
    argv.push_back((char*)arg.data());
  }
  argv.push_back(nullptr);
  int argc = int(argv.size()) - 1;
  char** argvv = argv.data();

  // reset flags
  FLAGS_type = "ocr";
  FLAGS_det = true;
  FLAGS_rec = true;
  FLAGS_cls = false;
  FLAGS_lang = "ch";
  FLAGS_image_dir.clear();
  FLAGS_x = 0;
  FLAGS_y = 0;
  FLAGS_w = 0;
  FLAGS_h = 0;

  google::ParseCommandLineFlags(&argc, &argvv, true);

  if (FLAGS_type == "ocr" && FLAGS_image_dir.empty()) {
    const std::string key = FLAGS_lang;
    if (ppocrs_.find(key) == ppocrs_.end()) {
      ppocrs_[key] = std::shared_ptr<PPOCR>(new PPOCR());
    }
    std::shared_ptr<PPOCR> ppocr = ppocrs_[key];
    if (!ppocr) {
      ppocr = std::shared_ptr<PPOCR>(new PPOCR());
    }
    do_ocr(ppocr, FLAGS_x, FLAGS_y, FLAGS_w, FLAGS_h);
    return;
  }

  std::vector<cv::String> cv_all_img_names;
  cv::glob(FLAGS_image_dir, cv_all_img_names);
  std::cout << "total images num: " << cv_all_img_names.size() << std::endl;
  if (FLAGS_type == "ocr") {
    const std::string key = FLAGS_lang;
    if (ppocrs_.find(key) == ppocrs_.end()) {
      ppocrs_[key] = std::shared_ptr<PPOCR>(new PPOCR());
    }
    std::shared_ptr<PPOCR> ppocr = ppocrs_[key];
    if (!ppocr) {
      ppocr = std::shared_ptr<PPOCR>(new PPOCR());
    }
    do_ocr(ppocr, cv_all_img_names);
  }
  else if (FLAGS_type == "structure") {
    // TODO structure
    const std::string key = "";
    if (ps_engines_.find(key) == ps_engines_.end()) {
      ps_engines_[key] = std::shared_ptr<PaddleStructure>(new PaddleStructure());
    }
    std::shared_ptr<PaddleStructure> ps_engine = ps_engines_[key];
    if (!ps_engine) {
      ps_engine = std::shared_ptr<PaddleStructure>(new PaddleStructure());
    }
    do_structure(ps_engine, cv_all_img_names);
  }
}

void Worker::do_ocr(std::shared_ptr<PPOCR> ppocr, int x, int y, int w, int h) {
  if (FLAGS_benchmark) {
    ppocr->reset_timer();
  }

  // TODO deal with w=0 and h=0
  cv::Mat img_with_alpha = captureScreenMat(x, y, w, h);
  if (!img_with_alpha.data) {
    std::cerr << "[ERROR] screenshot invalid data" << std::endl;
    return;
  }
  cv::imwrite(FLAGS_output + "/screenshot.png", img_with_alpha);

  std::cout << "\t ocr running..." << std::endl;
  cv::Mat img = cv::Mat();
  cv::cvtColor(img_with_alpha, img, cv::COLOR_BGRA2BGR);
  std::vector<OCRPredictResult> ocr_results = ppocr->ocr(img, FLAGS_det, FLAGS_rec, FLAGS_cls);
  std::cout << "\t ocr finished" << std::endl;

  std::unique_ptr<char[]> buf(new char[size_t(128)]);
  int size = std::snprintf(buf.get(), 128, "ss_%d_%d_%d_%d.png", x, y, w, h);
  std::cout << "predict screenshot: " << buf.get() << std::endl;
  Utility::print_result(ocr_results);
  if (FLAGS_visualize && FLAGS_det) {
    Utility::VisualizeBboxes(img, ocr_results, FLAGS_output + "/" + buf.get());
  }
  if (FLAGS_benchmark) {
    ppocr->benchmark_log(1);
  }
}

void Worker::do_ocr(std::shared_ptr<PPOCR> ppocr, std::vector<cv::String>& cv_all_img_names) {
  if (FLAGS_benchmark) {
    ppocr->reset_timer();
  }

  std::vector<cv::Mat> img_list;
  std::vector<cv::String> img_names;
  for (int i = 0; i < cv_all_img_names.size(); ++i) {
    cv::Mat img = cv::imread(cv_all_img_names[i], cv::IMREAD_COLOR);
    if (!img.data) {
      std::cerr << "[ERROR] image read failed! image path: " << cv_all_img_names[i] << std::endl;
      return;
    }
    img_list.push_back(img);
    img_names.push_back(cv_all_img_names[i]);
  }

  std::vector<std::vector<OCRPredictResult>> ocr_results =
    ppocr->ocr(img_list, FLAGS_det, FLAGS_rec, FLAGS_cls);

  for (int i = 0; i < img_names.size(); ++i) {
    std::cout << "predict img: " << cv_all_img_names[i] << std::endl;
    Utility::print_result(ocr_results[i]);
    if (FLAGS_visualize && FLAGS_det) {
      std::string file_name = Utility::basename(img_names[i]);
      cv::Mat srcimg = img_list[i];
      Utility::VisualizeBboxes(srcimg, ocr_results[i], FLAGS_output + "/" + file_name);
    }
  }
  if (FLAGS_benchmark) {
    ppocr->benchmark_log(cv_all_img_names.size());
  }
}

void Worker::do_structure(std::shared_ptr<PaddleStructure> engine, std::vector<cv::String>& cv_all_img_names) {
  if (FLAGS_benchmark) {
    engine->reset_timer();
  }

  for (int i = 0; i < cv_all_img_names.size(); i++) {
    std::cout << "predict img: " << cv_all_img_names[i] << std::endl;
    cv::Mat img = cv::imread(cv_all_img_names[i], cv::IMREAD_COLOR);
    if (!img.data) {
      std::cerr << "[ERROR] image read failed! image path: "
        << cv_all_img_names[i] << std::endl;
      continue;
    }

    std::vector<StructurePredictResult> structure_results = engine->structure(
      img, FLAGS_layout, FLAGS_table, FLAGS_det && FLAGS_rec);

    for (int j = 0; j < structure_results.size(); j++) {
      std::cout << j << "\ttype: " << structure_results[j].type
        << ", region: [";
      std::cout << structure_results[j].box[0] << ","
        << structure_results[j].box[1] << ","
        << structure_results[j].box[2] << ","
        << structure_results[j].box[3] << "], score: ";
      std::cout << structure_results[j].confidence << ", res: ";

      if (structure_results[j].type == "table") {
        std::cout << structure_results[j].html << std::endl;
        if (structure_results[j].cell_box.size() > 0 && FLAGS_visualize) {
          std::string file_name = Utility::basename(cv_all_img_names[i]);

          Utility::VisualizeBboxes(img, structure_results[j],
            FLAGS_output + "/" + std::to_string(j) +
            "_" + file_name);
        }
      }
      else {
        std::cout << "count of ocr result is : "
          << structure_results[j].text_res.size() << std::endl;
        if (structure_results[j].text_res.size() > 0) {
          std::cout << "********** print ocr result "
            << "**********" << std::endl;
          Utility::print_result(structure_results[j].text_res);
          std::cout << "********** end print ocr result "
            << "**********" << std::endl;
        }
      }
    }
  }
  if (FLAGS_benchmark) {
    engine->benchmark_log(cv_all_img_names.size());
  }
}

// We will NOT reconstruct PPOCR and PaddleStructure instances.
// For PPOCR instance, [det, rec, cls] will be stored.
// For PaddleStructure instance, [layout, table] will be stored.
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
      std::cout << "FAILED to getline" << std::endl;
      std::this_thread::sleep_for(1000ms);
      continue;
    }
    if (line == "DONE") {
      break;
    }

    // In tauri sidecar, will continuely receive empty line
    if (line.empty()) {
      std::this_thread::sleep_for(1000ms);
      continue;
    }

    std::cout << "LINE: [" << line << "]" << std::endl;
    worker.execute(line);
  }

  std::cout << "quiting..." << std::endl;
  return 0;
}
