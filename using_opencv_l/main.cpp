//
//  main.cpp
//  using_opencv_l
//
//  Created by Артём Семёнов on 23/12/2018.
//  Copyright © 2018 Артём Семёнов. All rights reserved.
//

#include <fstream>
#include <sstream>
#include <vector>
#include <string>

#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/dnn.hpp>

const char* keys =
    "{ help  h     | | Print help message. }"
    "{ input i     | | Path to input image or video file. Skip this argument to capture frames from a camera.}"
    "{ model m     | | Path to a binary file of model contains trained weights. "
                      "It could be a file with extensions .caffemodel (Caffe), "
                      ".pb (TensorFlow), .t7 or .net (Torch), .weights (Darknet) }"
    "{ config c    | | Path to a text file of model contains network configuration. "
                      "It could be a file with extensions .prototxt (Caffe), .pbtxt (TensorFlow), .cfg (Darknet) }"
    "{ framework f | | Optional name of an origin framework of the model. Detect it automatically if it does not set. }"
    "{ classes     | | Optional path to a text file with names of classes. }"
    "{ mean        | | Preprocess input image by subtracting mean values. Mean values should be in BGR order and delimited by spaces. }"
    "{ scale       | 1 | Preprocess input image by multiplying on a scale factor. }"
    "{ width       |   | Preprocess input image by resizing to a specific width. }"
    "{ height      |   | Preprocess input image by resizing to a specific height. }"
    "{ rgb         |   | Indicate that model works with RGB input images instead BGR ones. }"
    "{ backend     | 0 | Choose one of computation backends: "
                        "0: automatically (by default), "
                        "1: Halide language (http://halide-lang.org/), "
                        "2: Intel's Deep Learning Inference Engine (https://software.intel.com/openvino-toolkit), "
                        "3: OpenCV implementation }"
    "{ target      | 0 | Choose one of target computation devices: "
                        "0: CPU target (by default), "
                        "1: OpenCL, "
                        "2: OpenCL fp16 (half-float precision), "
                        "3: VPU }";

using namespace cv;
using namespace cv::dnn;

std::vector<std::string> classes; // Названия классов

int main(int argc, const char * argv[]) {

    CommandLineParser parser(argc, argv, keys); // разбираем параметры командной строки
    parser.about("Use this script to run classification deep learning networks using OpenCV.");
    if (argc == 1) { // Если при запуске не заданны параметры
        parser.printMessage(); // выводим ругательное сообщение.
        return 0; // выходим, делать тут всё равно нечего.
    }

    // в ином случае работаем
    float scale = parser.get<float>("scale"); // задаём масштаб
    Scalar mean = parser.get<Scalar>("mean"); // средний масштаб
    bool swapRB = parser.get<bool>("rgb");
    CV_Assert(parser.has("width"), parser.has("height"));
    int inpWidth = parser.get<int>("width"); // извлекаем ширину
    int inpHeight = parser.get<int>("height"); // извлекаем высоту
    String model = parser.get<String>("model"); // получаем название модели
    String config = parser.get<String>("config"); // получаем конфигурацию модели
    String framework = parser.get<String>("framework"); // получаем название фреймвёрка
    int backendId = parser.get<int>("backend"); // идентификатор бэкэнда
    int targetId = parser.get<int>("target"); // отслеживание

    // открытие файла с названиями классов
    if (parser.has("classes")) { // если название файла заданно
        std::string file = parser.get<String>("classes"); // создаём поток файла
        std::ifstream ifs(file.c_str());
        if (!ifs.is_open()) // если файла нет на диске
            CV_Error(Error::StsError, "File " + file + " not found"); // ругаемся
        std::string line;
        while (std::getline(ifs, line)) { // если удалось получить строку
            classes.push_back(line); // тащим строку в список классов
        }
    }

    CV_Assert(parser.has("model")); // проверяем модель
    Net net = readNet(model, config, framework); // создаём сеть.
    net.setPreferableBackend(backendId); // устанавливаем бэкэнд.
    net.setPreferableTarget(targetId); // устанавливаем отслеживание

        // создание окн
    static const std::string kWinName = "Deep learning image classification in OpenCV"; // получаем имя окна.
    namedWindow(kWinName, WINDOW_NORMAL); // создаём имя окна

        VideoCapture cap; // захват видео
    if (parser.has("input")) // если заданно имя входного файла
        cap.open(parser.get<String>("input")); // открываем файл входных данных
    else
        cap.open(0);

    Mat frame, blob;
    while (waitKey(1) < 0) {
        cap >> frame; // извлекаем изображения
        if (frame.empty()) { // если изображения нет
            waitKey(); // ждём нажатия клавиши
            break; // и прирываем цикл
        }
        blobFromImage(frame, blob, scale, Size(inpWidth, inpHeight), mean, swapRB, false); // выделяем часть изображения
        net.setInput(blob); // подаём сети изображение
        Mat prob = net.forward(); // берём из сети ответ
        Point classIdPoint; // точка для определения класса
        double confidence; // правдоподобие
        minMaxLoc(prob.reshape(1, 1), 0, &confidence, 0, &classIdPoint); // проверяем
        int classId = classIdPoint.x; // получаем номер класса
        // Получение информации об эффективности
        std::vector<double> layersTimes; // время работы слоёв
        double freq = getTickFrequency() / 1000;
        double t = net.getPerfProfile(layersTimes) / freq; // получение времени слоёв
        std::string label = format("Inference time: %.2f ms", t);
        putText(frame, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0)); // добавление текста
        // Вывод предсказанного класса
        label = format("%s: %.4f", (classes.empty() ? format("Class #%d", classId).c_str() :
                                    classes[classId].c_str()),
                       confidence);
        putText(frame, label, Point(0, 40), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));
        imshow(kWinName, frame); // ресуем изображение
    }

    return 0; // выход
}
