//
//  main.cpp
//  using_opencv_l
//
//  Created by Артём Семёнов on 23/12/2018.
//  Copyright © 2018 Артём Семёнов. All rights reserved.
//

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/dnn.hpp>
#include <mnist/mnist_reader.hpp>

void conversionMNISTToCVMAT(std::vector<std::vector<unsigned char>>& imgs, std::vector<cv::Mat>& res);

int main(int argc, const char * argv[]) {
    auto ds = mnist::read_dataset(); // читаем базу мнист
    std::vector<cv::Mat> trainImages, testImages; // так как объекты ML работают с cv::Mat
    conversionMNISTToCVMAT(ds.training_images, trainImages); // приобразовываем изображения
    conversionMNISTToCVMAT(ds.test_images, testImages);
    std::cout << "преобразованно " << ds.training_images.size() << " тренеровочных примеров в " << trainImages.size() << " примеров\n";
    std::cout << "Преобразованно " << ds.test_images.size() << " тестовых примеров в " << testImages.size() << " примеров\n";
    return 0;
}

void conversionMNISTToCVMAT(std::vector<std::vector<unsigned char>>& imgs, std::vector<cv::Mat>& res) {
    for (auto img : imgs) {
        cv::Mat tIMG(28, 28, CV_8UC1);
        tIMG.data = img.data();
        res.push_back(tIMG);
    }
}
