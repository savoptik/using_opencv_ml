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

void conversionMNISTToCVMAT(std::vector<std::vector<unsigned char>>& imgs, cv::Mat& res);
void convertionLablesToVector(std::vector<unsigned char>& lables, cv::Mat& res);

int main(int argc, const char * argv[]) {
    auto ds = mnist::read_dataset(); // читаем базу мнист
    cv::Mat trainImages(ds.training_images.size(), 28 * 28, CV_32F), testImages(ds.test_images.size(), 28 * 28, CV_32F);
    conversionMNISTToCVMAT(ds.training_images, trainImages);
    conversionMNISTToCVMAT(ds.test_images, testImages);
    cv::Mat trainLables(ds.training_labels.size(), 10, CV_32F), testLables(ds.test_labels.size(), 10, CV_32F);
    convertionLablesToVector(ds.training_labels, trainLables);
    convertionLablesToVector(ds.test_labels, testLables);
    auto traindata = cv::ml::TrainData::create(trainImages, cv::ml::ROW_SAMPLE, trainLables);
    return 0;
}

void conversionMNISTToCVMAT(std::vector<std::vector<unsigned char>>& imgs, cv::Mat& res) {
    for (int i = 0; i < res.rows; i++) {
        for (int j = 0; j < res.cols; j++) {
            res.at<float>(i, j) = imgs[i][j];
        }
    }
}

void convertionLablesToVector(std::vector<unsigned char>& lables, cv::Mat& res) {
    for (int i = 0; i < res.rows; i++) {
        res.at<float>(i, lables[i]) = 1;
    }
}
