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

int main(int argc, const char * argv[]) {
    cv::Mat outresponces;
    auto traindata = cv::ml::TrainData::loadFromCSV("./letter-recognition.data.csv", 0);
    auto  datatrayn = traindata->getTrainSamples();
    std::cout << "Имеем  " << datatrayn.rows << " примеров длинной вектора в " << datatrayn.cols << std::endl;
    auto trainResponse = traindata->getResponses();
    std::cout << "Имеем " << trainResponse.rows << " откликов размером в" << trainResponse.cols << std::endl;
    auto trainlables = traindata->getClassLabels();
    std::cout << "Имеем " << trainlables.rows << " меток размером в" << trainlables.cols << std::endl;
    for (int i = 0; i < trainlables.rows; i++) {
        std::cout << trainlables.at<short>(i, 0) << " ";
    } std::cout << std::endl;
    // разделение выборок на тестовую и обучающую
    traindata->setTrainTestSplit(18000);
    auto ts = traindata->getTestSamples();
    std::cout << "Получено " << ts.rows << " тестовых примеров\n";
    // случайный лес
    auto randomForest = cv::ml::RTrees::create();
    randomForest->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, 100, 1e-6));
    cv::Mat newtrainresponce(trainResponse.rows, datatrayn.cols, CV_32F);
    randomForest->setActiveVarCount(4);
    randomForest->train(traindata);
    std::cout <<"Ошибка случайного леса " << randomForest->calcError(traindata, true, outresponces);
    return 0; // выход
}
