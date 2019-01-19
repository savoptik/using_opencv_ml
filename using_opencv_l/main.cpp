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
    auto traindata = cv::ml::TrainData::loadFromCSV("./letter-recognition.data.csv", 0);
    auto  datatrayn = traindata->getTrainSamples();
    std::cout << "Имеем  " << datatrayn.rows << " примеров длинной вектора в " << datatrayn.cols << std::endl;
    auto trainResponse = traindata->getResponses();
    std::cout << "Имеем " << trainResponse.rows << " откликов размером в" << trainResponse.cols << std::endl;
    auto trainlables = traindata->getClassLabels();
    std::cout << "Имеем " << trainlables.rows << " меток размером в" << trainlables.cols << std::endl;
    // разделение выборок на тестовую и обучающую
    traindata->setTrainTestSplit(18000);
    auto ts = traindata->getTestSamples();
    std::cout << "Получено " << ts.rows << " тестовых примеров\n";
    // Нормальный Байсовский классификатор
    auto nbc = cv::ml::StatModel::train<cv::ml::NormalBayesClassifier>(traindata);
    return 0; // выход
}
