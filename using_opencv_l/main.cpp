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
    auto traindata = cv::ml::TrainData::loadFromCSV("./letter-recognition.data.csv", 0, 0);
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
    // многослойный персептрон
    auto mlp = cv::ml::ANN_MLP::create();
    cv::Mat ls({datatrayn.cols, 50, trainlables.rows});
    mlp->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, 100, 1e-6));
    mlp->setLayerSizes(ls);
    mlp->setActivationFunction(cv::ml::ANN_MLP::ActivationFunctions::SIGMOID_SYM);
    mlp->setTrainMethod(cv::ml::ANN_MLP::BACKPROP);
    cv::Mat newtrainresponce(trainResponse.rows, trainlables.rows, CV_32F);
    newtrainresponce = newtrainresponce * 0;
    for (int i = 0; i < newtrainresponce.rows; i++) {
        newtrainresponce.at<float>(i, trainResponse.at<float>(i)) = 1;
    }
    mlp->train(datatrayn, cv::ml::ROW_SAMPLE, newtrainresponce);
    float err = mlp->calcError(traindata, true, newtrainresponce);
    std::cout << "ошибка предсказания " << err << std::endl;
    return 0; // выход
}
