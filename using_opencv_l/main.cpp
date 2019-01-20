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
    // машина опорных векторов
/*    auto svm = cv::ml::SVM::create();
    svm->setType(cv::ml::SVM::C_SVC);
    svm->setKernel(cv::ml::SVM::LINEAR);
    svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, 1000, 1e-9));
    svm->trainAuto(traindata);
    float err = svm->calcError(traindata, true, outresponces);
    std::cout << "Ошибка предсказания " << err << std::endl; */
    // многослойный персептрон
/*    auto mlp = cv::ml::ANN_MLP::create();
    cv::Mat ls({datatrayn.cols, 8, datatrayn.cols});
    mlp->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, 100, 1e-6));
    mlp->setLayerSizes(ls);
    mlp->setActivationFunction(cv::ml::ANN_MLP::ActivationFunctions::SIGMOID_SYM);
    mlp->setTrainMethod(cv::ml::ANN_MLP::BACKPROP);
    cv::Mat newtrainresponce(trainResponse.rows, datatrayn.cols, CV_32F);
    newtrainresponce = newtrainresponce * 0;
    for (int i = 0; i < newtrainresponce.rows; i++) {
        newtrainresponce.at<float>(i, trainResponse.at<float>(i)) = 1;
    }
    mlp->train(datatrayn, cv::ml::ROW_SAMPLE, newtrainresponce);
    float err = mlp->calcError(traindata, true, newtrainresponce);
    std::cout << "ошибка предсказания " << err << std::endl; */
    // бустинг
/*    auto boosting = cv::ml::Boost::create();
    boosting->setBoostType(cv::ml::Boost::Types::REAL);
    boosting->setWeakCount(200);
    boosting->setWeightTrimRate(0);
    boosting->train(traindata);
    float err = boosting->calcError(traindata, true, outresponces);
    std::cout << "Ошибка для бустинга " << err << std::endl; */
    // случайный лес
    /*    auto randomForest = cv::ml::RTrees::create();
    randomForest->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, 100, 1e-6));
    cv::Mat newtrainresponce(trainResponse.rows, datatrayn.cols, CV_32F);
    randomForest->setActiveVarCount(4);
    randomForest->train(traindata);
    std::cout <<"Ошибка случайного леса " << randomForest->calcError(traindata, true, outresponces); */
    // к ближайших соседей
/*    auto kNearest = cv::ml::StatModel::train<cv::ml::KNearest>(traindata);
    std::cout << "Ошибка к соседей " << kNearest->calcError(traindata, true, outresponces) << " при " << kNearest->getDefaultK() << " соседях по умолчанию и " << std::endl; */
    return 0; // выход
}
