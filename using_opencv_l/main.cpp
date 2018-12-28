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
#include <mnist/mnist_utils.hpp>

void conversionMNISTToCVMAT(std::vector<std::vector<unsigned char>>& imgs, cv::Mat& res); // функция конвертирования примеров MNIST в матрицу
void convertionLablesToVector(std::vector<unsigned char>& lables, cv::Mat& res); // конвертирование меток MNIST в матрицу позиционных векторов

int main(int argc, const char * argv[]) {
    auto ds = mnist::read_dataset(); // читаем базу мнист
    mnist::binarize_dataset(ds);
    cv::Mat trainImages(ds.training_images.size(), 28 * 28, CV_32F), testImages(ds.test_images.size(), 28 * 28, CV_32F); // матрицы для выборок
    conversionMNISTToCVMAT(ds.training_images, trainImages); // преобразование тренеровочных изображений
    conversionMNISTToCVMAT(ds.test_images, testImages); // преобразование тестовых изображений
    cv::Mat trainLables(ds.training_labels.size(), 10, CV_32F), testLables(ds.test_labels.size(), 10, CV_32F); // матрицы позиционных векторов
    convertionLablesToVector(ds.training_labels, trainLables); // преобразование тренеровочных меток меток MNIST в матрицу позиционных векторов
    convertionLablesToVector(ds.test_labels, testLables); // преобразование тестовых меток MNIST в матрицу позиционных векторов
    std::cout << "Получено " << trainImages.rows << " примеров и " << trainLables.rows << " меток\n";
    auto traindata = cv::ml::TrainData::create(trainImages, cv::ml::ROW_SAMPLE, trainLables); // создание объекта тренеровочных данных
    // Нормальный Байсовский классификатор
    auto nbc = cv::ml::StatModel::train<cv::ml::NormalBayesClassifier>(traindata);
    return 0; // выход
}

void conversionMNISTToCVMAT(std::vector<std::vector<unsigned char>>& imgs, cv::Mat& res) {
    for (int i = 0; i < res.rows; i++) { // едем по столбцам
        for (int j = 0; j < imgs[i].size(); j++) { // едем по строкам
            res.at<float>(i, j) = imgs[i][j] / 255; // переписываем значения
        }
    }
}

void convertionLablesToVector(std::vector<unsigned char>& lables, cv::Mat& res) {
    for (int i = 0; i < res.rows; i++) { // едем по примерам
        res.at<float>(i, lables[i]) = 1; // ставим единицу в необходимом месте
    }
}
