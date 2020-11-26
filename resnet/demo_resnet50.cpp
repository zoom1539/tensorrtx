#include "class_resnet50.hpp"
#include <time.h>

void main()
{
	Resnet50 resnet50;

	//
	std::string path_wts = "resnet50.wts";
	int num_class = 2;

	std::fstream fs;
	fs.open("resnet50.engine", std::ios::in);
	if (!fs)
	{
		resnet50.build_engine(path_wts, num_class);
	}

	resnet50.init(num_class);

	//
	cv::Mat img = cv::imread("F:/zmh/val_img_resnet/test/243.bmp");

	for (int i = 0; i < 10; i++)
	{
		clock_t time = clock();
		int class_id;
		resnet50.classify(img, class_id);

		std::cout << "cost time: " << clock() - time << "ms" << std::endl;

		std::cout << "class_id: " << class_id << std::endl;

		std::cin.get();
	}
	
}