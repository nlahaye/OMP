#include "EdgeDetector.hpp"

#include <opencv2/highgui/highgui.hpp>

using namespace cv;

void EdgeDetector::detectEdges()
{
	Mat tmpImage(this->imageData.size(), this->imageData.type());

	Canny(this->imageData, tmpImage, THRESH_LOW, THRESH_LOW*THRESH_RATIO, KERNEL_SIZE);	


	this->outImage.create(this->imageData.size(), this->imageData.type());
	this->outImage = Scalar::all(0);

	this->imageData.copyTo(this->outImage, tmpImage);
 
	std::vector<int> compression_params;
	compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
	compression_params.push_back(3);


	std::cerr << this->outImName << !this->imageData.data << "\n\n";
	

	imwrite(this->outImName, tmpImage, compression_params); 
}






