
#ifndef EDGE_DETECTOR_HPP
#define EDGE_DETECTOR_HPP

#include <stdio.h>
#include "opencv2/opencv.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"



static const int THRESH_LOW=100;
static const int THRESH_RATIO=3;
static const int KERNEL_SIZE=3;

using namespace cv;

class EdgeDetector
{

        public:

                /** Constructor */
                EdgeDetector(std::string fname)
		{
			this->imFName = fname;
		
			this->outImName = fname  + ".edg.jpg";

			this->imageData = imread(fname, IMREAD_GRAYSCALE);	

			this->detectEdges();
		}

                /** Destructor */
                virtual ~EdgeDetector() {};


		virtual void setImageFName(std::string fname)
		{

			this->imFName = fname;

                        this->imageData = imread(fname, IMREAD_GRAYSCALE);

                        this->detectEdges();
		}		


		const Mat& getImageData() {return this->imageData;};
		const std::string& getImageFName() {return this->imFName;};
		const std::string& getOutImageFName() {return this->outImName;};
		const Mat& getOutputImage() {return this->outImage;};

        protected:

		Mat imageData;
		Mat outImage;
		
		std::string imFName;
		std::string outImName;	
	
		virtual void detectEdges();

};



#endif





