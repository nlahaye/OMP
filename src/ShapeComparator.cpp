//#include <omp.h> 
#include "ShapeComparator.hpp"
#include <utility>
#include <vector>
#include <stdexcept>

 
arma::mat ShapeComparator::cvImagetoArmaPts(const cv::Mat& img)
{
	std::vector<std::pair<int,int> > pts;

	//Dont know dimensionality beforehand, cannot make parallel
	for(int i = 0; i < img.rows; ++i)
	{
		const double* imgRow = img.ptr<double>(i);
		for(int j = 0; j < img.cols; ++j)
		{
			if(imgRow[j] != 0)
			{
	
				std::pair<int,int> pt(i,j);
				pts.push_back(pt);
			}
		}
	}


	arma::mat ret(pts.size(), 2);
	
	#pragma omp parallel shared(ret)
	{
		#pragma omp for
		for(int i = 0; i < pts.size(); ++i)
        	{
			ret(i,0) = pts[i].first;
			ret(i,1) = pts[i].second;	
		}
	}

	return ret;
}

void ShapeComparator::runComparisons(std::vector<std::string>& imFnames)
{
	int i = 0;
	
	this->edgeDetectors.clear();
	this->thinPlateSplines.clear();


	this->edgeDetectors.reserve(imFnames.size());

	#pragma omp parallel for  shared(edgeDetectors, thinPlateSplines)
	{
		for(std::vector<std::string>::iterator it = imFnames.begin(); it < imFnames.end(); ++it)
		{
			i = it - imFnames.begin();
			EdgeDetector ed(imFnames[i]);
			this->edgeDetectors.assign(i, ed);
		}
	}

	//#pragma omp for
	for(i = 0; i < this->edgeDetectors.size() - 1; ++i)
	{
	
		#pragma omp parallel for shared(edgeDetectors, thinPlateSplines)
		{	
			for(int j = i; j < this->edgeDetectors.size(); ++j)
			{
				try
				{
					arma::mat shape1 = this->cvImagetoArmaPts(this->edgeDetectors[i].getOutputImage());
					arma::mat shape2 = this->cvImagetoArmaPts(this->edgeDetectors[j].getOutputImage());

					std::cerr << "TPS COMPARISON " << i << " " << j << "\n";			
					ThinPlateSpline tps(shape1, shape2);
					std::cerr << "\n\n\n";
	

					std::pair<int,int> p(i,j);
					this->thinPlateSplines.insert(std::pair<std::pair<int,int>,ThinPlateSpline>(p,tps));

				}
				catch(std::length_error)
				{
					continue;
				}
			}
		}
	}
}


/** Main function */
int main(int argc, char** argv)
{
	int numShapes = atoi(argv[1]);	

	std::string file_base(argv[2]);

	std::vector<std::string> fnames;

	//Most likely too little code to see any speedup
	for(int i = 0; i < numShapes; ++ i)
	{
		fnames.push_back(file_base + "_SHAPE_" + std::to_string(i) + ".png");
	}


	ShapeComparator sc(fnames);

}

