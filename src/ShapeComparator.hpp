#ifndef SHAPE_COMPARATOR_HPP
#define SHAPE_COMPARATOR_HPP

#include <stdio.h>
#include <map>
#include <vector>
#include <armadillo>

#include "EdgeDetector.hpp"
#include "ThinPlateSpline.hpp"


class ShapeComparator
{

	public:
		/** Constructor */
		ShapeComparator(std::vector<std::string>& images )
		{
			this->imageFnames = images;
		
			this->runComparisons(this->imageFnames);
		}

		void setImageFnames(std::vector<std::string>& images )
                {
                        this->imageFnames = images;
                
                        this->runComparisons(this->imageFnames);
                }

		
		const EdgeDetector& getEdgeDetector(int i) {return this->edgeDetectors[i];};

		const ThinPlateSpline& getThinPlateSpline(int i, int j)
		{
			std::pair<int, int> key;
			if(i < j)
			{
				key.first = i;
				key.second = j;	
			}
			else if(j > i)
			{
				key.first = j;
                                key.second = i;
			}
			else
			{
				//TODO Exception
			}
		
			return this->thinPlateSplines.at(key);
		}


		const std::string& getImageFname(int i) {return this->imageFnames[i];};		

	protected:

		std::vector<std::string> imageFnames;
		
		std::vector<EdgeDetector> edgeDetectors;
		std::map<std::pair<int, int>, ThinPlateSpline> thinPlateSplines;
		
		arma::mat cvImagetoArmaPts(const cv::Mat& img);


		void runComparisons(std::vector<std::string>& imFnames);
};


#endif
