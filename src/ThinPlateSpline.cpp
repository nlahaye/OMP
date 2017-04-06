//#include <omp.h>
#include <cmath>
#include <stdexcept>
#include "ThinPlateSpline.hpp"


using namespace arma;

double ThinPlateSpline::basisFunc(double& h)
{
	if(h == 0.0)
	{
		return h;
	}
	return h*h*log(h);
}

void ThinPlateSpline::createSpline()
{
	if(this->oShape.n_rows < this->nShape.n_cols)
	{
		throw std::length_error("We need at least 3 points to create a Spline");
	}

	if(this->oShape.n_cols != this->nShape.n_cols)
        {
                throw std::length_error("Original shape and new shape need to have the same dimensions and the same amount of control points");
        }

	if(this->oShape.n_rows != this->nShape.n_rows)
	{
		if(this->oShape.n_rows < this->nShape.n_rows && this->oShape.n_rows > 0)
		{
			mat nNShape(this->oShape.n_rows, this->oShape.n_cols);
			int index = 0;
			int increase = (int)floor((double)this->nShape.n_rows/(double)this->oShape.n_rows);
		
			#pragma omp parallel shared(nNShape, nShape)
			{
				#pragma omp for
				for(int i = 0; i < this->oShape.n_rows; ++i)
				{
					//#pragma omp for
					index = i * increase;
					for(int j = 0; j < this->oShape.n_cols; ++j)
					{
						nNShape(i,j) = this->nShape(index,j);
					}
				}
			}
			this->nShape = nNShape;
		}
		else if(this->nShape.n_rows < this->oShape.n_rows && this->nShape.n_rows > 0)
		{
			mat nOShape(this->nShape.n_rows, this->oShape.n_cols);
			int index = 0;
                        int increase = (int)floor((double)this->oShape.n_rows/(double)this->nShape.n_rows);
			
			#pragma omp parallel shared(nOShape, oShape)
                        {
				#pragma omp for
				for(int i = 0; i < this->nShape.n_rows; ++i)
                        	{
					//#pragma omp for
					index = i*increase;
					for(int j = 0; j < this->oShape.n_cols; ++j)
                        	        {
						nOShape(i,j) = this->oShape(index,j);
					}
                        	}
				this->oShape = nOShape;
			}
		}
		else
		{
			throw std::length_error("0 sized array\n\n");
		}
	}
	



        Mat<double> gamma_11(this->nShape.n_rows, this->nShape.n_rows);
	Mat<double> gamma(this->nShape.n_rows + this->nShape.n_cols, this->nShape.n_rows + this->nShape.n_cols);
	Mat<double> leftMtx(this->nShape.n_rows + this->nShape.n_cols, this->nShape.n_cols);



	//Cannot make this parallel because of the sequential addition of a.	
	double a = 0;
	for(int i = 0; i < nShape.n_rows-1; ++i)
	{
		for(int j = 1; j < nShape.n_rows; ++j)	
		{
			rowvec t1 = this->oShape.row(i);
			rowvec t2 =  this->oShape.row(j);		
			
			double h = norm(t1 - t2);
			double val = this->basisFunc(h);


			gamma_11(i,j) = gamma_11(j,i) = gamma(i,j) = gamma(j,i) = val;

			a += val;
			
		}	
	}


	a /= (nShape.n_rows*nShape.n_rows);
	
	#pragma omp parallel shared(gamma, leftMtx)
	{
		#pragma omp for
		for(int i = 0; i < nShape.n_rows; ++i)
		{
			gamma(i,  nShape.n_rows-1) = gamma(nShape.n_rows-1, i) = 1.0;
		
			//#pragma omp for	
			for(int j = 0; j < this->nShape.n_cols; ++j)
			{
				gamma(i,  nShape.n_rows + j) = gamma(nShape.n_rows + j, i) = this->oShape(i,j);
			
				leftMtx(i,j) = this->nShape(i,j);
			}

		}
	

		#pragma omp for
		for(int i = 0; i < this->nShape.n_cols; ++i)
		{
			//#pragma omp for
			for(int j = 0; j < this->nShape.n_cols; ++j)
			{
       				 gamma(nShape.n_rows + i, nShape.n_rows + j) = 0.0;
        			 leftMtx(nShape.n_rows+i, j) = 0.0;
			}		
		}
	}



	this->weights = solve(gamma, leftMtx);
	this->calcBendingEnergy(gamma_11);
	
}



void ThinPlateSpline::calcBendingEnergy(mat& gamma_11)
{

	mat w = this->weights.submat(0, 0, this->nShape.n_rows-1, this->nShape.n_cols-1);

	mat be = (w.t() * gamma_11) * w;
	this->bendingEnergy = be(0,0);

	std::cerr << "BENDING ENERGY " << this->bendingEnergy << "\n\n";

}






