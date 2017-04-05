
#ifndef THIN_PLATE_SPLINE_HPP
#define THIN_PLATE_SPLINE_HPP

#include <stdio.h>
#include <armadillo>

using namespace arma;

class ThinPlateSpline
{

        public:

                /** Constructor */
                ThinPlateSpline(mat& origShape, mat& newShape)
		{
			this->oShape = origShape;
			this->nShape = newShape;

			this->createSpline();
		}

                /** Destructor */
                virtual ~ThinPlateSpline() {};


		virtual void setOrigShape(mat& origShape)
		{
			this->oShape = origShape;

			this->createSpline();
		}		

		virtual void setNewShape(mat& newShape)
		{
			this->nShape = newShape;

                        this->createSpline();
		}


		virtual void shapeReset(mat& origShape, mat& newShape) 
		{
			this->oShape = origShape;
                        this->nShape = newShape;

                        this->createSpline();
		}


		const mat& getOrigShape() {return this->oShape;};

		const mat& getNewShape() {return this->nShape;};

		const mat& getTranslationWeights() {return this->weights;};

		const double& getBendingEnergy() {return this->bendingEnergy;};

        protected:

		double bendingEnergy;

		mat oShape;
		mat nShape;
		mat weights;

		
		virtual void createSpline();
		virtual double basisFunc(double& h);

		virtual void calcBendingEnergy(mat& gamma_11);

};



#endif





