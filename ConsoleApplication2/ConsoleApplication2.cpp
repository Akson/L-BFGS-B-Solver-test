// ConsoleApplication2.cpp : main project file.

#include "stdafx.h"
#include "../L-BFGS/lbfgsb.h"
#include "../L-BFGS/lbfgsbcuda.h"
#include <iostream>

using namespace System;
using namespace std;

real stpscal;
cublasHandle_t cublasHd;
int m_iTotalTreatmentSpots;

real *d_weights;
int *d_NumberBounds;
real *d_LowerBounds;
real *d_UpperBounds;

real maxWeight;
real maxWeightIDX;

void InitializeWeights();
void CallLBFGS();


int main(array<System::String ^> ^args)
{
	cublasCreate_v2(&cublasHd);

    Console::WriteLine(L"Starting LBFGS-B Test...");

	m_iTotalTreatmentSpots = 11340;
	stpscal = 0.898f;

	maxWeight = -FLT_MAX;

	InitializeWeights();
	CallLBFGS();

	if (maxWeight > 1.0)
		printf("Test FAILED!: max weight error = %.5g at index %d\n", maxWeight, maxWeightIDX);
	else
		printf("Test PASSED: max weight error = %.5g at index %d\n", maxWeight, maxWeightIDX);

	system("pause");

	cublasDestroy_v2(cublasHd);

    return 0;
}

void InitializeWeights()
{
	real *weights = new real[m_iTotalTreatmentSpots];
	int *numberBounds = new int[m_iTotalTreatmentSpots];
	real *lowerBounds = new real[m_iTotalTreatmentSpots];
	real *upperBounds = new real[m_iTotalTreatmentSpots];

	for(int i = 0; i < m_iTotalTreatmentSpots; i++)
	{
		weights[i] = 1.0; 
		numberBounds[i] = 1;		// weight only has a lower bound
		lowerBounds[i] = 0.0 ;		// lower bound is zero to assure non-negative weights
		upperBounds[i] = 100000.0;
	}

	checkCudaErrors(cudaMalloc(&d_weights, m_iTotalTreatmentSpots * sizeof(real)));
	checkCudaErrors(cudaMemcpy(d_weights, weights, m_iTotalTreatmentSpots * sizeof(real), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc(&d_NumberBounds, m_iTotalTreatmentSpots * sizeof(int)));
	checkCudaErrors(cudaMemcpy(d_NumberBounds, numberBounds, m_iTotalTreatmentSpots * sizeof(int), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc(&d_LowerBounds, m_iTotalTreatmentSpots * sizeof(real)));
	checkCudaErrors(cudaMemcpy(d_LowerBounds, lowerBounds, m_iTotalTreatmentSpots * sizeof(real), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc(&d_UpperBounds, m_iTotalTreatmentSpots * sizeof(real)));
	checkCudaErrors(cudaMemcpy(d_UpperBounds, upperBounds, m_iTotalTreatmentSpots * sizeof(real), cudaMemcpyHostToDevice));

	delete[] weights;
	delete[] numberBounds;
	delete[] lowerBounds;
	delete[] upperBounds;
}

void CostAndGradient(real *x, real &f, real *g, const cudaStream_t& stream)
{
	cudaError_t err = cudaGetLastError();
	if(err != cudaSuccess)
	{
		printf("!! Cuda error at CostAndGradient: %s\n", cudaGetErrorString(err));
	}

	static int passNum = 0;
	string line;

	if (passNum > 12)
		return;

	printf("iteration: %d\n", passNum);

	//******** Test weights

	real *wts = new real[m_iTotalTreatmentSpots];
	cudaMemcpyAsync(wts, x, m_iTotalTreatmentSpots * sizeof(real), cudaMemcpyDeviceToHost, stream);
	if(wts[0] > 1000.0)
		printf("%.3g\n", wts[0]);

	std::ifstream fileWt;
	string fileNameWt = "..//Data//testWeights" + std::to_string(passNum) + ".txt";
	fileWt.open(fileNameWt);

	if (!fileWt.is_open())
		printf("can't open file\n");

	real *testWts = new real[m_iTotalTreatmentSpots];
	for (int i=0; i<m_iTotalTreatmentSpots; i++)
	{
		testWts[i] = 0.0;

		float val;
		getline(fileWt, line);
		sscanf_s(line.c_str(), "%f\n", &val);
		testWts[i] = val;

		if (wts[i] > maxWeight && testWts[i] > 0.0)
		{
			maxWeight = (wts[i] - testWts[i])/testWts[i];
			maxWeightIDX = i;
		}

		//if (testWts[i] != wts[i] && i < 10)
		//	printf("%d wt: %.5g diff: %.5g\n", i, wts[i], (wts[i] - testWts[i])/testWts[i]);
	}

	fileWt.close();

	//********* Read Cost Function

	std::ifstream fileCF;
	fileCF.open("..//Data//costFunc.txt");


	for (int i=0; i<passNum+1; i++) // need to skip lines
		getline(fileCF, line);

	float cost;
	sscanf_s(line.c_str(), "%f\n", &cost);
	f = cost;
	fileCF.close();

	//printf("cost: %.5g\n", cost);

	//********* Read Gradient

	std::ifstream fileGrad;
	string fileName = "..//Data//testGrad" + std::to_string(passNum) + ".txt";
	fileGrad.open(fileName);

	real *grad = new real[m_iTotalTreatmentSpots];
	for (int i=0; i<m_iTotalTreatmentSpots; i++)
	{
		float val;
		getline(fileGrad, line);
		sscanf_s(line.c_str(), "%f\n", &val);
		grad[i] = val;
	}
	fileGrad.close();

	cudaMemcpyAsync(g, grad, m_iTotalTreatmentSpots * sizeof(real), cudaMemcpyHostToDevice, stream);
	delete [] grad;


	passNum++;
}

void CallLBFGS()
{
	real *d_testWts;
	real *d_testGrad;
	cudaMalloc(&d_testWts, m_iTotalTreatmentSpots * sizeof(real));
	cudaMalloc(&d_testGrad, m_iTotalTreatmentSpots * sizeof(real));

	real *testWts = new real[m_iTotalTreatmentSpots];
	for(int i = 0; i < m_iTotalTreatmentSpots; i++)
		testWts[i] = 1.0;

	cudaMemcpyAsync(d_testWts, testWts, m_iTotalTreatmentSpots*sizeof(real), cudaMemcpyHostToDevice, 0);
	delete[]  testWts;

	const real epsg = EPSG;
	const real epsf = EPSF;
	const real epsx = EPSX;
	const int maxIterations = 10;//MAXITS;
	int M = 7;  // Number of corrections in the BFGS scheme of Hessian approximation update. Recommended value:  3<=M<=7.
	int info; // return value 

	lbfgsbminimize(m_iTotalTreatmentSpots, M, d_weights, 
		epsg, epsf, epsx, maxIterations, 
		d_NumberBounds, d_LowerBounds, d_UpperBounds, 
		info, 
		&CostAndGradient);
}
