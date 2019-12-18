/**
* @file This file is part of stockast.
*
* @section LICENSE
* MIT License
*
* Copyright (c) 2017-2019 Rajdeep Konwar
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in all
* copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.
*
* @section DESCRIPTION
* Stock Market Forecasting using parallel Monte-Carlo simulations
* (src:wikipedia) The Black–Scholes model assumes that the market consists of
* at least one risky asset, usually called the stock, and one riskless asset,
* usually called the money market, cash, or bond. The rate of return on the
* riskless asset is constant and thus called the risk-free interest rate.
**/


#include <helper_functions.h>   // helper functions for string parsing
#include <helper_cuda.h>        // helper functions CUDA error checking and initialization
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <chrono>
#include <random>
#include <memory>
#include <cuda_runtime.h>
#include "BlackScholes_kernel.cuh"

#define DIV_UP(a, b) ( ((a) + (b) - 1) / (b) )
static float kernelTime = 0.0f;

//! ---------------------------------------------------------------------------
//! Calculates volatility from ml_data.csv file
//! ---------------------------------------------------------------------------
float calcVolatility(float spotPrice, int timesteps)
{
	//! Open ml_data.csv in read-mode, exit on fail
	const std::string fileName("ml_data.csv");
	std::ifstream fp;
	fp.open(fileName, std::ifstream::in);
	if (!fp.is_open())
	{
		std::cerr << "Cannot open ml_data.csv! Exiting..\n";
		exit(EXIT_FAILURE);
	}

	std::string line;
	//! Read the first line then close file
	if (!std::getline(fp, line))
	{
		std::cerr << "Cannot read from ml_data.csv! Exiting..\n";
		fp.close();
		exit(EXIT_FAILURE);
	}
	fp.close();

	int i = 0, len = timesteps - 1;
	//std::unique_ptr<float[]> priceArr = std::make_unique<float[]>(timesteps - 1);
        float *priceArr;
        priceArr = (float *)malloc((timesteps - 1) * sizeof(float));
       
        std::istringstream iss(line);
	std::string token;

	//! Get the return values of stock from file (min 2 to 180)
	while (std::getline(iss, token, ','))
		priceArr[i++] = std::stof(token);

	float sum = spotPrice;
	//! Find mean of the estimated minute-end prices
	for (i = 0; i < len; i++)
		sum += priceArr[i];
	float meanPrice = sum / (len + 1);

	//! Calculate market volatility as standard deviation
	sum = powf((spotPrice - meanPrice), 2.0f);
	for (i = 0; i < len; i++)
		sum += powf((priceArr[i] - meanPrice), 2.0f);

	float stdDev = sqrtf(sum);
        free(priceArr);
	//! Return as percentage
	return (stdDev / 100.0f);
}

/** ---------------------------------------------------------------------------
Finds mean of a 2D array across first index (inLoops)
M is in/outLoops and N is timesteps
----------------------------------------------------------------------------*/
float * find2DMean(float **matrix, int numLoops, int timesteps)
{
	int j;
	float* avg = new float[timesteps];
	float sum = 0.0f;

	for (int i = 0; i < timesteps; i++)
	{
		/**
		A private copy of 'sum' variable is created for each thread.
		At the end of the reduction, the reduction variable is applied to
		all private copies of the shared variable, and the final result
		is written to the global shared variable.
		*/
//#pragma omp parallel for private(j) reduction(+:sum)
		for (j = 0; j < numLoops; j++)
		{
			sum += matrix[j][i];
		}

		//! Calculating average across columns
		avg[i] = sum / numLoops;
		sum = 0.0f;
	}

	return avg;
}

/** ---------------------------------------------------------------------------
Generates a random number seeded by system clock based on standard
normal distribution on taking mean 0.0 and standard deviation 1.0
----------------------------------------------------------------------------*/
float randGen(float mean, float stdDev)
{
	auto seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine generator(static_cast<unsigned int>(seed));
	std::normal_distribution<float> distribution(mean, stdDev);
	return distribution(generator);
}

//! ---------------------------------------------------------------------------
//! Simulates Black Scholes model
//! ---------------------------------------------------------------------------
float * runBlackScholesModel_CPU(float spotPrice, int timesteps, float riskRate, float volatility)
{
	float  mean = 0.0f, stdDev = 1.0f;			//! Mean and standard deviation
	float  deltaT = 1.0f / timesteps;			//! Timestep
	//std::unique_ptr<float[]> normRand = std::make_unique<float[]>(timesteps - 1);	//! Array of normally distributed random nos.
        float *normRand;
        normRand = (float *)malloc((timesteps - 1) * sizeof(float));
	float* stockPrice = new float[timesteps];	//! Array of stock price at diff. times
	stockPrice[0] = spotPrice;					//! Stock price at t=0 is spot price

	//! Populate array with random nos.
	for (int i = 0; i < timesteps - 1; i++)
		normRand[i] = randGen(mean, stdDev);

        //print data
        for (int i = 0; i < timesteps-1; i++)
        {
            printf("stockPrice[%d] = %f\t", i, stockPrice[i]);
            printf("normRand[%d] = %f\n", i, normRand[i]);
        }
        printf("stockPrice[%d] = %f\n", (timesteps-1), stockPrice[(timesteps-1)]);

	//! Apply Black Scholes equation to calculate stock price at next timestep
	for (int i = 0; i < timesteps - 1; i++)
		stockPrice[i + 1] = stockPrice[i] * exp(((riskRate - (powf(volatility, 2.0f) / 2.0f)) * deltaT) + (volatility * normRand[i] * sqrtf(deltaT)));
        
        for (int i = 0; i < timesteps-1; i++)
        {
            printf("stockPrice[%d] = %f\t", i, stockPrice[i]);
            printf("normRand[%d] = %f\n", i, normRand[i]);
        }
        printf("stockPrice[%d] = %f\n", (timesteps-1), stockPrice[(timesteps-1)]);
        free(normRand);
	return stockPrice;
}


float * runBlackScholesModel_GPU(float spotPrice, int timesteps, float riskRate, float volatility, int inLoops)
{
        float  mean = 0.0f, stdDev = 1.0f;                      //! Mean and standard deviation
        float  deltaT = 1.0f / timesteps;                       //! Timestep
        int paths_num = inLoops;
        float **h_stockPrice, **h_normRand;
        float *h_stockPriceData, *h_normRandData;
        float **d_stockPrice, **d_normRand;
        float *d_stockPriceData, *d_normRandData;

        printf("allocating host  memory...\n");
        h_stockPrice = (float **)malloc(paths_num * sizeof(float *)); 
        h_normRand = (float **)malloc(paths_num * sizeof(float *));
        h_stockPriceData = (float *)malloc(paths_num * timesteps * sizeof(float));
        h_normRandData = (float *)malloc(paths_num * (timesteps-1) * sizeof(float));

        printf("...allocating GPU memory for options.\n");
        checkCudaErrors(cudaMalloc((void **)&d_stockPrice, paths_num * sizeof(float *)));
        checkCudaErrors(cudaMalloc((void **)&d_normRand, paths_num *  sizeof(float *)));
        checkCudaErrors(cudaMalloc((void **)&d_stockPriceData, paths_num * timesteps * sizeof(float)));
        checkCudaErrors(cudaMalloc((void **)&d_normRandData, paths_num * (timesteps-1) * sizeof(float)));

        printf("Initializing data...\n");
        for(int i = 0; i < paths_num; i++){
            h_stockPriceData[i * timesteps] = spotPrice;
            for (int j = 1; j < timesteps; j++){
                h_stockPriceData[i * timesteps + j] = 0.0f;
                h_normRandData[i*(timesteps-1)+j-1] = randGen(mean, stdDev);
            }
        }
        //print data
        for(int i = 0; i < paths_num; i++){
           for(int j = 0; j < timesteps-1; j++){
                printf("h_stockPrice[%d][%d] = %f\t", i, j, h_stockPriceData[i * timesteps + j]);
                printf("h_normRand[%d][%d] = %f\n", i, j, h_normRandData[i * (timesteps-1) + j]);
            }
            printf("h_stockPrice[%d][%d] = %f\n", i, (timesteps-1), h_stockPriceData[i * timesteps -1]);
        }
        // host pointer map with device data pointer
        for(int i = 0; i < paths_num; i++){
            h_stockPrice[i] = d_stockPriceData + i * timesteps;
            h_normRand[i] = d_normRandData + i * (timesteps-1);
        }
        printf("...copying input data to GPU mem.\n");
        //Copy options data to GPU memory for further processing
        checkCudaErrors(cudaMemcpy(d_stockPrice,  h_stockPrice, paths_num * sizeof(float *), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_normRand, h_normRand, paths_num * sizeof(float *), cudaMemcpyHostToDevice));

        checkCudaErrors(cudaMemcpy(d_stockPriceData,  h_stockPriceData, paths_num * timesteps * sizeof(float), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_normRandData, h_normRandData, paths_num * (timesteps- 1) * sizeof(float), cudaMemcpyHostToDevice));
        StopWatchInterface *hTimer = NULL;
        sdkCreateTimer(&hTimer);
        printf("Data init done.\n\n");

        sdkResetTimer(&hTimer);
        sdkStartTimer(&hTimer);

        dim3 dimblock(paths_num);
        dim3 dimGrid(1);
        runBlackScholesModel_kernel<<<dimGrid, dimblock>>>(
            (float *)d_stockPriceData,
            (float *)d_normRandData,
            timesteps,
            riskRate,
            volatility,
            deltaT,
            paths_num
        );
        //runBlackScholesModel_kernel_2D<<<dimGrid, dimblock>>>(
        //   (float **)d_stockPrice,
        //   (float **)d_normRand,
        //   timesteps,
        //   riskRate,
        //   volatility,
        //   deltaT,
        //   paths_num
        //);
        getLastCudaError("BlackScholesGPU() execution failed\n"); 
        checkCudaErrors(cudaDeviceSynchronize());
        printf("\nReading back GPU results...\n");
        checkCudaErrors(cudaMemcpy(h_stockPriceData, d_stockPriceData, paths_num * timesteps * sizeof(float), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(h_normRandData, d_normRandData, paths_num * (timesteps-1) * sizeof(float), cudaMemcpyDeviceToHost));
        sdkStopTimer(&hTimer);
        double gpuTime;
        gpuTime = sdkGetTimerValue(&hTimer);
        printf("Options count             : %i     \n", paths_num * timesteps);
        kernelTime = gpuTime / (paths_num * timesteps);
        printf("BlackScholesGPU() time    : %f msec\n", kernelTime);

        //print data
        for(int i = 0; i < paths_num; i++){
           for(int j = 0; j < timesteps-1; j++){
                printf("h_stockPrice[%d][%d] = %f\t", i, j, h_stockPriceData[i * timesteps + j]);
                printf("h_normRand[%d][%d] = %f\n", i, j, h_normRandData[i * (timesteps-1) + j]);
            }
            printf("h_stockPrice[%d][%d] = %f\n", i, (timesteps-1), h_stockPriceData[i * timesteps -1]);
        }

        printf("...releasing CPU and GPU memory.\n");
        free(h_normRand);
        free(h_stockPrice);
        free(h_normRandData);
        checkCudaErrors(cudaFree(d_stockPrice));
        checkCudaErrors(cudaFree(d_stockPriceData));
        checkCudaErrors(cudaFree(d_normRand));
        checkCudaErrors(cudaFree(d_normRandData));
        return h_stockPriceData;
}





//! ---------------------------------------------------------------------------
//! Main function
//! ---------------------------------------------------------------------------
int main(int argc, char **argv)
{
	clock_t t = clock();

	int inLoops = 8;		//! Inner loop iterations
	int outLoops = 1;	//! Outer loop iterations
	int timesteps = 180;	//! Stock market time-intervals (min)

        findCudaDevice(argc, (const char **)argv);

        float *result;
	//! Matrix for stock-price vectors per iteration
        float **stock = new float*[inLoops];
        for (int i = 0; i < inLoops; i++){
                stock[i] = new float[timesteps];
        }

	//! Matrix for mean of stock-price vectors per iteration
	//float **avgStock = new float*[outLoops];
	//for (int i = 0; i < outLoops; i++)
	//	avgStock[i] = new float[timesteps];
        float *avgStock = new float[timesteps];

	//! Vector for most likely outcome stock price
	float *optStock = new float[timesteps];
	
        float riskRate = 0.001f;	//! Risk free interest rate (%)
	float spotPrice = 100.0f;	//! Spot price (at t = 0)

	//! Market volatility (calculated from ml_data.csv)
	float volatility = calcVolatility(spotPrice, timesteps);
        //float volatility = 0.1;
	//! Welcome message
	std::cout << "--Welcome to Stockast: Stock Forecasting Tool--\n";
	std::cout << "  Copyright (c) 2017-2019 Rajdeep Konwar\n\n";
	std::cout << "  Using market volatility = " << volatility << std::endl;
	int i;
	for (i = 0; i < outLoops; i++)
	{
		/**
		Using Black Scholes model to get stock price every iteration
		Returns data as a column vector having rows=timesteps
		*/
		//for (int j = 0; j < inLoops; j++){
		//    stock[j] = runBlackScholesModel_CPU(spotPrice, timesteps, riskRate, volatility);
                //}	

                // run inloops BlackScholesModel using gpu Parallel
                result = runBlackScholesModel_GPU(spotPrice, timesteps, riskRate, volatility, inLoops);
                for(int i = 0; i < inLoops; i++){
                    stock[i] = result + i * timesteps;
                }
                //! Stores average of all estimated stock-price arrays
		avgStock = find2DMean(stock, inLoops, timesteps);
	}

	//! Average of all the average arrays
	//optStock = find2DMean(avgStock, outLoops, timesteps);

	//! Write optimal outcome to disk
	std::ofstream fp;
        printf("Write optimal outcome to disk\n");
	fp.open("opt.csv", std::ofstream::out);
	if (!fp.is_open())
	{
		std::cerr << "Couldn't open opt.csv! Exiting..\n";
		return EXIT_FAILURE;
	}
        for (int i=0; i < timesteps; i++){
            fp << stock[0][i];
	    for (int j=1; j < inLoops; j++){
		fp << "," << stock[j][i];
            }
            fp << "," << avgStock[i];
            fp << "\n";
        }
	fp.close();

	delete[] stock;
        //delete[] totalStock;

        delete[] avgStock;
	delete[] optStock;
        free(result);

	t = clock() - t;
        printf("Options count             : %i     \n", timesteps*inLoops);
        printf("BlackScholesGPU() time    : %f msec\n", kernelTime);
	std::cout << " done!\n  Time taken = " << static_cast<float>(t / CLOCKS_PER_SEC) << "s" << std::endl;
	return EXIT_SUCCESS;
}
