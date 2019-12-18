////////////////////////////////////////////////////////////////////////////////
//Process BlackScholesModel options on GPU using 1D Array
////////////////////////////////////////////////////////////////////////////////
__global__ void runBlackScholesModel_kernel(
            float *d_stockPrice,
            float *d_normRand,
            int timesteps,
            float riskRate,
            float volatility,
            float deltaT,
            int paths_num
        ){
    int threadId = blockDim.x * blockIdx.x + threadIdx.x;
    if(threadId < paths_num){
        int stockPriceNum = threadId * timesteps;
        int normRandNum = threadId * (timesteps-1);
        for(int i = 0; i < (timesteps - 1); i++){
            float A = volatility * d_normRand[normRandNum + i] * sqrtf(deltaT);
            float B = (riskRate - ((volatility * volatility) / 2.0f)) * deltaT;
            float C = __expf(A + B);
            d_stockPrice[stockPriceNum + i + 1] = d_stockPrice[stockPriceNum + i] * C;
        }
    }
}


////////////////////////////////////////////////////////////////////////////////
//Process BlackScholesModel options on GPU using 2D Array
////////////////////////////////////////////////////////////////////////////////
__global__ void runBlackScholesModel_kernel_2D(
            float **d_stockPrice,
            float **d_normRand,
            int timesteps,
            float riskRate,
            float volatility,
            float deltaT,
            int paths_num
        ){
    int threadId = blockDim.x * blockIdx.x + threadIdx.x;
    if(threadId < paths_num){
        for(int i = 0; i < (timesteps - 1); i++){
            float A = volatility * d_normRand[threadId][i] * sqrtf(deltaT);
            float B = (riskRate - ((volatility * volatility) / 2.0f)) * deltaT;
            float C = __expf(A + B);
            d_stockPrice[threadId][i+1] = d_stockPrice[threadId][i] * C;
        }
    }
}

