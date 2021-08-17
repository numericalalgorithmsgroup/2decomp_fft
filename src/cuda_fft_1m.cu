//=======================================================================
// This is part of the 2DECOMP&FFT library
// 
// 2DECOMP&FFT is a software framework for general-purpose 2D (pencil) 
// decomposition. It also implements a highly scalable distributed
// three-dimensional Fast Fourier Transform (FFT).
//
// Copyright (C) 2009-2021 Ning Li, the Numerical Algorithms Group (NAG)
//
//=======================================================================

// This contains CUDA code that compute multiple 1D FFTs on NVidia GPU

#include <stdio.h>
#include <stdlib.h>
#include "cufft.h"
#include "cuda.h"

extern "C" void fft_1m_r2c_(int *nx, int *m, cufftDoubleReal *h_a, cufftDoubleComplex *h_b)
{
  unsigned long size1 = sizeof(cufftDoubleReal) * (*nx)*(*m);
  unsigned long size2 = sizeof(cufftDoubleComplex) * (*nx/2+1)*(*m);
  cufftDoubleReal *d_ic = NULL;
  cufftDoubleComplex *d_oc = NULL; 
  cufftHandle plan;
  cudaMalloc((void **)&d_ic, size1);
  cudaMalloc((void **)&d_oc, size2);
  cudaMemcpy(d_ic, h_a, size1, cudaMemcpyHostToDevice);
  int dims[1] = {*nx};
  cufftPlanMany(&plan,1,dims,NULL,1,0,NULL,1,0,CUFFT_D2Z,*m);
  cufftExecD2Z(plan, d_ic, d_oc);
  cudaMemcpy(h_b, d_oc, size2, cudaMemcpyDeviceToHost);
  cudaFree(d_ic);
  cudaFree(d_oc);
  cufftDestroy(plan);
}


extern "C" void fft_1m_c2r_(int *nx, int *m, cufftDoubleComplex *h_a, cufftDoubleReal *h_b)
{
  unsigned long size1 = sizeof(cufftDoubleComplex) * (*nx/2+1)*(*m);
  unsigned long size2 = sizeof(cufftDoubleReal) * (*nx)*(*m);
  cufftDoubleComplex *d_ic = NULL;
  cufftDoubleReal *d_oc = NULL; 
  cufftHandle plan;
  cudaMalloc((void **)&d_ic, size1);
  cudaMalloc((void **)&d_oc, size2);
  cudaMemcpy(d_ic, h_a, size1, cudaMemcpyHostToDevice);
  int dims[1] = {*nx};
  cufftPlanMany(&plan,1,dims,NULL,1,0,NULL,1,0,CUFFT_Z2D,*m);
  cufftExecZ2D(plan, d_ic, d_oc);
  cudaMemcpy(h_b, d_oc, size2, cudaMemcpyDeviceToHost);
  cudaFree(d_ic);
  cudaFree(d_oc);
  cufftDestroy(plan);
}


extern "C" void fft_1m_c2c_(int *nx, int *m, cufftDoubleComplex *h_a, cufftDoubleComplex *h_b, int *sign)
{
  unsigned long size1 = sizeof(cufftDoubleComplex) * (*nx)*(*m);
  cufftDoubleComplex *d_ic = NULL;
  cufftDoubleComplex *d_oc = NULL; 
  cufftHandle plan;
  cudaMalloc((void **)&d_ic, size1);
  cudaMalloc((void **)&d_oc, size1);
  cudaMemcpy(d_ic, h_a, size1, cudaMemcpyHostToDevice);
  int dims[1] = {*nx};
  cufftPlanMany(&plan,1,dims,NULL,1,0,NULL,1,0,CUFFT_Z2Z,*m);
  cufftExecZ2Z(plan, d_ic, d_oc, *sign);
  cudaMemcpy(h_b, d_oc, size1, cudaMemcpyDeviceToHost);
  cudaFree(d_ic);
  cudaFree(d_oc);
  cufftDestroy(plan);
}
