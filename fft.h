//#ifndef FFT_H_
//#define FFT_H_

#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "fftw3.h"

//参数含义 x:实部 y：虚部 n：fft长度
//在默认含义下如果n>0表示做fft，否则做ifft
int fft(float* x , float* y , int n)
{
	// fft
	if(n>0)
	{
	const int fft_len = n ;
	fftw_complex in[fft_len], out[fft_len];
        fftw_plan p_fft;
        p_fft=fftw_plan_dft_1d(fft_len,in,out,FFTW_FORWARD,FFTW_MEASURE);

        for(int i=0 ;i<fft_len;i++)
        {
        	in[i][0] = x[i]; //real
        	in[i][1] = y[i]; //img
        }
        fftw_execute(p_fft);
        //writeback the after_fft data
        for(int i=0 ; i < fft_len ;i++)
        {
        	x[i] = out[i][0];//real
        	y[i] = out[i][1];//img
        }
        return 0;
	}
	// ifft
	else if(n<0)
	{
	const int ifft_len = -n ;
	fftw_complex in[ifft_len], out[ifft_len];
        fftw_plan p_ifft;
        p_ifft=fftw_plan_dft_1d(ifft_len,out,in,FFTW_BACKWARD,FFTW_MEASURE);
        //copy the data to the fftw input
        for (int i=0 ; i<ifft_len;i++)
        {
        	out[i][0] = x[i];
        	out[i][1] = y[i];
        }

        fftw_execute(p_ifft);
        // write the data to the space
        for(int i=0 ; i< ifft_len ;i++)
        {
        	x[i] = in[i][0]/ifft_len;
        	y[i] = in[i][1]/ifft_len;
        }
        return 0;
	}
}
