#include <iostream>
#include <string>
#include <math.h>
#include <iomanip>
#include <fstream>
#include "wav.h"
#include "fft.h"
#include "time.h"

using namespace std;

#define PI 3.14159265358979323846
#define FFT_LENGTH 512
#define FRAME_LENGTH 400
#define FRAME_OFFSET 160
#define SAMPLE_RATE__ 16000
#define BAND_LENGTH 257
#define MU 0.8
#define NTAPS 10

// set constants
const float fl = FRAME_OFFSET / SAMPLE_RATE__;
const int smoothscale = fl / 0.01;
const float spksmoothfactor = pow(0.985, smoothscale);
const float adaptsignalthreshold = 0.001;
const float resPowSmoothFactor = pow(0.93, smoothscale);
const float transferThresERLE = 2.0;
const float transferThresPowRatio = 3.0;
int k;

// Inverse Discrete Cosine Transform
void IDCT( float ** input, float ** output, int row, int col );
// Discrete Cosine Transform
void DCT( float ** input, float ** output, int row, int col );
// Pow 2 and sum for an array
float pow2sum(float **input, int length);
// Flip the 2d array upside down
void flip_up_down(float **input, int row, int col);
// Get the same result as the buffer() with 3 parameters in matlab
int buffer_matlab_float(float* input, float* output, int len_in, int buf_len, int pre_overlap);
// int format of buffer_matlab
int buffer_matlab_int(int* input, int* output, int len_in, int buf_len, int pre_overlap);
// fft, x for real part, y for imag part, n for length
//int fft(float* x , float* y , int n);
// input the spec of each frame(mic and spk), use them to update the filter banks
float AdaptFilter(float * mic_spec_real, float * mic_spec_imag, float * spk_spec_real, float * spk_spec_imag,\
				float * mic_out_real, float * mic_out_imag);
// complex times
void complex_times(float real1, float imag1, float real2, float imag2, float* real_result, float*imag_result);

// arrays store the data for filter banks
float * bestResidual_real, * bestResidual_imag;
float * bestResidualPow;
float * spkHistoryPower;
float * spkSmoothedHistPower;
float * spkHistory_real, * spkHistory_imag;
float * micout_real, * micout_imag;
float * h_front_real, * h_front_imag;
float * h_back_real, * h_back_imag;
float * avg_micpower, * avg_respowfront, * avg_respowback;

int main(int argc, char *argv[])
{
	// reading the reference wave and the microphone wave, get the number of frames
	// asumming the bit per sample is 16 for all
	const char* mic_filename = argv[1];
	const char* spk_filename = argv[2];
	const char* out_filename = argv[3];

	WavReader spk_reader(spk_filename);
	WavReader mic_reader(mic_filename);
	float *spk_pcm = spk_reader.Data();
	float *mic_pcm = mic_reader.Data();

	int spk_len = spk_reader.NumSample();
	int mic_len = mic_reader.NumSample();
	cout << "Sample Rate: " << spk_reader.SampleRate() << endl;
	cout << "Number of sample points of reference wav: " << spk_len << endl;
	cout << "Number of sample points of microphone wav: " << mic_len << endl;
	cout << "Reference Bits per sample: " << spk_reader.BitsPerSample() << endl;
	cout << "Microphone Bits per sample: " << mic_reader.BitsPerSample() << endl;

	// normalize the data in wav
	for(int i = 0 ; i < spk_len; i++)
		spk_pcm[i] = spk_pcm[i] / 32767;
	for(int i = 0 ; i < mic_len; i++)
		mic_pcm[i] = mic_pcm[i] / 32767;

	int mic_frame_number = mic_len/FRAME_OFFSET;
	int spk_frame_number = spk_len/FRAME_OFFSET;
	int frame_number = mic_frame_number > spk_frame_number ? spk_frame_number : mic_frame_number;

	int invalid_frame = FRAME_LENGTH / FRAME_OFFSET;
	frame_number -= invalid_frame;
	cout << "frame_number: " << frame_number << endl;

	// hamming window function
	float* h = new float[FRAME_LENGTH]();
	float a = 2 * PI / (FRAME_LENGTH - 1);
	for(int i = 0; i < FRAME_LENGTH; i++)
    	h[i] = 0.54 - 0.46 * cos(a * (i - 1));

	// the update of the filter bank need a lot of complex arrays, here calloced for them all
	bestResidual_real = new float[BAND_LENGTH]();
	bestResidual_imag = new float[BAND_LENGTH]();

	bestResidualPow = new float[BAND_LENGTH]();
	spkHistoryPower = new float[BAND_LENGTH]();
	spkSmoothedHistPower = new float[BAND_LENGTH]();

	spkHistory_real = new float[NTAPS*BAND_LENGTH]();
	spkHistory_imag = new float[NTAPS*BAND_LENGTH]();

	h_front_real = new float[NTAPS*BAND_LENGTH]();
	h_front_imag = new float[NTAPS*BAND_LENGTH]();

	h_back_real = new float[NTAPS*BAND_LENGTH]();
	h_back_imag = new float[NTAPS*BAND_LENGTH]();

	avg_micpower = new float[BAND_LENGTH]();
	avg_respowfront = new float[BAND_LENGTH]();
	avg_respowback = new float[BAND_LENGTH]();

	// main part
	float* erle = new float[frame_number]();// 1998

	float* current_spk = new float[FRAME_LENGTH]();
	float* current_mic = new float[FRAME_LENGTH]();

	float* spk_real = new float[FFT_LENGTH]();
	float* spk_imag = new float[FFT_LENGTH]();
	float* mic_real = new float[FFT_LENGTH]();
	float* mic_imag = new float[FFT_LENGTH]();

	// temporary memory for fft
	float* real_array = new float[FFT_LENGTH]();
	float* imag_array = new float[FFT_LENGTH]();

	// memory for output ifft
	float* out_real = new float[BAND_LENGTH]();
	float* out_imag = new float[BAND_LENGTH]();
	// memory for output data
	float* out_spec = new float[FRAME_LENGTH]();
	float* out_pcm = new float[spk_len]();

	clock_t start, finish;

	// main recycle
	for(k = 0; k < frame_number; k++)
	{
		start = clock();
		// pick out the current data
		memcpy(current_spk, spk_pcm+k*FRAME_OFFSET, 4*FRAME_LENGTH);
		memcpy(current_mic, mic_pcm+k*FRAME_OFFSET, 4*FRAME_LENGTH);
		// apply the hamming window function
		for(int i = 0; i < FRAME_LENGTH; i++)
		{
			current_spk[i] *= h[i];
			current_mic[i] *= h[i];
		}
		// real_array: front 400 is equal to current data, 401 to 512 is 0
		// spk fft
		for(int i = 0; i < FFT_LENGTH; i++)
		{
			real_array[i] = 0.0;
			imag_array[i] = 0.0;
		}
		memcpy(real_array, current_spk, 4*FRAME_LENGTH);
		fft(real_array, imag_array, FFT_LENGTH);
		memcpy(spk_real, real_array, 4*FFT_LENGTH);
		memcpy(spk_imag, imag_array, 4*FFT_LENGTH);
		// mic fft
		for(int i = 0; i < FFT_LENGTH; i++)
		{
			real_array[i] = 0.0;
			imag_array[i] = 0.0;
		}
		memcpy(real_array, current_mic, 4*FRAME_LENGTH);
		fft(real_array, imag_array, FFT_LENGTH);
		memcpy(mic_real, real_array, 4*FFT_LENGTH);
		memcpy(mic_imag, imag_array, 4*FFT_LENGTH);

		// send the spec 512 data of mic and spk into Adaptfilter, will be cutted into 257 inside Adaptfilter
		erle[k] = AdaptFilter(mic_real, mic_imag, spk_real, spk_imag, out_real, out_imag);

		float* out_full_real = new float[FFT_LENGTH]();
		float* out_full_imag = new float[FFT_LENGTH]();

		memcpy(out_full_real, out_real, 4*BAND_LENGTH);
		memcpy(out_full_imag, out_imag, 4*BAND_LENGTH);

		// Symmetric Complex Conjugation
		for(int i = BAND_LENGTH; i < FFT_LENGTH; i++)
		{
			out_full_real[i] = out_real[FFT_LENGTH-i];
			out_full_imag[i] = -1 * out_imag[FFT_LENGTH-i];
		}

		fft(out_full_real, out_full_imag, -FFT_LENGTH);
		memcpy(out_spec, out_full_real, 4*FRAME_LENGTH);

		// overlap adding with applying window function
		for(int j = 0; j < FRAME_LENGTH; j++)
		{
			out_pcm[k*FRAME_OFFSET+j] += out_spec[j] * h[j];
		}
		delete [] out_full_imag;
		delete [] out_full_real;
		finish = clock();
		double duration = (double)(finish - start) / CLOCKS_PER_SEC;
		cout << duration << endl;
	}
	cout << endl;

	for(int i = 0; i < spk_len; i++)
		out_pcm[i] *= 32767;

	WavWriter writer(out_pcm, spk_reader.NumSample(), spk_reader.NumChannel(),spk_reader.SampleRate(), spk_reader.BitsPerSample());
	writer.Write(out_filename);

	// delete the memory of the pcm data
	delete [] h;
	delete [] current_spk;
	delete [] current_mic;

	delete [] spk_real;
	delete [] spk_imag;
	delete [] mic_real;
	delete [] mic_imag;

	// delete the memory for fft
	delete [] real_array;
	delete [] imag_array;

	// delete the memory of global arrays
	delete [] bestResidual_real;
	delete [] bestResidual_imag;
	delete [] bestResidualPow;
	delete [] spkHistoryPower;
	delete [] spkSmoothedHistPower;
	delete [] spkHistory_real;
	delete [] spkHistory_imag;
	delete [] micout_real;
	delete [] micout_imag;
	delete [] h_front_real;
	delete [] h_front_imag;
	delete [] h_back_real;
	delete [] h_back_imag;
	delete [] avg_micpower;
	delete [] avg_respowfront;
	delete [] avg_respowback;

	// delete the memory applied in main
	delete [] erle;

	return 0;
}

// for each frame
float AdaptFilter(float * mic_spec_real, float * mic_spec_imag, float * spk_spec_real, float * spk_spec_imag,\
				float * mic_out_real, float * mic_out_imag)
{
	// choose the use bands
	int use_bands_offset = 0;
	float* cmic_real_buffer = new float[BAND_LENGTH]();
	float* cmic_imag_buffer = new float[BAND_LENGTH]();
	float* cspk_real_buffer = new float[BAND_LENGTH]();
	float* cspk_imag_buffer = new float[BAND_LENGTH]();
	memcpy(cmic_real_buffer, mic_spec_real+use_bands_offset, BAND_LENGTH*4);
	memcpy(cmic_imag_buffer, mic_spec_imag+use_bands_offset, BAND_LENGTH*4);
	memcpy(cspk_real_buffer, spk_spec_real+use_bands_offset, BAND_LENGTH*4);
	memcpy(cspk_imag_buffer, spk_spec_imag+use_bands_offset, BAND_LENGTH*4);

	// memory for temporary array
	float* micpower = new float[BAND_LENGTH]();
	float* echoEstimate_real = new float[BAND_LENGTH]();
	float* echoEstimate_imag = new float[BAND_LENGTH]();
	float* echoResidualFront_real = new float[BAND_LENGTH]();
	float* echoResidualFront_imag = new float[BAND_LENGTH]();
	float* echoResidualBack_real = new float[BAND_LENGTH]();
	float* echoResidualBack_imag = new float[BAND_LENGTH]();
	float* res_pow_front = new float[BAND_LENGTH]();
	float* res_pow_back = new float[BAND_LENGTH]();
	float* erlefront = new float[BAND_LENGTH]();
	float* erleback = new float[BAND_LENGTH]();

	// matlab line 37 and 38
	float spkHis_Last_mod, spk_Spec_mod;
	for(int i = 0; i < BAND_LENGTH; i++)
	{
		spkHis_Last_mod = spkHistory_real[(NTAPS-1)*BAND_LENGTH+i]*spkHistory_real[(NTAPS-1)*BAND_LENGTH+i] + \
							spkHistory_imag[(NTAPS-1)*BAND_LENGTH+i]*spkHistory_imag[(NTAPS-1)*BAND_LENGTH+i];
		spk_Spec_mod = cspk_real_buffer[i] * cspk_real_buffer[i] + cspk_imag_buffer[i] * cspk_imag_buffer[i];
		spkHistoryPower[i] = spkHistoryPower[i] - spkHis_Last_mod + spk_Spec_mod;
		spkSmoothedHistPower[i] = spkSmoothedHistPower[i] * spksmoothfactor + spkHistoryPower[i] * (1 - spksmoothfactor);
	}
	// matlab line 39
	for(int i = NTAPS-1; i > 0 ; i--)
	{
		memcpy(spkHistory_real+i*BAND_LENGTH, spkHistory_real+(i-1)*BAND_LENGTH, 4*BAND_LENGTH);
		memcpy(spkHistory_imag+i*BAND_LENGTH, spkHistory_imag+(i-1)*BAND_LENGTH, 4*BAND_LENGTH);
	}
	// matlab line 40
	memcpy(spkHistory_real, cspk_real_buffer, 4*BAND_LENGTH);
	memcpy(spkHistory_imag, cspk_imag_buffer, 4*BAND_LENGTH);

	// matlab line 42
	for(int i = 0; i < BAND_LENGTH; i++)
	{
		micpower[i] = cmic_real_buffer[i] * cmic_real_buffer[i] + cmic_imag_buffer[i] * cmic_imag_buffer[i];
	}
	// matlab line 43-47
	if(k==0)
	{
		memcpy(avg_micpower, micpower, 4*BAND_LENGTH);
	}
	else
	{
		for(int i = 0; i < BAND_LENGTH; i++)
		{
			avg_micpower[i] = avg_micpower[i] * resPowSmoothFactor + micpower[i] * (1 - resPowSmoothFactor);
		}
	}
	// complex operation memory
	//float * result_real = new float(), * result_imag = new float();

	// matlab line 50: front path filtering
	for(int i = 0; i < BAND_LENGTH; i++)
	{
		echoEstimate_real[i] = 0;
		echoEstimate_imag[i] = 0;
		for(int j = 0; j < NTAPS; j++)
		{
			echoEstimate_real[i] += spkHistory_real[j*BAND_LENGTH+i] * h_front_real[j*BAND_LENGTH+i] - spkHistory_imag[j*BAND_LENGTH+i] * h_front_imag[j*BAND_LENGTH+i];
			echoEstimate_imag[i] += spkHistory_real[j*BAND_LENGTH+i] * h_front_imag[j*BAND_LENGTH+i] + spkHistory_imag[j*BAND_LENGTH+i] * h_front_real[j*BAND_LENGTH+i];
		}
	}
	// matlab line 51
	for(int i = 0; i < BAND_LENGTH; i++)
	{
		echoResidualFront_real[i] = cmic_real_buffer[i] - echoEstimate_real[i];
		echoResidualFront_imag[i] = cmic_imag_buffer[i] - echoEstimate_imag[i];
	}
	// matlab line 52
	for(int i = 0; i < BAND_LENGTH; i++)
		res_pow_front[i] = echoResidualFront_real[i] * echoResidualFront_real[i] + echoResidualFront_imag[i] * echoResidualFront_imag[i];
	// matlab line 53
	if(k==0)
	{
		memcpy(avg_respowfront, res_pow_front, 4*BAND_LENGTH);
	}
	else
	{
		for(int i = 0; i < BAND_LENGTH; i++)
		{
			avg_respowfront[i] = avg_respowfront[i] * resPowSmoothFactor + res_pow_front[i] * (1 - resPowSmoothFactor);
		}
	}
	// matlab line 58
	for(int i = 0; i< BAND_LENGTH; i++)
	{
		erlefront[i] = avg_micpower[i] / avg_respowfront[i];
	}
	// matlab line 60: back path filtering
	for(int i = 0; i < BAND_LENGTH; i++)
	{
		echoEstimate_real[i] = 0;
		echoEstimate_imag[i] = 0;
		for(int j = 0; j < NTAPS; j++)
		{
			echoEstimate_real[i] += spkHistory_real[j*BAND_LENGTH+i] * h_back_real[j*BAND_LENGTH+i] - spkHistory_imag[j*BAND_LENGTH+i] * h_back_imag[j*BAND_LENGTH+i];
			echoEstimate_imag[i] += spkHistory_real[j*BAND_LENGTH+i] * h_back_imag[j*BAND_LENGTH+i] + spkHistory_imag[j*BAND_LENGTH+i] * h_back_real[j*BAND_LENGTH+i];
		}
	}

	// matlab line 61
	for(int i = 0; i < BAND_LENGTH; i++)
	{
		echoResidualBack_real[i] = cmic_real_buffer[i] - echoEstimate_real[i];
		echoResidualBack_imag[i] = cmic_imag_buffer[i] - echoEstimate_imag[i];
	}
	// matlab line 62
	for(int i = 0; i < BAND_LENGTH; i++)
		res_pow_back[i] = echoResidualBack_real[i] * echoResidualBack_real[i] + echoResidualBack_imag[i] * echoResidualBack_imag[i];
	// matlab line 63
	if(k==0)
	{
		memcpy(avg_respowback, res_pow_back, 4*BAND_LENGTH);
	}
	else
	{
		for(int i = 0; i < BAND_LENGTH; i++)
		{
			avg_respowback[i] = avg_respowback[i] * resPowSmoothFactor + res_pow_back[i] * (1 - resPowSmoothFactor);
		}
	}
	// matlab line 68
	for(int i = 0; i< BAND_LENGTH; i++)
		erleback[i] = avg_micpower[i] / avg_respowback[i];
	// matlab line 71: find best echo residual for each band
	float lowestrespow;
	for(int i = 0; i < BAND_LENGTH; i++)
	{
		lowestrespow = 0;
		if(res_pow_front[i] < res_pow_back[i])
		{
			lowestrespow = res_pow_front[i];
			bestResidual_real[i] = echoResidualFront_real[i];
			bestResidual_imag[i] = echoResidualFront_imag[i];
		}
		else
		{
			lowestrespow = res_pow_back[i];
			bestResidual_real[i] = echoResidualBack_real[i];
			bestResidual_imag[i] = echoResidualBack_imag[i];
		}
		if(lowestrespow > micpower[i])
		{
			lowestrespow = micpower[i];
			bestResidual_real[i] = cmic_real_buffer[i];
			bestResidual_imag[i] = cmic_imag_buffer[i];
		}
		bestResidualPow[i] = lowestrespow;
	}
	// matlab line 88
	memcpy(mic_out_real, bestResidual_real, 4*BAND_LENGTH);
	memcpy(mic_out_imag, bestResidual_imag, 4*BAND_LENGTH);
	// matlab line 89
	float erle_present, sum_micpower = 0, sum_bestResidualPow = 0;
	for(int i = 0; i < BAND_LENGTH; i++)
	{
		sum_micpower += micpower[i];
		sum_bestResidualPow += bestResidualPow[i];
	}
	float todb = sum_micpower / (sum_bestResidualPow + 1e-10);
	erle_present = 10 * log10(todb*todb) / 2; // db() function in matlab

	// matlab line 92
	int ntransfertofront = 0;
	for(int i = 0; i < BAND_LENGTH; i++)
	{
		if(erleback[i] > transferThresERLE && avg_respowfront[i] > avg_respowback[i] * transferThresPowRatio)
		{
			// copy a col of h_back to h_front
			for(int j = 0; j < NTAPS; j++)
			{
				h_front_real[j*BAND_LENGTH+i] = h_back_real[j*BAND_LENGTH+i];
				h_front_imag[j*BAND_LENGTH+i] = h_back_imag[j*BAND_LENGTH+i];
			}
			ntransfertofront += 1;
		}
	}
	// matlab line 100
	int ntransfertoback = 0;
	for(int i = 0; i < BAND_LENGTH; i++)
	{
		if(erlefront[i] > transferThresERLE && avg_respowback[i] > avg_respowfront[i] * transferThresPowRatio)
		{
			// copy a col of h_front to h_back
			for(int j = 0; j < NTAPS; j++)
			{
				h_back_real[j*BAND_LENGTH+i] = h_front_real[j*BAND_LENGTH+i];
				h_back_imag[j*BAND_LENGTH+i] = h_front_imag[j*BAND_LENGTH+i];
			}
			ntransfertoback += 1;
		}
	}
	// matlab line 109
	float sum_spkHistoryPower = 0;
	for(int i = 0; i < BAND_LENGTH; i++)
		sum_spkHistoryPower += spkHistoryPower[i];
	float * spkPower, * beta_real, * beta_imag;
	spkPower = new float[BAND_LENGTH]();
	beta_real = new float[BAND_LENGTH]();
	beta_imag = new float[BAND_LENGTH]();
	if(sum_spkHistoryPower/(BAND_LENGTH*NTAPS) > adaptsignalthreshold * adaptsignalthreshold)
	{
		for(int j = 0; j < BAND_LENGTH; j++)
			spkPower[j] = (spkHistoryPower[j] + spkSmoothedHistPower[j])>1e-6?spkHistoryPower[j] + spkSmoothedHistPower[j]:1e-6;
		for(int j = 0; j < BAND_LENGTH; j++)
		{
			beta_real[j] = MU * echoResidualBack_real[j] / spkPower[j];
			beta_imag[j] = MU * echoResidualBack_imag[j] / spkPower[j];
		}
		for(int i = 0; i < NTAPS; i++)
			for(int j = 0; j < BAND_LENGTH; j++)
			{
				h_back_real[i*BAND_LENGTH+j] += beta_real[j] * spkHistory_real[i*BAND_LENGTH+j] + beta_imag[j] * spkHistory_imag[i*BAND_LENGTH+j];
				h_back_imag[i*BAND_LENGTH+j] += beta_imag[j] * spkHistory_real[i*BAND_LENGTH+j] - beta_real[j] * spkHistory_imag[i*BAND_LENGTH+j];
			}
	}
	delete [] spkPower;
	delete [] beta_real;
	delete [] beta_imag;
	// matlab line 117: calculate spkmicdelay for each frame
	/*
	int spkMicDelay;
	float * tapPower, * h_front_mod;
	tapPower = (float*)calloc(NTAPS, sizeof(float));
	if(erle > 2)
	{
		for(int i = 0; i < NTAPS; i++)
			for(int j = 0; j < BAND_LENGTH; j++)
				tapPower[i] += h_front_real[i*BAND_LENGTH+j] * h_front_real[i*BAND_LENGTH+j] + h_front_imag[i*BAND_LENGTH+j] * h_front_imag[i*BAND_LENGTH+j];
		int max_index = 0;
		for(int i = 0; i < NTAPS; i++)
		{
			if(tapPower[i] > max)
			{
				max_index = i;
			}
		}
		spkMicDela = max_index - 1;
	}
	else if(!first_frame_flag)
	{
		spkMicDelay = spkMicDelay_last;
	}
	*/
	delete [] cmic_real_buffer;
	delete [] cmic_imag_buffer;
	delete [] cspk_real_buffer;
	delete [] cspk_imag_buffer;

	delete [] micpower;
	delete [] echoEstimate_real;
	delete [] echoEstimate_imag;
	delete [] echoResidualFront_real;
	delete [] echoResidualFront_imag;
	delete [] echoResidualBack_real;
	delete [] echoResidualBack_imag;
	delete [] res_pow_front;
	delete [] res_pow_back;
	delete [] erlefront;
	delete [] erleback;

	return erle_present;
}

