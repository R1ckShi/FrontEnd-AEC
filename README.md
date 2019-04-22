# FrontEnd-AEC

Acoustic echo cancelation(AEC) is a main algorithm in the pipe line of acoustic devices.

Input file:reference wav and mic wav, output file:mic wav - reference wav.

The algorithm will iterate an adaptive fillter bank to simulate the echo path.

The specific implementation method used is FNLMS. The original two signals are first framed, windowed and FFT transformed.

Suggestions about the frontend operation are needed(FFT, windowing, sampling...).

![image](https://github.com/Jeffery-nwpu/FrontEnd-AEC/blob/master/image/Snipaste_2019-04-22_16-03-15.png)
