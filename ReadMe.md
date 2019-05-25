# Time signal classification using Convolutional Neural Network
TensorFlow python code for a Machine Learning project using Convolutional Neural Network.

## Related Articles and Practical Examples
[Time signal classification using Convolutional Neural Network in TensorFlow - Part 1](https://datamadness.github.io/time-signal-CNN)
[Time signal classification using Convolutional Neural Network in TensorFlow - Part 2](https://datamadness.github.io/time-signal-CNN-part2)

## Project Overview
This example explores the possibility of using a Convolutional Neural Network(CNN) to classify time domain signal. The fundamental thesis of this work is that an arbitrarily long sampled time domain signal can be divided into short segments using a window function. These segments can be further converted to frequency domain data via [Short Time Fourier Transform(STFT)](https://en.wikipedia.org/wiki/Short-time_Fourier_transform)	. This approach is well known from acoustics, but it is easily applicable on any frequency spectrum.
The goal of this work is to add an alternative tool to an ensemble of classifiers for time signal predictions. To illustrate, this specific classifier example was developed on [VSB Power Line Fault Detection dataset](https://www.kaggle.com/c/vsb-power-line-fault-detection/data) where I aimed to combine three classifiers:

- Long Short-Term Memory (LSTM) Recurent Neural Network(RNN),
- Gradient Boosted Decision Tree using signal statistics, and finally the
- Convolutional Neural Network(CNN)

These three methods are based on very different principles and can complement each other with different sets of strengths and weaknesses.

## Visualization of the Implemented Convolutional Neural Networks
![Model 1](https://datamadness.github.io/assets/images/time_signal_CNN/CNN%20architecture%204096.png)
![Model 2](https://datamadness.github.io/assets/images/time_signal_CNN/CNN%20architecture%202048.png)
