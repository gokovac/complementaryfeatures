# Complementary Energy Features for Audio Spoof Detection

This repository contains codes for feature extraction and Se-Res2Net model used in [Complementary regional energy features for spoofed speech detection](https://www.sciencedirect.com/science/article/abs/pii/S0885230823001213).

Regional energy features are independent from the utterence lenghts, hence can be combined with deep embeddings. They provide a compact representation of the spectrogram. Time and frequency resolutions can be adjusted independenly. Examples for the utterances taken from LA and PA conditions of ASVspoof2019 dataset are given below. Combined with SE-Res2Net embeddings, 0.9% EER was obtained for the evaluation set of ASVspoof 2019 LA condition.

![image](https://github.com/gokovac/complementaryfeatures/assets/117350948/56d4be77-8d18-4ffc-bf6e-b8cdcf5684b3)

![image](https://github.com/gokovac/complementaryfeatures/assets/117350948/a0b49646-c3be-4c09-bcf4-ca273e7f90f6)

