# Embedded-Neural-Network-For-Fall-Detection
View Poster For More Information
Model Accuracy:
  Before Quantization: 97 Percent
  After Quantization: 96.9 Percent
Model Size: 50 kB
Inference Time: 0.04

For our optimization we increased the kernel size of our model 3, and we added dropout layers in order to prevent overfitting and minimize size. In order to minimize the inference Time we lowered our window and we made sure to use the sliding implementation of our model so we do not have to wait for 15 new data points for a prediction.
