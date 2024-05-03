# Quantization of Neural Network

## Part 1:Architecture Overview

we used the LeNet5 network structure, which includes two convolutional layers and three fully connected layers. It corresponds to two datasets, MNIST and CIFAR-10, each processed by two models. The difference between the two models lies in the stride and channel of the first convolutional layer, with adjustments also made to the padding to suit the different datasets.

### Layers and Functions

Between the convolutional layers, we only use max pooling and ReLU, while between the fully connected layers, only ReLU is used. The output of the last layer employs softmax as the activation function.

### Loss Function and Optimization
Loss Function: The network utilizes the negative log-likelihood loss (nll_loss), which works in tandem with the log softmax output. 
Optimization Algorithm: The Adam optimizer is employed to adjust model parameters based on the computation of adaptive learning rates for each parameter. 

### Quantization meathods
For this project, I'am using the "post-training quantization" meathod by the following steps:
1 Determining Scale and Zero-Point: For each layer of the network, the range of weights and activations is analyzed to calculate the appropriate scale and zero-point values. The scale factor is used to convert the floating-point numbers into integers, and the zero-point is the value used to align the integers with the original floating-point range, accommodating for the offset that occurs during quantization.

2 Asymmetric Quantization: The method I'am using is asymmetric, meaning the quantization process does not assume symmetry around zero. This allows for a potentially more efficient use of the available integer range, accommodating for biases in the weight and activation distributions. With the calculated scale and zero-point for each layer, the floating-point numbers are linearly transformed into integers.

3 Quantization to Fixed Bits: The quantized values are then mapped to a predefined bit-width. This step reduces the precision of the model's parameters to the set number of bits, which reduces the model size and can potentially accelerate computation, especially on hardware that is optimized for low-precision arithmetic.

4 Dequantization: After each layer's forward pass, the quantized integers are converted back into floating-point numbers using the inverse of the scale and zero-point adjustments. This dequantization step is necessary for the subsequent layers to perform accurate calculations, particularly if they have not been quantized or if the network outputs need to be interpreted in a floating-point context.

## Part 2: Dataset
### MNIST
The MNIST dataset is a collection of handwritten digits ranging from 0 to 9. It consists of 60,000 training images and 10,000 test images. Each image is grayscale and has a dimension of 28x28 pixels.
### Cifar10
The CIFAR-10 dataset contains 60,000 color images in 10 different classes, with each class representing a type of object (such as airplanes, cars, birds, etc.). There are 50,000 training images and 10,000 test images. Each image is in color (RGB) and has a dimension of 32x32 pixels.

Compared to MNIST, CIFAR-10 has two additional channels and larger pixel dimensions of 32. Therefore, for the MNIST dataset, the parameters for the first convolutional layer are: in_channels=1, out_channels=6, kernel_size=5, stride=1, and padding=2. For CIFAR-10, the parameters for the first layer are: in_channels=3, out_channels=6, kernel_size=5, stride=1, and padding=0.

## Part 2: Accuracy
In the notebook, I ploted two figures. The first graph illustrates how LeNet-MNIST accuracy varies with different bit precisions used in quantization. At very low precision (1-2 bits), accuracy is poor,which are 9.74% and 10.5%,  suggesting insufficient representation of the model's parameters. Between 3 and 5 bits, which are 75.74%,  96.78% and 98.7%, there is a rapid increase in accuracy, indicating that these precisions are capturing more relevant information for making predictions. Beyond 6 bits, accuracy levels off, indicating that higher precision may offer minimal accuracy gains, which is crucial for optimizing performance within hardware constraints. For 6 bits the accuracy is 98.96%, 7 bits-99.06% and for the 8 bits, the accuracy is 99.04%.

Compared to the fully connected layers in the first solution, the addition of convolutional layers in the LeNet network increased the accuracy of the MNIST dataset from 96% to 99%

For the second image, it shows the impact of different parameter precisions on the accuracy for CIFAR-10, which is similar to MNIST. At 1-bit and 2-bit precisions, the network's accuracy is only around 10%, similar to a network with random initializations. Starting from 3-bit, the accuracy gradually increases, reaching near peak values at 7-bit precision, close to the accuracy of the network at full precision (70%).


## Part 3: Commentary on Observed Accuracy 

It can be observed that for the same network structure, LeNet, the quantization results show some differences across different datasets. For MNIST, a precision of approximately 4-bit is sufficient to achieve fairly good performance, whereas for CIFAR-10, a precision of 6 or 7 bits might be necessary to approach the accuracy of full precision. Moreover, when comparing different network structures on the same dataset—between the first solution and the final solution—we find that the quantization trends of both networks are consistent, achieving near full precision performance around 4-bit. This suggests that quantization may be more sensitive to the dataset.

### Future Directions

If further improvement in the accuracy of quantized networks is desired, subsequent steps could include quantize-aware training. This involves incorporating the quantization error into the training process, thereby taking this aspect into account during training to further enhance the final network's accuracy.