# Quantization of Neural Network

## Part 1:Architecture Overview

The neural network model, referred to as FC_NN, is a fully connected (FC) architecture designed for classification tasks. The model consists of three linear layers interconnected by ReLU activation functions. 

### Layers and Functions
Input Layer: The input to the model is reshaped to a 1D vector of size 784 (28x28), suitable for processing images from datasets such as MNIST. This reshaping is crucial for transforming 2D image data into a format that can be fed into the fully connected layers.

First Hidden Layer: A linear layer that takes the reshaped input vector of size 784 and transforms it to a 64-dimensional space. It is followed by a ReLU activation function, introducing non-linearity to the model and enabling it to learn complex patterns in the data.

Second Hidden Layer: Another linear transformation that reduces the dimensionality from 64 to 32. It is similarly followed by a ReLU activation function for non-linear transformations.

Output Layer: The final linear layer reduces the feature space to 10 dimensions, corresponding to the number of classes in the classification task. The log softmax function is applied to this layer's output, providing a probability distribution over the 10 classes. This output is particularly useful for multi-class classification problems, where each input should be classified into one of several categories.

### Loss Function and Optimization
Loss Function: The network utilizes the negative log-likelihood loss (nll_loss), which works in tandem with the log softmax output. 
Optimization Algorithm: The Adam optimizer is employed to adjust model parameters based on the computation of adaptive learning rates for each parameter. 

### Quantization meathods
For this project, I'am using the "post-training quantization" meathod by the following steps:
1 Determining Scale and Zero-Point: For each layer of the network, the range of weights and activations is analyzed to calculate the appropriate scale and zero-point values. The scale factor is used to convert the floating-point numbers into integers, and the zero-point is the value used to align the integers with the original floating-point range, accommodating for the offset that occurs during quantization.

2 Asymmetric Quantization: The method I'am using is asymmetric, meaning the quantization process does not assume symmetry around zero. This allows for a potentially more efficient use of the available integer range, accommodating for biases in the weight and activation distributions. With the calculated scale and zero-point for each layer, the floating-point numbers are linearly transformed into integers.

3 Quantization to Fixed Bits: The quantized values are then mapped to a predefined bit-width. This step reduces the precision of the model's parameters to the set number of bits, which reduces the model size and can potentially accelerate computation, especially on hardware that is optimized for low-precision arithmetic.

4 Dequantization: After each layer's forward pass, the quantized integers are converted back into floating-point numbers using the inverse of the scale and zero-point adjustments. This dequantization step is necessary for the subsequent layers to perform accurate calculations, particularly if they have not been quantized or if the network outputs need to be interpreted in a floating-point context.

## Part 2: Accuracy
In the notebook, I ploted a figure. This graph illustrates how neural network accuracy varies with different bit precisions used in quantization. At very low precision (1-2 bits), accuracy is poor,which are 9.74% and 11.5%,  suggesting insufficient representation of the model's parameters. Between 3 and 5 bits, which are 76.74%,  92.29% and 96%, there is a rapid increase in accuracy, indicating that these precisions are capturing more relevant information for making predictions. Beyond 6 bits, accuracy levels off, indicating that higher precision may offer minimal accuracy gains, which is crucial for optimizing performance within hardware constraints. For 6 bits the accuracy is 96.65%, 7 bits-97% and for the 8 bits, the accuracy is 96.71%.


## Part 3: Commentary on Observed Accuracy and Potential Improvements
My current quantization approach applied to a fully connected neural network has yielded significant results. The visible effects of quantization on model accuracy have set a promising foundation for our research into neural network efficiency.

### Future Directions
1 Enhance Quantization Codebase: Refining my quantization algorithms is an ongoing effort. By improving the precision of our scale and zero-point calculations, and by fine-tuning the quantization-dequantization process, we can minimize the loss of information and, therefore, maintain or even improve accuracy post-quantization. Also I will entend this quantization module to the CNN architecture instead of only fully connected layers.

2 Experimentation with Larger Models: To validate the robustness and scalability of my quantization techniques, I aim to apply them to larger and more complex neural network models. 

3 Model Generalization: For the larger CNN module and complicated dataset-cifar-10. I may consider methods to enhance generalization, such as data augmentation, regularization techniques, and hyperparameter tuning, to ensure that the  quantized models perform well better.
