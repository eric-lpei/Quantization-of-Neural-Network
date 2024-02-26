# Quantization of Neural Network

## Part 1: Conceptual Design

### Introduction
As deep learning technology continues to evolve, we have been able to develop increasingly large and complex models that demonstrate exceptional performance across various domains. However, this progress also introduces new challenges, particularly in terms of hardware resource requirements. The high computational demands of large deep learning models lead to increased latency, higher energy consumption, and issues with chip size and cost. Therefore, the quantization of neural networks becomes critically important.

Quantization, in simple terms, is the process of converting model parameters from a high-precision format (e.g., 32-bit floating-point, float32) to a low-precision format (e.g., 16-bit integer, int16, or 8-bit integer, int8). This process can significantly reduce the model's demand for storage space and, due to the hardware's higher efficiency in low-precision computations, can accelerate the model's inference process.

Numerous studies have shown that parameters in neural networks tend to be highly redundant, with most parameter values concentrated in the -1 to 1 range. This characteristic provides ideal conditions for quantization because it means that the performance loss of the model can be kept within an acceptable range even after conversion to a lower data precision.

Quantization not only helps reduce storage and computational costs but also makes it possible to run deep learning models on resource-constrained devices, such as mobile and embedded systems. This is important for the widespread application of deep learning, especially in edge computing scenarios.

With carefully designed quantization strategies, we can significantly reduce hardware resource requirements while maintaining model performance. This includes choosing the appropriate level of quantization, quantization methods (e.g., symmetric vs. asymmetric quantization), and the timing of quantization (pre-training, during training, post-training). Furthermore, advanced quantization techniques, such as mixed-precision training, can further optimize the balance between performance and accuracy.

In summary, neural network quantization is not only a vital direction for optimizing deep learning models but also a key technological approach to making models more efficient, economical, and adaptable to various computing environments. As research deepens and technology evolves, quantization will continue to play an indispensable role in advancing deep learning technology.

### Proposed Solution
I'll start with a basic Multi-Layer Perceptron (MLP) model and gradually move to slightly more complex Convolutional Neural Network (CNN) architectures, focusing on quantizing both the weights and the outputs of each layer's activations to explore the impact of different quantization precisions on model performance. This exploration will delve into how quantization affects computational efficiency, model size, and the accuracy and generalization capabilities of neural networks across tasks. By experimenting with various precision levels, from high (e.g., float32) to low (e.g., int8 or int16), the aim is to strike an optimal balance between performance and efficiency. Additionally, if time permits, I may also explore quantization-aware training to further refine the quantization process and potentially enhance model performance under quantization constraints.

### Data Requirements
#### Training Set
For simple MLP module, I plan to use MNIST or Fashion MNIST to test, and for the complex CNN I may use cifar10.

#### Validation Set
Since this project aims to quantize the module based on pretrain module, so it's unnecessary to split to the validation set.


## Part 2: Datasets

### Source
MNIST： We can derectly download the datasets through pytroch. Here are partial codes to do that.

train_set= datasets.MNIST("../DATA", train=True,  download=True, transform=pipeline)
test_set= datasets.MNIST("../DATA", train=False,  download=True, transform=pipeline)

train_data = DataLoader(train_set, batch_size= BATCH_SIZE, shuffle=True)
test_data = DataLoader(test_set, batch_size = BATCH_SIZE, shuffle=True)

EMNIST:  EMNIST is Extended MNIST. It contains six different classes: By_Class, By_Merge, Balanced, Digits , Letters and MNIST. And all the images have 28*28 pixels. EMNIST digits has 28000 images, which are classified into 10 categories.

Thank for Adam's suggestion. EMNIST digits could be a great testset to verify the performances of my final network.

Same as the MNIST, pytorch can provide the same API to directly download the EMNIST digits.

train_set = datasets.EMNIST("../DATA", split='digits', train=True, download=True, transform=pipeline)
test_set = datasets.EMNIST("../DATA", split='digits', train=False, download=True, transform=pipeline)

### Train and validation subsets
Train subsets are used to let our model learn different characters and details. And the validation subsets are used to test the performance of our model after each epoch. We need to adjust our hyperparameters based on the reflection of validation subsets to improve the overall performance. Of course, based on the tendence of the loss, we can choose to early stop the training to avoid some overfitting.

