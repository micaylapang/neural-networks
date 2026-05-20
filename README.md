# Neural Networks Project

This repository contains three progressively more advanced neural networks implemented from scratch in Python.

| File                                | Description                                    |
| ----------------------------------- | ---------------------------------------------- |
| evaluator_nn.py            | Simple feedforward neural network evaluation                       |
| xor_trainer_nn.py          | Neural network training with backpropagation               |
| geometric_classifier_nn.py | Generalized nonlinear classification and function approximation |

---

## Multilayer Perceptron (MLP) 
Multilayer perceptrons are a foundational type of artificial neural network. 
They processes info through an input layer, hidden layers, and an output layer. 
MLPs are fully connected, meaning every neuron links to every neuron in the next layer, allowing them to model complex patterns.

<img width="800" height="400" alt="image" src="https://github.com/user-attachments/assets/caae3afe-4bcb-42ed-bc95-4e06eb14d947" />

---

## 1. Neural Network Evaluator

The first project implements a basic feedforward neural network evaluator.

1. Reads pretrained weights from a file
2. Applies activation functions
3. Performs forward propagation
4. Produces final outputs

Uses: 
* Dot-product neuron computation
* Multiple activation functions
* Layer-by-layer neural computation
* Logic gate evaluation 

---

## 2. XOR Neural Network Trainer

The second project expands the evaluator into a trainable neural network.

1. Randomly initializes weights
2. Performs forward propagation
3. Computes prediction error
4. Uses backpropagation
5. Updates weights with gradient descent
6. Learns XOR from training examples

Uses:
* Backpropagation implementation
* Gradient descent optimization
* Error minimization
* Weight updates
* Hidden-layer learning
* XOR classification

---

## 3. Geometric Classifier Neural Network

The third project generalizes the neural network into a nonlinear geometric classifier.

Instead of learning only XOR, the network:

1. Dynamically generates training data
2. Learns mathematical inequalities
3. Approximates nonlinear decision boundaries
4. Classifies points in continuous 2D space

Uses:
* Synthetic dataset generation
* 2D coordinate classification
* Nonlinear decision boundaries
* Multiple hidden layers
* Improved numerical stability
* Function approximation
* Continuous-space learning
* overflow protection for sigmoid calculations
---

# Math Concepts

## Activation Functions

### Identity

```math
f(x)=x
```

### ReLU

```math
f(x)=max(0,x)
```

### Sigmoid

```math
f(x)=\frac{1}{1+e^{-x}}
```

### Tanh-like Function

```math
f(x)=-1 + \frac{2}{1+e^{-x}}
```


## Dot Product

Neuron weighted sum:

```math
z=\sum_i x_iw_i
```



## Sigmoid Derivative

Used during backpropagation:

```math
f'(x)=f(x)(1-f(x))
```



## Loss Function

This minimizes squared error:

```math
E=\frac{1}{2}\sum (t-y)^2
```

Where:

* `t` = target output
* `y` = predicted output

## Gradient Descent

Weight update rule:

```math
w_{new}=w_{old}+\alpha\nabla E
```

Where:

* `α` = learning rate
* `∇E` = gradient of error

---
## Author 
Micayla Pang
