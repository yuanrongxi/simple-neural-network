# simple-neural-network
Simple NN is a neural network implemented in C language, which includes a multilayer perceptron and a convolutional network. It is a digital handwriting recognition program that utilizes the MNIST dataset for training the neural network model.

## compile and run
### MNIST file
Running requires the MNIST data set, which can be downloaded from http://yann.lecun.com/exdb/mnist/ and store the downloaded data in the ./data directory.Please view data/readme.txt
### Download
      git clone https://github.com/yuanrongxi/simple-neural-network.git
### Compile
      cd ./simple-neural-network
      make
### Run CNN 
	./nn cnn
### Run ANN
	./nn ann

### Windows
Visual studio 2013  open nn.sln and compile\run

### MacBook (X86)
Install llvm with libomp before Compile

```
brew install llvm libomp
```
