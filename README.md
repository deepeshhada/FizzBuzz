# Project 1: FizzBuzz

## Objective 
The project is designed to quickly gain familiarity with neural network basics, Python and Machine Learning frameworks. The problem is to be solved using two approaches:  
1. *Logic-based* approach - **Software 1.0**.
2. *Deep learning* approach - **Software 2.0**.  

## Task
We consider the task of FizzBuzz. In this task an integer divisible by 3 is printed as *Fizz*, an integer divisible by 5 is printed as *Buzz* and an integer divisible by both 3 and 5 is printed as *FizzBuzz*. The model is tested on how well it performs in converting integers from 1 to 100 to the FizzBuzz labels.  
1. **Logic-based:**  
Implement the logic in Python using standard logic (if-then-else statements using modulo arithmetic). The program will presumably work perfectly on all 100 input integers.
2. **Machine Learning:**  
First create a training data set for numbers ranging from 101 to 1000. We avoid training on integers 1 to 100 because that forms the testing set. To design the learning program, make decisions on the neural network architecture *(no. of layers, no of units per layer)*, hyper-parameters such as the *learning rate, number of epochs, loss function, regularizer, etc.*  

## Tools
The code must be written in **Python**. Implement Software 2.0 using any of the Machine Learning frameworks such as *PyTorch, Keras and Gluon.*

## Result
Implemented the FizzBuzz using PyTorch. Achieved an accuracy of *95%* on the test dataset.
___