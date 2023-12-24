# MACHINE LEARNING SPECIALIZATION

These Jupyter Notebooks are derived from the labs presented in the "Supervised Machine Learning: Regression and Classification" course by Andrew Ng.

# Linear Regression with Gradient Descent

The example focuses on linear regression and the implementation of gradient descent for parameter optimization.

## Overview

Linear regression is a crucial concept in supervised machine learning, establishing a relationship between features and targets. This example specifically considers simple linear regression, using house size as the feature and house price as the target. The model is trained on a set of training data, and its parameters \(w\) and \(b\) are optimized to make accurate predictions on new data.

## Key Concepts

- **Parameters**: The linear regression model has two parameters, \(w\) (weight) and \(b\) (bias), which are determined through training.

- **Cost Equation**: The cost equation assesses the alignment between model predictions and training data. Minimizing the cost yields optimal values for \(w\) and \(b\).

## Gradient Descent

Gradient descent is a fundamental optimization algorithm employed to minimize the cost function. This notebook explores the intricacies of gradient descent for a single variable, illustrating how to:

1. Compute the gradient of the cost function.
2. Visualize the gradient to gain insights.
3. Implement a gradient descent routine.
4. Apply gradient descent to find optimal values for \(w\) and \(b\).

## Contents

1. **Data Loading**: Load the dataset containing house sizes and prices.
2. **Data Visualization**: Visualize the relationship between house size and price.
3. **Linear Regression Model**: Implement the linear regression model.
4. **Cost Function**: Define the cost function to evaluate prediction accuracy.
5. **Gradient Descent**: Explain and implement gradient descent.
6. **Training the Model**: Employ gradient descent to train the model and determine optimal parameters.
7. **Prediction**: Make predictions on new data using the trained model.


# Python, NumPy, and Vectorization in Machine Learning

## Overview

This Jupyter Notebook provides a brief introduction to the fundamental concepts of scientific computing using Python, with a focus on the NumPy package. In particular, it explores the use of Python and NumPy in the context of machine learning, emphasizing vectorization for efficient numerical operations.

## Contents

1. **Python and NumPy**: An introduction to using Python as a programming language and NumPy as a powerful scientific computing package.

2. **Vectors**: Understanding the concept of vectors and their representation in Python using NumPy.

    - **NumPy Arrays**: Exploring NumPy arrays as the fundamental data structure for vector operations.
    
    - **Vector Creation**: Demonstrating various methods to create vectors using NumPy.

    - **Operations on Vectors**: Performing mathematical operations on vectors efficiently using NumPy.

3. **Matrices**: Introducing matrices as two-dimensional arrays and their manipulation with NumPy.

    - **NumPy Arrays**: Understanding the basics of NumPy arrays in the context of matrices.
    
    - **Matrix Creation**: Creating matrices using NumPy for various applications.
    
    - **Operations on Matrices**: Conducting matrix operations efficiently using NumPy.

## Key Concepts

- **NumPy**: A powerful open-source library for numerical computing in Python, providing support for large, multi-dimensional arrays and matrices, along with mathematical functions to operate on these data structures.

- **Vectorization**: The process of converting operations into vector form, enabling parallel processing and enhancing computational efficiency.


# Multiple Variable Linear Regression in Python

This Jupyter Notebook explores the implementation of Multiple Variable Linear Regression using Python. The lab extends the data structures and routines previously developed for single-variable linear regression to accommodate multiple features. While several routines are updated, they primarily involve minor adjustments to the existing code, making the review process quick and straightforward.

## Contents

1. **Matrix X and Parameter Vector**
    - Introduction to the matrix X containing our examples.
    - Definition of the parameter vector w and bias term b.

2. **Model Prediction With Multiple Variables**
    - Overview of predicting outcomes using multiple variables.

3. **Single Prediction (Element by Element)**
    - Explanation and implementation of predicting a single outcome element by element.

4. **Single Prediction (Vector)**
    - Illustration of predicting a single outcome using vectorized operations.

5. **Compute Cost With Multiple Variables**
    - Calculation of the cost function with multiple variables.

6. **Gradient Descent With Multiple Variables**
    - Introduction to the gradient descent algorithm for multiple variable linear regression.

7. **Compute Gradient with Multiple Variables**
    - Details on computing the gradient with respect to the parameters.

8. **Gradient Descent With Multiple Variables (Implementation)**
    - Step-by-step implementation of gradient descent for multiple variable linear regression.

9. **Routine Redevelopment**
    - Summary of the redeveloped routines for linear regression, now supporting multiple variables.

10. **NumPy `np.dot` Vectorization**
    - Explanation of utilizing NumPy's `np.dot` to vectorize the implementations, enhancing efficiency.


# Feature Scaling and Learning Rate (Multi-variable) in Python

This Jupyter Notebook builds upon the multiple variable linear regression routines developed in the previous lab. The focus of this lab is on running Gradient Descent on a dataset with multiple features. Additionally, we will explore the impact of the learning rate (alpha) on the efficiency of the gradient descent algorithm. To further improve the performance of gradient descent, we will implement feature scaling using z-score normalization.

## Contents

1. **Utilizing Multiple Variables Routines**
    - Brief overview of the routines developed in the previous lab for multiple variables.

2. **Running Gradient Descent with Multiple Features**
    - Implementation and demonstration of running Gradient Descent on a dataset with multiple features.

3. **Exploring the Impact of Learning Rate (Alpha)**
    - Experimentation with different values of the learning rate to understand its effect on gradient descent.

4. **Improving Performance with Feature Scaling**
    - Introduction to feature scaling and implementation of z-score normalization.

5. **Effect of Feature Scaling on Gradient Descent**
    - Comparison of gradient descent performance before and after applying feature scaling.


# Feature Engineering and Polynomial Regression

This Jupyter Notebook guides you through the process of feature engineering and explores polynomial regression, showcasing how linear regression techniques can be applied to model complex and highly non-linear functions. Feature engineering is a crucial aspect of machine learning that involves transforming raw data into a format suitable for modeling.

## Overview

Feature engineering is an integral part of the machine learning pipeline, and this lab focuses on its application in the context of polynomial regression. The key objectives of this lab include:

- **Exploring Feature Engineering**: Understanding the significance of feature engineering in enhancing model performance.

- **Polynomial Regression**: Applying linear regression techniques to fit complex and non-linear functions by introducing polynomial features.

- **Modeling Non-Linear Functions**: Recognizing how linear regression can effectively model intricate, non-linear relationships through the strategic use of feature engineering.

- **Feature Scaling**: Emphasizing the importance of feature scaling when conducting feature engineering to ensure optimal model performance.

## Contents

1. **Introduction to Feature Engineering**: Brief overview of the importance of feature engineering in machine learning.

2. **Polynomial Regression**: Introduction to polynomial regression and its application in modeling non-linear relationships.

3. **Feature Engineering Techniques**: Hands-on exploration of feature engineering techniques to enhance model flexibility.

4. **Implementation of Polynomial Regression**: Practical implementation of polynomial regression using feature-engineered data.

5. **Model Evaluation**: Assessing the performance of the polynomial regression model.

6. **Feature Scaling Importance**: Understanding why feature scaling is crucial, particularly in the context of feature engineering.
