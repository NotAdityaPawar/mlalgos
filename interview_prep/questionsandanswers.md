AI Engineering Interview Questions (2–4 years experience)

## Python Programming and Scripting

**Is Python a compiled language or an interpreted language?**
Python is an interpreted language. The source code is compiled to bytecode (.pyc files) at runtime, then executed by the Python Virtual Machine (PVM). This makes it platform-independent but slower than compiled languages.

**How can you concatenate two lists in Python?**
```python
# Method 1: + operator
list1 + list2
# Method 2: extend()
list1.extend(list2)
# Method 3: unpacking
[*list1, *list2]
# Method 4: itertools.chain
list(itertools.chain(list1, list2))
```

**What is the difference between for and while loops in Python?**
- **for loop**: Iterates over sequences (lists, strings, ranges). Used when you know the number of iterations.
- **while loop**: Continues until a condition becomes False. Used when the number of iterations is unknown.

**How do you floor a number in Python?**
```python
import math
math.floor(4.7)  # Returns 4
# Or use // operator for integer division
4.7 // 1  # Returns 4.0
```

**What is the difference between / and // division operators in Python?**
- `/`: True division, returns float result (5/2 = 2.5)
- `//`: Floor division, returns integer result (5//2 = 2)

**Is indentation required in Python? Why?**
Yes, indentation is mandatory in Python. It defines code blocks and scope, replacing braces {} used in other languages. This enforces readable code structure.

**Can you pass a function as an argument to another function in Python?**
Yes, functions are first-class objects in Python:
```python
def apply_func(func, value):
    return func(value)

apply_func(len, "hello")  # Returns 5
```

**What does it mean that Python is a dynamically typed language?**
Variable types are determined at runtime, not compile time. You don't need to declare variable types explicitly:
```python
x = 5      # x is int
x = "hello" # x is now string
```

**What is the pass statement in Python and when would you use it?**
`pass` is a null operation placeholder. Used when syntax requires a statement but no action is needed:
```python
class EmptyClass:
    pass  # Placeholder for future implementation
```

**How are arguments passed in Python (by value, reference, or otherwise)?**
Python uses "pass by object reference". Immutable objects (int, string, tuple) behave like pass-by-value, while mutable objects (list, dict) behave like pass-by-reference.

**What is a lambda (anonymous) function in Python?**
A lambda is a small anonymous function defined inline:
```python
square = lambda x: x**2
# Equivalent to:
def square(x):
    return x**2
```

**What is list comprehension in Python? Give an example.**
A concise way to create lists:
```python
# Traditional approach
squares = []
for x in range(10):
    squares.append(x**2)

# List comprehension
squares = [x**2 for x in range(10)]
```

**What are *args and **kwargs in Python functions?**
- `*args`: Accepts variable number of positional arguments as tuple
- `**kwargs`: Accepts variable number of keyword arguments as dictionary
```python
def func(*args, **kwargs):
    print(args)    # (1, 2, 3)
    print(kwargs)  # {'a': 4, 'b': 5}

func(1, 2, 3, a=4, b=5)
```

**What are Python decorators and how are they used?**
Decorators modify or extend function behavior without changing the function itself:
```python
def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"Time: {time.time() - start}")
        return result
    return wrapper

@timer
def slow_function():
    time.sleep(1)
```

**What is an iterator in Python?**
An object that implements `__iter__()` and `__next__()` methods, allowing iteration through elements one at a time.

**What is a generator in Python?**
A function that uses `yield` to return values lazily, creating an iterator that generates values on-demand, saving memory:
```python
def fibonacci():
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b
```

Machine Learning Algorithms and Implementation

What is the difference between supervised and unsupervised learning?

Can you explain overfitting and underfitting in machine learning models?

What is cross-validation and why is it important?

What is the bias–variance trade-off in machine learning?

What is the role of the cost (loss) function in training a model?

What is the curse of dimensionality and how can you mitigate it?

Why is the Naïve Bayes classifier “naïve”?

What is semi-supervised learning? Give an example scenario.

What is self-supervised learning and how does it differ from unsupervised learning?

Explain curriculum learning in machine learning. When is it useful?

Describe how a decision tree algorithm works. When would you choose a decision tree?

What is the difference between bagging and boosting?

How do decision trees handle continuous numerical variables?

What are entropy and information gain in the context of decision trees?

What is pruning in decision trees and why is it used?

Explain the differences between CART, ID3, and C4.5 decision tree algorithms.

How can decision trees handle missing values during training or prediction?

What is bootstrapping in Random Forests?

How does Random Forest perform feature selection compared to a single tree?

Why is a Random Forest often less prone to overfitting than a single decision tree?

How can you estimate feature importance in a Random Forest?

What are the key hyperparameters to tune in a Random Forest model?

How does XGBoost differ from traditional gradient boosting?

What is the difference between hard voting and soft voting in ensemble methods?

How does LightGBM differ from XGBoost?

Deep Learning Concepts and Frameworks

What are the advantages and disadvantages of deep neural networks?

How does a convolutional neural network (CNN) work?

What is an activation function in a neural network, and why is it needed?

What are vanishing and exploding gradients, and how can they be addressed?

What is batch normalization and how does it differ from layer normalization?

How does an LSTM differ from a standard RNN? What problem does it solve?

What is dropout in a neural network, and how do you choose the dropout rate?

Explain the concept of attention mechanisms in neural networks.

What is transfer learning? When and why would you use it?

How do Generative Adversarial Networks (GANs) work? What are common applications of GANs?

In CNNs, what is the difference between “same” padding and “valid” padding?

How does Word2Vec work? Explain CBOW and skip-gram models.

What are Transformers and how do they differ from traditional sequence models?

What is a neural network (Artificial Neural Network)? Describe its layers and basic operation.

How are weights and biases used in a neural network?

How are weights typically initialized in neural networks (e.g. Xavier/He initialization)?

What is an activation function (e.g. ReLU, sigmoid, tanh), and why introduce nonlinearity?

What is gradient descent and what are its common variants (batch, stochastic, mini-batch)?

Difference between Batch Gradient Descent, Stochastic Gradient Descent, and Mini-Batch Gradient Descent?

What is the vanishing gradient problem in deep networks? How can it be mitigated?

What is dropout regularization and how does it prevent overfitting?

What is batch normalization, and how does it help training deep networks?

Explain the basic idea of a Neural Network being a “black box.” (Understanding interpretability).

Data Preprocessing and Feature Engineering

How do you handle missing or corrupted data in a dataset?

How would you handle an imbalanced dataset during model training?

What is feature selection in machine learning? What are filtering vs wrapper methods?

How do you handle categorical variables for machine learning models?

What is principal component analysis (PCA) and when is it used?

How does PCA differ from independent component analysis (ICA)?

How would you handle time-based (temporal) features in a model?

What is feature hashing and when might you use it?

How do you handle hierarchical (nested) categorical variables?

What are embedding layers and when should you use them?

What strategies can be used to handle outliers in data?

How do you ensure your features are appropriately scaled or normalized for a model? (e.g. standardization, normalization) (knowledge of feature scaling).

How do you perform one-hot encoding or ordinal encoding of categorical data? (Common techniques for categorical features).

System Design for ML/AI Applications

How would you design a real-time recommendation system for a large-scale e-commerce platform?

Describe the system design of a machine learning model that predicts stock prices.

How would you build a fraud detection system using machine learning?

How would you ensure that an ML-based system is scalable and can handle growing load?

What trade-offs exist between using SQL vs NoSQL databases in ML systems?

Explain how to design a scalable ML pipeline for large-scale data. (design a pipeline with distributed processing).

How would you design an ML system to perform real-time network anomaly detection?

What considerations are important when deploying a model to production (in terms of monitoring, drift, etc.)?

How would you optimize a model that is underfitting the training data?

How would you optimize a model that is overfitting (not just memorizing) the training data?

Describe steps to ensure the security and privacy of data in an ML pipeline.

What are the challenges of using deep learning models in production?

How could you use reinforcement learning in a system that adapts to user behavior?

Describe how you would design a demand forecasting system and improve its accuracy.

What machine learning techniques improve the relevance of search results (e.g. for a search engine)?

How would you design a churn prediction and prevention system for a subscription service?

Explain how convolutional neural networks (CNNs) are used in image recognition systems.

How would you implement a context-aware NLP system (e.g. using BERT or transformers)?

Mathematics for Machine Learning

What is the significance of eigenvalues and eigenvectors in machine learning (e.g. PCA)?

How is matrix factorization used in recommendation systems?

How does logistic regression differ from linear regression (from a mathematical standpoint)?

What is the role of the sigmoid function in logistic regression?

Explain the chain rule in calculus and how it is used in neural network backpropagation. (Fundamental calculus for ML).

What is a gradient, and how is it used in optimization (e.g. in gradient descent)?

What is a Jacobian or Hessian matrix in the context of deep learning optimization? (Concept of second derivatives in optimization).

Explain bias and variance from a statistical perspective in model errors.

How would you apply Bayesian probability in a classification problem (e.g. Naïve Bayes)? (Bayes’ theorem as probability concept).

Describe how probability distributions (e.g. Gaussian, Bernoulli) are used in modeling data. (Probability fundamentals for ML).

What is the difference between convex and non-convex optimization problems? (Optimization fundamentals).

How does eigen decomposition relate to dimensionality reduction techniques? (Linear algebra connection).

What is Principal Component Analysis (PCA) and how does it work mathematically? (Covariance, eigenvectors).

What is the bias term in linear regression, and how is it determined? (Basics of regression math).

How does the choice of loss function (L1 vs L2) affect optimization? (Optimization and calculus).

Model Evaluation, Metrics, and Debugging

Which evaluation metrics would you use for binary classification? Multi-class classification? Regression?

What is the difference between accuracy and AUC (area under ROC curve)?

What metrics would you use to evaluate the performance of a recommender system?

How do you ensure a model is not just memorizing the training data (i.e., detect overfitting)?

What is early stopping in model training and how is it implemented?

Explain model distillation (knowledge distillation) and when to use it.

How do you handle class imbalance when evaluating models? (e.g. class weighting, resampling, metrics)

What is precision and recall, and when would you focus on one over the other? (precision/recall context).

How do you use a confusion matrix to evaluate a classification model? (Interpretation of TP/FP/FN/TN).

Explain the concept of a ROC curve and how it relates to model thresholds. (Binary classification metric analysis).

How do you compute and use F1-score? Why is it important? (Precision/recall harmonic mean).

How would you debug a machine learning model that is not performing well? (Diagnostic steps like checking data, features, assumptions).

How do you tune hyperparameters and compare model versions? (Cross-validation, grid/random search).

MLOps Fundamentals

What is MLOps (Machine Learning Operations) and why is it important?

What are the main components of an MLOps pipeline?

What is a CI/CD (Continuous Integration/Deployment) pipeline in the context of ML?

How would you deploy a machine learning model in a production environment?

What is model versioning or a model registry, and why is it needed?

How do you implement A/B testing for machine learning models?

How do you monitor machine learning models in production (e.g. drift detection)?

What are shadow deployments in ML, and when would you use them?

What strategies are used for ML pipeline orchestration and workflow management?

How do you ensure model fairness and mitigate bias in deployed systems?

What role do feature stores play in ML systems?

How do you handle concept drift versus data drift?

What is data versioning and why is it critical in ML projects?

What are popular tools for experiment tracking and model monitoring (e.g. MLflow, Kubeflow)? (MLOps tool knowledge).

What is the difference between online learning and batch learning in production models? (Streaming data model updates vs batch).
