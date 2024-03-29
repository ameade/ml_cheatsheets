
# Glossary of Common Machine Learning Terms

1. **Activation Function**: A function that introduces non-linearity into a neural network, allowing it to learn complex patterns. Examples include sigmoid, tanh, and ReLU.

2. **Backpropagation**: The process of calculating gradients of the loss function with respect to the weights of a neural network, used for updating the weights during training.

3. **Batch**: A subset of the training data used for a single iteration of gradient descent during training.

4. **Bias**: A learnable parameter in a neural network that allows neurons to shift their activation functions.

5. **Cross-Validation**: A technique for assessing the performance of a model by splitting the data into multiple subsets, training and evaluating the model on different combinations of these subsets.

6. **Deep Learning**: A subfield of machine learning that focuses on learning representations from data using deep neural networks.

7. **Dropout**: A regularization technique that randomly drops out neurons during training to prevent overfitting.

8. **Epoch**: A complete pass through the entire training dataset during training.

9. **Feature**: Characteristic of a phenomenon being observed, like sentiment of a sentence or presence of a face.

10. **Gradient Descent**: An optimization algorithm used for minimizing the loss function of a model by iteratively updating its parameters in the direction of the negative gradient.

11. **Hyperparameter**: A parameter of a machine learning model that is set before training and controls the learning process. Examples include learning rate, number of hidden layers, and regularization strength.

12. **Loss Function**: A function that measures the difference between the predicted and actual outputs of a model, used for training the model.

13. **Neuron**: A basic unit in a neural network that takes inputs, applies weights and biases, and produces an output.

14. **Overfitting**: A situation where a model performs well on the training data but fails to generalize to new, unseen data.

15. **Regularization**: Techniques used to prevent overfitting by adding constraints or penalties to the model during training.

16. **Stochastic Gradient Descent (SGD)**: A variant of gradient descent that updates the model parameters using a single randomly selected example at a time.

17. **Tensor**: A multi-dimensional array used in machine learning frameworks to represent data and model parameters.

18. **Transfer Learning**: A technique where a model trained on one task is used as a starting point for training on a related task, allowing for faster and more efficient learning.

19. **Underfitting**: A situation where a model is too simple to capture the underlying patterns in the data, resulting in poor performance on both training and test data.

20. **Weight**: A learnable parameter in a neural network that represents the strength of the connection between neurons.

21. **One-Hot Encoding**: A technique for representing categorical variables as binary vectors, where each category is represented by a vector with a single 1 and all other values set to 0.

22. **Learning Rate**: A hyperparameter that controls the step size at which the model's weights are updated during training.

23. **Batch Normalization**: A technique used to normalize the activations of a neural network layer by subtracting the batch mean and dividing by the batch standard deviation, which can help improve the training speed and stability.

24. **Computational Graph** for out = x * y + z with gradients for inputs of all 1
```
   (x)1   (y)1   (z)1
    |      |      |
    |      |      |
   dx(.25) dy(.25)dz(0.5)
    |      |      |
    |     /       |
    |    /        |
  (mult)         dz (0.5)
      \          /
    d_mult(0.5)/
         \     /
         (add)
            |
         d_add(1)
            |
          (out)
````

25. **Supervised Learning** is a machine learning approach where an AI model is trained on labeled data, meaning that the input data is paired with corresponding target or output values.
26. **Reinforcement Learning** is a machine learning approach where an agent learns to make decisions by interacting with an environment. The agent learns to take actions that maximize a cumulative reward signal over time, based on the feedback it receives from the environment.
27. **on-policy** The data used for updating the policy is generated by the same policy that is being optimized.
28. **off-policy** The data used for updating the policy can come from any policy, not necessarily the one being optimized.