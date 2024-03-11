# Brief Study Guide

## Neural Networks
- Artificial neural networks inspired by biological neural networks
- Consist of layers of interconnected nodes (neurons)
- Used for tasks like classification, regression, and pattern recognition

### Weights
- Weights are learnable parameters in a neural network
- Weights represent the strength of the connection between neurons
- During training, weights are adjusted to minimize the loss function
- Weights are typically initialized randomly or using techniques like Xavier or He initialization

### Biases
- Biases are additional learnable parameters in a neural network
- Each neuron in a layer has an associated bias term
- Biases allow neurons to shift their activation functions
- Like weights, biases are adjusted during training to minimize the loss function
- Biases provide flexibility for neurons to adapt their output independently of their inputs
- Biases are typically initialized to zero or small random values

### Optimization Algorithms
- During training, weights and biases are updated using an optimization algorithm
- Gradient descent updates weights and biases in the direction of the negative gradient of the loss function
- The learning rate determines the step size of the weight updates
- Variants of gradient descent include batch gradient descent, stochastic gradient descent (SGD), and mini-batch gradient descent
- More advanced optimization algorithms like Adam, RMSprop, and Adagrad adapt the learning rate for each parameter

### Backprop (Back propagation of gradients)
- Algorithm for training neural networks
- Computes gradients of the loss function with respect to network weights
- Allows efficient updating of weights to minimize the loss

### Hyperparameters
- Parameters set before training a neural network
- Examples: learning rate, number of layers, number of neurons per layer
- Tuning hyperparameters is crucial for optimal performance

### CNNs (Convolutional Neural Networks)
- Specialized neural networks for processing grid-like data (e.g., images)
- Use convolutional layers to learn local features
- Highly effective for tasks like image classification and object detection

### ResNets (Residual Neural Networks)
- CNN architecture with skip connections (residual connections)
- Skip connections allow learning of residual functions, enabling training of very deep networks
- Addresses the problem of vanishing gradients in deep networks
- Widely used for image classification tasks

### GANs & VAEs
- Generative models for creating new data similar to training data
- GANs (Generative Adversarial Networks): Two networks (generator and discriminator) trained simultaneously
- VAEs (Variational Autoencoders): Encode data into a latent space and decode to reconstruct the original data

## Reinforcement Learning
- Learning through interaction with an environment
- Agent takes actions to maximize cumulative reward
- Used for tasks like game playing and robotics

### Q-learning
- Value-based reinforcement learning algorithm
- Learns the optimal action-value function (Q-function/policy)
- Uses the Bellman equation to update Q-values
Agent: The entity that learns and makes decisions.
Environment: The world in which the agent operates and interacts.
State: A specific configuration of the environment.
Action: A move or decision made by the agent in a given state.
Reward: Feedback from the environment based on the agent's actions.
Q-value: The expected cumulative reward for taking an action in a given state.

The Q-learning algorithm updates the Q-values iteratively based on the Bellman equation:

Q(s, a) ← Q(s, a) + α * [R(s, a) + γ * max(Q(s', a')) - Q(s, a)]

where:

Q(s, a) is the current Q-value for state s and action a
α is the learning rate (0 < α ≤ 1)
R(s, a) is the reward obtained for taking action a in state s
γ is the discount factor (0 ≤ γ ≤ 1), which determines the importance of future rewards
s' is the next state after taking action a in state s
max(Q(s', a')) is the maximum Q-value for the next state s' across all possible actions a'

### Deep Q Learning
- Combines Q-learning with deep neural networks
- Neural network approximates the Q-function
- Enables learning from high-dimensional state spaces (e.g., images)

Deep Q-Networks (DQN) is an extension of the Q-learning algorithm that combines Q-learning with deep neural networks to enable learning from high-dimensional state spaces, such as images or complex sensory inputs. DQN has been successfully applied to play Atari games, achieving human-level performance on many of them.

Key concepts in DQN:

1. Q-Network: A deep neural network that approximates the Q-function, mapping states to Q-values for each action.
2. Experience Replay: A technique used to store the agent's experiences (state, action, reward, next state) in a replay buffer and sample from it randomly during training to break the correlation between consecutive samples.
3. Target Network: A separate neural network that is used to calculate the target Q-values during training, which helps stabilize the learning process.

The DQN algorithm works as follows:

1. Initialize the Q-Network with random weights and the Target Network with the same weights as the Q-Network.
2. Initialize the replay buffer.
3. For each episode:
   - Initialize the initial state.
   - For each time step:
     - Choose an action using an exploration strategy (e.g., epsilon-greedy).
     - Execute the action and observe the reward and the next state.
     - Store the experience (state, action, reward, next state) in the replay buffer.
     - Sample a mini-batch of experiences from the replay buffer.
     - For each experience in the mini-batch:
       - Calculate the target Q-value using the Target Network:
         - If the next state is terminal: target = reward
         - Otherwise: target = reward + γ * max(Q_target(next_state, a))
       - Calculate the current Q-value using the Q-Network: Q_current(state, action)
       - Calculate the loss: (target - Q_current(state, action))^2
     - Perform gradient descent on the Q-Network to minimize the loss.
     - Periodically update the Target Network weights with the Q-Network weights.


### Policy gradient methods

#### Reinforce
- REINFORCE is a basic policy gradient method introduced by Williams in 1992.
- It estimates the gradient of the expected return using the likelihood ratio trick and Monte Carlo sampling.
- The policy is updated using stochastic gradient ascent on the estimated gradient.
- REINFORCE is known to have high variance in gradient estimates, which can lead to slow convergence.

#### Actor-Critic
- Actor-Critic methods combine the policy gradient approach with a value function approximation.
- The "Actor" refers to the policy function, while the "Critic" refers to the value function.
- The Critic estimates the value function, which is used to evaluate the quality of the Actor's actions.
- The Actor is updated using the policy gradient, while the Critic is updated using temporal difference learning.

#### Trust Region Policy Optimization (TRPO)
- It introduces a trust region constraint on the policy update, limiting the amount of change in the policy at each step.
- While TRPO has strong theoretical foundations, it can be computationally expensive and complex to implement.

#### PPO
- Proximal Policy Optimization
- Policy gradient method for reinforcement learning
- Uses a clipped surrogate objective function to improve stability and sample efficiency. The PPO algorithm uses this objective function in conjunction with stochastic gradient ascent to adjust the parameters of the policy network, seeking to maximize the expected return while preventing destructive large updates.

### RLHF
- Reinforcement Learning from Human Feedback
- Trains an agent using feedback from human evaluators
- Allows learning complex behaviors without explicit reward functions

## Mechanistic Interpretability
- Understanding the internal workings of neural networks
- Aims to explain how networks make decisions and represent information

### Circuits
- Subgraphs within a neural network that perform specific functions
- Analogous to circuits in electronic systems
- Understanding circuits can provide insights into network behavior and decision-making

### Tools
- Ablation: involves systematically removing or deactivating specific components (neurons, layers) and observing the effect on the model's performance or behavior. This can be done by setting the activations or weights of the selected components to zero.
- Activation Maximization: Generating synthetic inputs that maximize the activation of specific neurons or channels to understand what features they respond to.


# GPT (Generative Pre-trained Transformer)
1. Transformer architecture: GPT models are based on the transformer architecture, which uses self-attention mechanisms to process and generate text. This allows the models to capture long-range dependencies and understand context effectively.
2. Unsupervised pre-training: GPT models are pre-trained on large, unlabeled text datasets using unsupervised learning. During this phase, the models learn to predict the next word in a sequence based on the previous words, allowing them to capture the underlying patterns and structures of language.
3. Fine-tuning: After pre-training, GPT models can be fine-tuned on specific tasks using labeled data. This process adapts the model to perform well on downstream tasks such as text classification, question answering, and language translation.

## Transformers
1. Multi-Head Attention: Transformers employ multi-head attention, which allows the model to attend to different parts of the input sequence in parallel. Each attention head learns different attention patterns, enabling the model to capture diverse relationships between elements.
2. Self-Attention Mechanism: Transformers rely heavily on self-attention, which allows the model to weigh the importance of different parts of the input sequence when processing each element. This enables the model to capture long-range dependencies and understand the context effectively.
3. Encoder-Decoder Structure: Transformers consist of an encoder and a decoder. The encoder processes the input sequence and generates a contextualized representation, while the decoder takes the encoder's output and generates the target sequence.

