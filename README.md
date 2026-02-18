Word2Vec

This is an implementation of Word2Vec (Skip-gram with Negative Sampling) written entirely from scratch using only Python and NumPy.

There are no deep learning frameworks like PyTorch or TensorFlow involved here. The goal of this project is to explicitly demonstrate how the underlying matrix operations, manual gradient derivation, and backpropagation actually work under the hood.

Project Structure (What is what)
I split the code into four files to keep the logic clean and modular, similar to a real production environment:

settings.py
Just the configuration. This holds the hyperparameters like learning rate, embedding dimension, window size, and epochs. I kept this separate so you don't have to dig through the math to tweak the model.

preprocessing.py
Handles the text cleanup. It strips out punctuation, builds the vocabulary, drops rare words, and converts all the text tokens into integer IDs for faster processing.

model.py
The actual math and neural network logic. This contains the two weight matrices (w_in and w_out). It handles the forward pass (dot products and sigmoid), calculates the gradient of the loss, and applies the Stochastic Gradient Descent (SGD) updates.

main.py
The entry point. It loads the dataset, manages the sliding window over the text to generate training pairs, handles the negative sampling logic, and runs the main training loop.

How to Run
1. Install dependencies
You only need numpy for the math and datasets to pull the training text from Hugging Face.

Bash
pip install numpy datasets

2. Execute the code
If you want to run it with the default fka/prompts.chat dataset from Hugging Face, just run:

Bash
python main.py
If you want to feed it your own local text file instead:

Bash
python main.py path/to/your/textfile.txt
A Note on the Gradients (For the Interviewer)
Since this is built from scratch, the gradients for backpropagation were derived manually.

We use Binary Cross-Entropy loss. The model tries to predict 1 for actual context words (positive samples) and 0 for randomly chosen words (negative samples).

Thanks to the chain rule, the derivative of the loss with respect to the pre-activation dot product simplifies to just:
error = prediction - true_label

To update the weights, we simply multiply this scalar error by the vector of the other word in the pair, scale it by the learning rate, and subtract it from the current weights. To prevent exploding gradients during the dot products, the sigmoid function includes a hard clip to avoid np.exp() overflow errors.
