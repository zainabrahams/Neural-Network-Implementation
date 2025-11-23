import numpy as np
import argparse
import gzip
import matplotlib.pyplot as plt

#sigmoid activation
def sigmoid(x):
    """
    Computes the sigmoid activation function

    Inputs:
    x (array): array of values

    Outputs:
    array: sigmoid-transformed output in the range (0, 1)
    """
    return 1/(1 + np.exp(-x)) #np.exp used for exponential

def sigmoid_derivative(sigmoid_x): 
    """
    Computes the derivative of the sigmoid activation function

    Inputs:
    sigmoid_x (array): the output values from the sigmoid function

    Outputs:
    array: the derivative of the sigmoid values
    """
    return sigmoid_x * (1 - sigmoid_x)

def output_converter(y, output_neurons):
    """
    Converts numeric class labels into vector

    Inputs:
    y (array): array of class labels
    output_neurons (int): total number of output neurons

    Outputs:
    array: array where each row corresponds to a class label that has been converted into a vector of 0s and 1s 
    e.g. label 2 = [0, 0, 1, 0]
    """
    Y = np.zeros((y.size, output_neurons)) #create matrix filled with zeroes (one row per label, one column per class)
    Y[np.arange(y.size), y.astype(int)] = 1  #places a 1 in the column corresponding to the samples label
    return Y

def load_data(path):
    """
    Loads dataset from a compressed CSV file and separates labels and pixel s matrix.

    Inputs:
    path (str): file path to the compressed CSV dataset

    Outputs:
    X (array): 2D array containing the pixel values of the associated image which has been normalised by dividing by 255.0 to improve training efficiency
    y (array): 1D array containing class labels from the first column of the dataset
    """
    with gzip.open(path, 'rt') as f:
        data = np.loadtxt(f, delimiter=',', dtype=np.float32, skiprows=1)
    y = data[:, 0].astype(int)
    X = data[:, 1:] / 255.0
    return X, y

class NN:
    def __init__(self, n_input, n_hidden, n_output, learning_rate = 0.1, seed = 42):
        """
        Initialises a simple neural network with one hidden layer

        Inputs:
        n_input (int): number of neurons in the input layer
        n_hidden (int): number of neurons in the hidden layer
        n_output (int): number of output neurons
        learning_rate (float): learning rate used in gradient descent updates
        seed (int): random seed for reproducibility (default is 42)

        Attributes:
        weights_1 (array): weight matrix (row = input neurons, column = hidden neurons) connecting the input layer to the hidden layer, initialised with small random values, multiplied by 0.01 so training starts near 0 
        
        weights_2 (array): weight matrix (row = input neurons, column = hidden neurons) connecting the hidden layer to the output layer, initialised with small random values, multiplied by 0.01 so training starts near 0 
        
        learning_rate (float): learning rate used in gradient descent updates
        """
        np.random.seed(seed)
        self.weights_1 = np.random.randn(n_input, n_hidden) * 0.01 #creates a 2D array (matrix) of random numbers (row = input neurons, column = hidden neurons)
        self.weights_2 = np.random.randn(n_hidden, n_output) * 0.01 #creates a 2D array (matrix) of random numbers (row = input neurons, column = hidden neurons)
        self.learning_rate = learning_rate

    def forward(self, X):
        """
        Performs forward propagation through the neural network

        Inputs:
        X (array): input data matrix (number of samples in batch, input_neurons)

        Outputs:
        array: output activations/predictions from the final layer after applying sigmoid activation function (number of samples in batch, output neurons)
        """
        self.Z1 = np.dot(X, self.weights_1) #pre-activated hidden neurons
        self.A1 = sigmoid(self.Z1) #activated hidden neurons
        self.Z2 = np.dot(self.A1, self.weights_2) #pre-activated output neurons
        self.A2 = sigmoid(self.Z2) #activated output neurons
        return self.A2

    def batch_loss(self, A2, Y):
        """
        Calculates the mean squared error (MSE) loss for a batch of predictions

        Inputs:
        A2 (array): predicted output activation from the network
        (number of samples in batch, predicted output_neurons)
        Y (array): true target values 
        (number of samples in batch, true output_neurons)

        Outputs:
        float: mean squared error (MSE) for the entire batch calculated as:
        E = 1/2 * mean((t - y)^2) across all output neurons and samples
        """
        error = (Y-A2)
        return 0.5 * np.mean(np.sum(error * error, axis=1))

    def backward(self, X, Y):
        """
        Performs backpropagation to update the networks weights based on prediction errors

        Inputs:
        X (array): input data matrix (number of samples in batch, input_neurons)
        Y (array): true target values (number of samples in batch, output_neurons)

        Outputs:
        None: Updates network weights in place
        """
        m = X.shape[0] #number of samples in current mini batch
        
        #---- Output layer ----
        dZ2 = (self.A2 - Y) * sigmoid_derivative(self.A2) #dE/dy * dy/dz
        dW2 = np.dot(self.A1.T, dZ2) / m #(dE/dy * dy/dz) * x / m aka error derivative wrt to weight

        #---- Hidden layer ----
        dA1 = np.dot(dZ2, self.weights_2.T)
        dZ1 = dA1 * sigmoid_derivative(self.A1)
        dW1 = np.dot(X.T, dZ1) / m

        #---- Gradient descent update ----
        self.weights_2 = self.weights_2 - (self.learning_rate * dW2)
        self.weights_1 = self.weights_1 - (self.learning_rate * dW1)

    def train(self, X, Y, X_test, y_test, epochs=10, batch_size=32, shuffle_seed=0):
        """
        Trains neural network using mini batches, gradient descent and epochs

        Inputs:
        X (array): training input data matrix with shape (number of training samples, input_neurons)
        Y (array): vector true training labels with shape (number of training samples, label as vector)
        X_test (array): test input data matrix (number of test samples, input_neurons)
        y_test (array): true test labels as integer (number of test samples)
        epochs (int):number of epochs
        batch_size (int): number of samples per mini-batch
        shuffle_seed (int): random seed used for reproducible data shuffling (default = 0)

        Outputs:
        history (dict): stores loss (list of mean batch losses per epoch), train accuracy (list of training accuracies per epoch) and test accuracy (list of test accuracies per epoch)
        """
        n = X.shape[0]
        rng = np.random.default_rng(shuffle_seed)

        #store history for assignment graphs
        history = {"loss": [], "train_accuracy": [], "test_accuracy": []}

        for epoch in range(epochs):
            order = rng.permutation(n) #generate random ordering indices from 0 to 60000
            Xs, Ys = X[order], Y[order] #shuffle data each epoch

            epoch_losses = [] #store a list of the loss for each mini batch within this epoch (calculate epochs total loss)

            #loop through mini-batches
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                Xb, Yb = Xs[start:end], Ys[start:end] #Xb = input(samples in batch, input neurons), #Yb = true labels (samples in batch, true labels)
                A2 = self.forward(Xb) #forward pass
                epoch_losses.append(self.batch_loss(A2, Yb)) #batch loss
                self.backward(Xb, Yb) #backward pass + update

            #accuracy on full training set at end of epoch
            train_preds = np.argmax(self.forward(X), axis=1) #chooses index of  the output neuron with the highest activation to compare against actual label
            train_labels = np.argmax(Y, axis=1) #chooses index with value 1
            acc = (train_preds == train_labels).mean() #average number of correct predictions with train data (60k)

            #accuracy on test set at the end of each epoch
            test_preds = self.predict(X_test) #predicts the label for each sample in the test set, argmax isn't as the test data doesn't have output converter on it
            test_acc = (test_preds == y_test).mean() #average number of correct predictions with test data (10k)

            #epoch loss
            epoch_loss = np.mean(epoch_losses)

            #print metrics
            print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Train Accuracy: {acc*100:.2f}% - Test Accuracy: {test_acc*100:.2f}%")

            #store results
            history["loss"].append(np.mean(epoch_losses))
            history["train_accuracy"].append(acc)
            history["test_accuracy"].append(test_acc)

        return history
    
    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)

#---- main/output ----

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FFNN (sigmoid) with gradient descent and squared-error loss")
    parser.add_argument("NInput", type=int) #input neurons (784)
    parser.add_argument("NHidden", type=int) #hidden neurons
    parser.add_argument("NOutput", type=int) #output neurons (10)
    parser.add_argument("train_path", type=str) #train file
    parser.add_argument("test_path", type=str) #test file
    parser.add_argument("--epochs", type=int, default=10) #number of epochs
    parser.add_argument("--learning_rate", type=float, default=0.1) #learning rate
    parser.add_argument("--batch", type=int, default=32) #number of batches
    parser.add_argument("--seed", type=int, default=42) #seed for reproducibility
    args = parser.parse_args()

    X_train, y_train = load_data(args.train_path) #separate the train data
    X_test, y_test = load_data(args.test_path) #separate the test data
    Y_train = output_converter(y_train, args.NOutput) #convert true label of train data to vector for MSE and backpropagation

    #---- normal test ----

    nn = NN(args.NInput, args.NHidden, args.NOutput, learning_rate=args.learning_rate, seed=args.seed) #initialise neural network
    history = nn.train(X_train, Y_train, X_test, y_test, epochs=args.epochs, batch_size=args.batch, shuffle_seed=args.seed) #train neural network

    print(f"Final Test Accuracy: {history['test_accuracy'][-1]*100:.2f}%")

    #test accuracy
    plt.plot(range(1, args.epochs+1), history["test_accuracy"], marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy")
    plt.title(f"Test Accuracy vs Epochs (η={args.learning_rate}, epochs={args.epochs}, batch size={args.batch}))")
    plt.grid(True)
    plt.savefig("test_accuracy.png")
    plt.show()
    plt.close()



    # #---- test for learning rates ----

    # learning_rates = [0.001, 0.01, 1.0, 10.0, 100.0]
    # histories = {}

    # for rate in learning_rates:
    #     print(f"\nTraining with learning rate η={rate}")
    #     nn = NN(args.NInput, args.NHidden, args.NOutput, learning_rate=rate, seed=args.seed)
    #     hist = nn.train(X_train, Y_train, X_test, y_test, epochs=args.epochs, batch_size=args.batch, shuffle_seed=args.seed)
    #     histories[rate] = hist

    #     #max test accuracy and the epoch it occurs
    #     test_accs = np.array(hist["test_accuracy"])
    #     best_ep = int(test_accs.argmax()) + 1
    #     best_acc = float(test_accs.max())
    #     print(f"η={rate:g} → max test acc = {best_acc*100:.2f}% at epoch {best_ep}")

    # #plot all curves on the same figure
    # import matplotlib.pyplot as plt
    # epochs_axis = range(1, args.epochs + 1)
    # for rate, hist in histories.items():
    #     plt.plot(epochs_axis, hist["test_accuracy"], marker="o", label=f"η={rate}")
    # plt.xlabel("Epoch")
    # plt.ylabel("Test Accuracy")
    # plt.title(f"Test Accuracy vs Epoch (Learning-rate sweep) (epochs={args.epochs}, batch size={args.batch}))")
    # plt.legend()
    # plt.grid(True)
    # plt.savefig("lr_test_accuracy.png")
    # plt.show()


    # #---- test for mini-batches ----
    # batch_sizes = [1, 5, 20, 100, 300]
    # results = []

    # for b in batch_sizes:
    #     print(f"\n=== Training with mini-batch size = {b} ===")
    #     nn = NN(args.NInput, args.NHidden, args.NOutput, learning_rate=args.learning_rate, seed=args.seed)
    #     hist = nn.train(X_train, Y_train, X_test, y_test, epochs=args.epochs, batch_size=b, shuffle_seed=args.seed)

    #     #max test accuracy and the epoch it occurs
    #     test_accs = np.array(hist["test_accuracy"])
    #     best_ep = int(test_accs.argmax()) + 1
    #     best_acc = float(test_accs.max())
    #     print(f"batch={b} | max test acc = {best_acc*100:.2f}% at epoch {best_ep}")
    #     results.append((b, best_acc, best_ep))

    # #plot: max test accuracy vs batch size
    # bs = [r[0] for r in results]
    # best = [r[1] for r in results]

    # plt.figure()
    # plt.plot(bs, best, marker="o")
    # plt.xticks(bs)
    # plt.ylim(0.0, 1.0)
    # plt.xlabel("Mini-batch size")
    # plt.ylabel("Max Test Accuracy")
    # plt.title(f"Max Test Accuracy vs Mini-batch Size (η={args.learning_rate}, epochs={args.epochs})")
    # plt.grid(True)
    # plt.savefig("batch_test_accuracy.png")
    # plt.show()

    # for b, acc, ep in results:
    #     print(f"[Summary] batch={b:<3} | max test acc={acc*100:5.2f}% @ epoch {ep}")