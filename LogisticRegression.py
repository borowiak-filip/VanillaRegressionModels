import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

class GenData:
    
    def __init__(self, size : int, n_features : int, noise_mu : int, noise_std: float):
        # Create random dataset
        self.m = size
        self.n_features = n_features  
        self.mu = noise_mu # noise mean
        self.sigma = noise_std # noise std larger std -> larger noise
    
    
    def create(self):
        data =np.random.rand(self.m, self.n_features)
    
        # creating a noise with the same dimension as the dataset (100,8) 
        noise = np.random.normal(self.mu, self.sigma, data.shape)

        # combine
        noise_data = data + noise 
        print("shape of the data: ", noise_data.shape)
        
        labels = np.random.randint(0, 2, size=self.m)
        print("shape of the labels: ", labels.shape)
        
    
        return [data, noise_data, labels]

    
    def plotData(self, feature_no, labels, data, noise_data):
        
        plt.scatter(labels.T, np.arange(self.m))
        plt.title("Labels Distribution")
        plt.show()
        
        plt.figure(figsize=(10, 5)) 
        plt.plot(data[:, feature_no], label='Original Data', marker='o')
        plt.plot(noise_data[:, feature_no], label='Noisy Data', marker='x') 
        plt.legend() 
        plt.title('Original Data vs Noisy Data') 
        plt.xlabel('Index') 
        plt.ylabel('Value') 
        plt.show() 
        




## Logisitc Regression for binary classification
class LogisticRegression:
    
    def __init__(self, m: int, n_features: int, lr: float):
        self.learning_rate = lr
        self.m = m
        self.weights = np.zeros(n_features, dtype=np.float64)
        self.bias = 0
        self.l2 = 0.1
    
    def calculate_loss(self, y_hat, y):
        # Compute binary cross entropy loss with MAP
            # Maximum a Priori (MAP) works as a regularisation factor L2 decreasing overfitting in our data
            # L = - 1/n Sum (yi * log(pi) + (1 - yi) log(1 - pi))
            # L = negative log-likelihood
            # L + L2_reg = MAP
        binary_cross_entropy_loss =  - np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
        L2_reg = 0.5 * self.l2 * np.sum(self.weights ** 2) # Alpha /2 * W.T * W
        
        return binary_cross_entropy_loss + L2_reg
                        
    def weightsDim(self):
        print(self.weights.shape)
        
    def sigmoid(self, x_in):
        return 1/(1 + np.exp(-x_in)) # Create Probabilities

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias # X * W + B
    
    def train(self, X, Y, epochs):
        for epoch in range(epochs):
            cumulative_loss = 0 # set initial loss to 0
            
            # perform forward propagation
            # logits = sigmoid((X * W) + B)

            # (X * W) + B
            z = self.predict(X) 
            
            # logits are y_hats
            # Acitivate nonlinearity and get probabilities
            logits = self.sigmoid(z) 
            
            loss = self.calculate_loss(logits, Y)
            cumulative_loss += loss
            
            # Gradient DLoss/Dw
            # Chain rule DL/Dw = DL/Dlogits * Dlogits/Dz * Dz/Dw
                # DL/DLogits = (y_hat - y) / y_hat (1 - y_hat)
                # DLogits/Dz = y_hat(1 - y_hat)
                # Dz/Dw = X
                
                # Multiply => DL/Dw = (y_hat - y)*X
                # Add 1/m and sum and we are good to go!
                
            dw = np.dot(X.T, (logits - Y)) / self.m
            db = np.mean(logits - Y)

            # Update
            self.weights -= self.learning_rate * dw
            self.bias    -= self.learning_rate * db
            
            print ("Sample logits improvement: ", logits[:5])
            print(f'Epoch: {epoch + 1}, Loss: {cumulative_loss} \n')
                
            

M = 100
N = 8
Lr = 0.01

dataGen = GenData(size=M, n_features=N, noise_mu=0, noise_std=0.3) 

data, noise_data, labels = dataGen.create()  

# feature_n plots data w.r.t specifc feature column ~ [0 N-1]
# uncomment to visualise data
dataGen.plotData(feature_no=5, labels=labels, data=data, noise_data=noise_data)

       
model = LogisticRegression(M, N, Lr)

model.train(noise_data, labels, epochs=100)

