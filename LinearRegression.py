import numpy as np

# Create random dataset
np.random.seed(42)
data =np.random.rand(100, 8)
print("shape of the data: ", data.shape)


# Create labels 'y'
labels = np.random.randint(10, 40, size=100)
print("shape of the labels: ", labels.shape)

class LinearRegression:
    
    def __init__(self, n: int, n_features: int, lr: float):
        self.learning_rate = lr # learning rate
        self.n = n # size
        self.m = np.random.rand(n_features) # init random values for m
                   
    def calculate_loss(self, y_hat, y):
        # Mean Square Error Loss function
        # 1/n Sum(y - y_hat)**2
        return np.mean((y_hat - y) ** 2)
        
    def predict(self, x_in):
        # y = x*m
        return np.dot(x_in, self.m.T)
    
    def train(self, X, Y, epochs):
        
        for epoch in range(epochs):
            cumulative_loss = 0 # set initial loss to 0
            
            # make a prediction
            Y_hat = self.predict(X)
            
            # calculate loss of the predictions
            loss = self.calculate_loss(Y_hat, Y)
            
            # add loss to total loss
            cumulative_loss += loss
            
            ## Gradient Descent
            # Now we need to compute our MSE gradient
            
            # MSE = 1/n SUM_n (yi - (mxi))^2
            #   Apply Chain Rule
            #       1) 2(yi - (mxi))
            #       2)  (yi - (mxi))
            #       chain: 2(yi - (mxi)) * (yi - (mxi))
            #
            #  Compute derivative w.r.t to m
            # m --> -2/n * (yi - (mxi)) * -xi = -2/n Sum(xi(yi - (mxi)))
            
            grad_m = (-2 / self.n) * np.dot((Y - Y_hat).T, X)
           
            # Update hyper parameter by gradient value, we minimaise so decrease 
            self.m -= self.learning_rate * grad_m
              
            print(f'Epoch: {epoch + 1}, Loss: {cumulative_loss} \n')
                
                
                
# Defining Linear Regression model! --> y = mx
model = LinearRegression(n=100, n_features=8, lr=0.001)

model.train(data, labels, 100)