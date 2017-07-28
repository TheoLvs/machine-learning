import numpy as np

'''
RESOURCES : 
- CS231n Stanford and in particular http://cs231n.github.io/neural-networks-case-study/
'''


class LogisticRegression():
    def __init__(self,learning_rate = 1e-0,random_init = 1e-2,reg = 1e-3,batch_size = 256):
        self.W = []
        self.X = []
        self.y = []
        self.weights = []
        self.training_set = []
        self.targets = []
        self.learning_rate = learning_rate
        self.random_init = random_init
        self.reg = reg
        self.batch_size = batch_size
    

    def train(self,X,y,optimization = "random",n_steps = 1000):
        self.y = y
        self.targets = y
        self.n_samples = len(X)
        self.n_classes = len(np.unique(self.targets))
        size = X.shape[1]+1,self.n_classes, #here it is (10*3072+1)
        if len(self.weights)==0:
            self.weights = np.random.randn(*size)*self.random_init   #initialization

        self.W = self.weights
        #self.X = np.apply_along_axis(lambda x : (x-np.mean(x))/np.std(x),arr = X,axis = 1)
        self.training_set = np.concatenate((X,np.ones((1,self.n_samples)).T),axis = 1)
        self.X = self.training_set
        


        #RANDOM OPTIMIZATION : the weights are updated randomly 1000 times, we keep the best weights
        if optimization == "random":
            print(">> Processing random search for %s attempts"%n_steps)
            loss,probas = self.loss()
            best_loss = loss
            best_W = self.W
            print('... At step 0, the minimum loss is {0:3g}'.format(best_loss))
            for i in range(1,n_steps):
                self.W = np.random.randn(*size)*self.random_init
                loss,probas = self.loss()
                if loss <= best_loss:
                    best_loss = loss
                    best_W = self.W
                    print('... At step {0}, the minimum loss is {1:3g}'.format(i,best_loss))
                elif i >= 10 and i%(n_steps/10) == 0:
                    print('... At step {0}, the minimum loss is {1:3g}'.format(i,best_loss))
                    
            print('>> Updating weights for the best loss found : {0:3g}'.format(best_loss))
            self.W = best_W




        #BLINDFOLDED HIKER OPTIMIZATION : we take a small step and see if it leads to a smaller loss, if so we update the weights
        elif optimization == "blindfolded_hiker":
            print(">> Processing blindfolded hiker optimization for %s attempts"%n_steps)
            loss,probas = self.loss()
            best_loss = loss
            print('... At step 0, the minimum loss is {0:3g}'.format(best_loss))
            for i in range(1,n_steps):
                dW = np.random.randn(*size)*0.0001
                loss,probas = self.loss(weights = self.W + dW)
                if loss < best_loss:
                    best_loss = loss
                    self.W = self.W + dW
                elif i >= 10 and i%(n_steps/10) == 0:
                    print('... At step {0}, the minimum loss is {1:3g}'.format(i,best_loss))                    
            print('>> Updating weights for the best loss found : {0:3g}'.format(best_loss))


        #GRADIENT DESCENT
        elif optimization == "gradient_descent":
            print(">> Processing gradient descent optimization for %s attempts"%n_steps)
            for i in range(0,n_steps):

                loss,probas = self.loss()
                if (i == 0) or (i >= 10 and i%(n_steps/10) == 0) : print('... At step {0}, the loss is {1:3g}'.format(i,loss))

                dW = self.gradient(probas)
                self.W = self.W - self.learning_rate*dW

            print('>> Updating weights for the last loss found : {0:3g}'.format(loss))




        #MINI BATCH GRADIENT DESCENT
        elif optimization == "mini_batch_gradient_descent":
            print(">> Processing mini batch gradient descent optimization for %s attempts and batch size of %s"%(n_steps,self.batch_size))

            
            for i in range(0,n_steps):
                #Selecting batch of data
                batch = np.random.randint(len(self.training_set),size = self.batch_size)
                self.X = self.training_set[batch,:]
                self.y = self.targets[batch]

                loss,probas = self.loss()
                if (i == 0) or (i >= 10 and i%(n_steps/10) == 0) : print('... At step {0}, the loss is {1:3g}'.format(i,loss))

                dW = self.gradient(probas)
                self.W = self.W - self.learning_rate*dW

                    


            print('>> Updating weights for the last loss found : {0:3g}'.format(loss))




        elif optimization == None:
            loss,probas = self.loss()
            best_loss = loss
            print('... At step 0, the minimum loss is {0:3g}'.format(best_loss))

        return loss




    
    def loss(self,weights = []):
        if len(weights) == 0: 
            weights = self.W
        scores = self.X.dot(weights)
        scores = scores - np.max(scores,axis = 1,keepdims = True) #keepdims = True equivalent to .reshape(1,-1).T
        probas = np.exp(scores)/np.sum(np.exp(scores),axis = 1,keepdims = True)
        cross_entropy = -np.log(probas[range(len(probas)),self.y])
        #here insert the regularization parameter
        loss = np.sum(cross_entropy)/len(cross_entropy)
        return loss,probas

    def gradient(self,probas):
        n_examples = len(probas)

        #Gradient on scores
        dscores = probas
        dscores[range(n_examples),self.y] -= 1
        dscores /= n_examples

        #Backpropagation of the gradients into the weights
        dW = self.X.T.dot(dscores) #here add the regularization gradient

        return dW




    def predict(self,X,y = []):
        # X = np.apply_along_axis(lambda x : (x-np.mean(x))/np.std(x),arr = X,axis = 1)
        scores = np.concatenate([X,np.ones((len(X),1))],axis = 1).dot(self.W)
        prediction = np.argmax(scores,axis = 1)
        if len(y)>0:
            print('>> Accuracy of {0:4g}'.format(np.mean(prediction == y)))
        return prediction
            




class NeuralNetwork():
    def __init__(self,n_layers = 2,layer_size = 100,learning_rate = 1e-0,random_init = 1e-2,reg = 1e-3,batch_size = 256):
        self.W = []
        self.X = []
        self.y = []
        self.weights = []
        self.training_set = []
        self.targets = []
        self.learning_rate = learning_rate
        self.random_init = random_init
        self.reg = reg
        self.layer_size = layer_size
        self.batch_size = batch_size


    def train(self,X,y,optimization = "random",n_steps = 1000):
        self.y = y
        self.targets = y
        self.n_samples = len(X)
        self.n_classes = len(np.unique(self.targets))
        size = X.shape[1]+1,self.n_classes, #here it is (10*3072+1)

        for layer in range(n_layers):
            if len(self.weights)==0:
                self.weights += [np.random.randn(*size)*self.random_init]   #initialization

        self.W = self.weights
        #self.X = np.apply_along_axis(lambda x : (x-np.mean(x))/np.std(x),arr = X,axis = 1)
        self.training_set = np.concatenate((X,np.ones((1,self.n_samples)).T),axis = 1)
        self.X = self.training_set
        


        #RANDOM OPTIMIZATION : the weights are updated randomly 1000 times, we keep the best weights
        if optimization == "random":
            print(">> Processing random search for %s attempts"%n_steps)
            loss,probas = self.loss()
            best_loss = loss
            best_W = self.W
            print('... At step 0, the minimum loss is {0:3g}'.format(best_loss))
            for i in range(1,n_steps):
                self.W = np.random.randn(*size)*self.random_init
                loss,probas = self.loss()
                if loss <= best_loss:
                    best_loss = loss
                    best_W = self.W
                    print('... At step {0}, the minimum loss is {1:3g}'.format(i,best_loss))
                elif i >= 10 and i%(n_steps/10) == 0:
                    print('... At step {0}, the minimum loss is {1:3g}'.format(i,best_loss))
                    
            print('>> Updating weights for the best loss found : {0:3g}'.format(best_loss))
            self.W = best_W




        #BLINDFOLDED HIKER OPTIMIZATION : we take a small step and see if it leads to a smaller loss, if so we update the weights
        elif optimization == "blindfolded_hiker":
            print(">> Processing blindfolded hiker optimization for %s attempts"%n_steps)
            loss,probas = self.loss()
            best_loss = loss
            print('... At step 0, the minimum loss is {0:3g}'.format(best_loss))
            for i in range(1,n_steps):
                dW = np.random.randn(*size)*0.0001
                loss,probas = self.loss(weights = self.W + dW)
                if loss < best_loss:
                    best_loss = loss
                    self.W = self.W + dW
                elif i >= 10 and i%(n_steps/10) == 0:
                    print('... At step {0}, the minimum loss is {1:3g}'.format(i,best_loss))                    
            print('>> Updating weights for the best loss found : {0:3g}'.format(best_loss))


        #GRADIENT DESCENT
        elif optimization == "gradient_descent":
            print(">> Processing gradient descent optimization for %s attempts"%n_steps)
            for i in range(0,n_steps):

                loss,probas = self.loss()
                if (i == 0) or (i >= 10 and i%(n_steps/10) == 0) : print('... At step {0}, the loss is {1:3g}'.format(i,loss))

                dW = self.gradient(probas)
                self.W = self.W - self.learning_rate*dW

            print('>> Updating weights for the last loss found : {0:3g}'.format(loss))




        #MINI BATCH GRADIENT DESCENT
        elif optimization == "mini_batch_gradient_descent":
            print(">> Processing mini batch gradient descent optimization for %s attempts and batch size of %s"%(n_steps,self.batch_size))

            
            for i in range(0,n_steps):
                #Selecting batch of data
                batch = np.random.randint(len(self.training_set),size = self.batch_size)
                self.X = self.training_set[batch,:]
                self.y = self.targets[batch]

                loss,probas = self.loss()
                if (i == 0) or (i >= 10 and i%(n_steps/10) == 0) : print('... At step {0}, the loss is {1:3g}'.format(i,loss))

                dW = self.gradient(probas)
                self.W = self.W - self.learning_rate*dW

                    


            print('>> Updating weights for the last loss found : {0:3g}'.format(loss))




        elif optimization == None:
            loss,probas = self.loss()
            best_loss = loss
            print('... At step 0, the minimum loss is {0:3g}'.format(best_loss))

        return loss




    
    def loss(self,weights = []):
        if len(weights) == 0: 
            weights = self.W
        scores = self.X.dot(weights)
        scores = scores - np.max(scores,axis = 1,keepdims = True) #keepdims = True equivalent to .reshape(1,-1).T
        probas = np.exp(scores)/np.sum(np.exp(scores),axis = 1,keepdims = True)
        cross_entropy = -np.log(probas[range(len(probas)),self.y])
        #here insert the regularization parameter
        loss = np.sum(cross_entropy)/len(cross_entropy)
        return loss,probas

    def gradient(self,probas):
        n_examples = len(probas)

        #Gradient on scores
        dscores = probas
        dscores[range(n_examples),self.y] -= 1
        dscores /= n_examples

        #Backpropagation of the gradients into the weights
        dW = self.X.T.dot(dscores) #here add the regularization gradient

        return dW




    def predict(self,X,y = []):
        # X = np.apply_along_axis(lambda x : (x-np.mean(x))/np.std(x),arr = X,axis = 1)
        scores = np.concatenate([X,np.ones((len(X),1))],axis = 1).dot(self.W)
        prediction = np.argmax(scores,axis = 1)
        if len(y)>0:
            print('>> Accuracy of {0:4g}'.format(np.mean(prediction == y)))
        return prediction