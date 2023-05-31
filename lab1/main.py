import numpy as np
import matplotlib.pyplot as plt
def generate_XOR_easy():
    inputs = []
    labels = []

    for i in range(11):
        inputs.append([0.1*i,0.1*i])
        labels.append(0)

        if 0.1*i == 0.5:
            continue

        inputs.append([0.1*i,1-0.1*i])
        labels.append(1)
    return np.array(inputs), np.array(labels).reshape(21,1)

def generate_linear(n=100):
    pts = np.random.uniform(0,1,(n,2))
    inputs = []
    labels = []
    for pt in pts:
        inputs.append([pt[0],pt[1]])
        distance = (pt[0]-pt[1])/1.414
        if pt[0]>pt[1]:
            labels.append(0)
        else:
            labels.append(1)
    return np.array(inputs), np.array(labels).reshape(n,1)

def show_result(x,y,pred_y):
    plt.subplot(1,2,1)
    plt.title('Ground truth',fontsize = 18)
    for i in range(x.shape[0]):
        if y[i]==0:
            plt.plot(x[i][0],x[i][1],'ro')
        else:
            plt.plot(x[i][0],x[i][1],'bo')
    plt.subplot(1,2,2)
    plt.title('Predict result',fontsize = 18)
    for i in range(x.shape[0]):
        if pred_y[i]==0:
            plt.plot(x[i][0],x[i][1],'ro')
        else:
            plt.plot(x[i][0],x[i][1],'bo')
    plt.show()

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def derivative_sigmoid(x):
    return np.multiply(x , 1.0-x)

def forward(params,input):
    HiddenLayer_1 = np.matmul(input , params['W1']) + params['b1']
    HiddenLayer_1 = sigmoid(HiddenLayer_1)   
    params['X2'] = HiddenLayer_1
    HiddenLayer_2 = np.matmul(HiddenLayer_1 , params['W2']) + params['b2']
    HiddenLayer_2 = sigmoid(HiddenLayer_2)        ##RELU
    params['X3'] = HiddenLayer_2
    OutputLayer =  np.matmul(HiddenLayer_2 , params['W3']) + params['b3']
    OutputLayer = sigmoid(OutputLayer)
    #print(OutputLayer)
    return OutputLayer , params

def setparams(W2_row,W2_col,input):
    params= {}
    params['X1'] = input
    params['W1'] = np.random.randn(input.shape[1],W2_row) 
    params['b1'] = np.full((1,W2_row),0)
    params['A1'] = 'sigmoid'   
    params['W2'] = np.random.randn(W2_row,W2_col) 
    params['b2'] = np.full((1,W2_col),0)
    params['A2'] = 'RELU'       ##RELU
    params['W3'] = np.random.randn(W2_col,1) 
    params['b3'] = np.full((1,1),0)
    params['A3'] = 'sigmoid'
    opt_v = {}
    for key,value in params.items():    #optimizer
        opt_v[key] = np.zeros_like(params[key])
        #print(key)

    return params,opt_v

def MSE(output,y_true,epoch):
    print('epoch : ',epoch,' loss : ','%.20f' % np.mean(np.square(output-y_true)))
    return np.mean(np.square(output-y_true))

def accuracy(y_pred,y_true):
    #print('accuracy: ',100 * np.sum(y_pred == y_true)/y_pred.shape[0],'%')
    return 100 * np.sum(y_pred == y_true)/y_pred.shape[0]

def backward(params,output,y,y_pred):
    params_grad = {}
    derivative_loss = output-y
    params_grad['b3'] = derivative_sigmoid(output) * derivative_loss
    params_grad['W3'] = np.matmul(np.transpose(params['X3']) , params_grad['b3'])
    params_grad['X3'] = np.matmul(params_grad['b3'] , np.transpose(params['W3']))
    params_grad['b2'] = derivative_sigmoid(params['X3']) *  params_grad['X3'] ##RELU
    params_grad['W2'] = np.matmul(np.transpose(params['X2']) , params_grad['b2'])
    params_grad['X2'] = np.matmul(params_grad['b2'] , np.transpose(params['W2']))
    params_grad['b1'] = derivative_sigmoid(params['X2']) * params_grad['X2']
    params_grad['W1'] = np.matmul(np.transpose(params['X1']) , params_grad['b1'])
    params_grad['X1'] = np.matmul(params_grad['b1'] , np.transpose(params['W1']))
    return params_grad

def back_propogate(params,params_grad,lr=0.01):
    for key,value in params.items():
        if 'W' in key:
            #print(key,value)
            params[key] = params[key] - (lr * params_grad[key])
            #print('a',params_grad[key])
    for key,value in params.items():
        if 'b' in key:
            #print(key,value)
            params[key] = params[key] - (lr * np.sum(params_grad[key],axis=0))
            #print(params[key].shape)
            #print('a',params_grad[key])

    return params

def optimizer_bp(params,params_grad,opt_v,lr=0.01,momentum=0.5):

    for key,value in params.items():
        if 'W' in key:
            opt_v[key] = momentum * opt_v[key] - (lr * params_grad[key])
            params[key] = params[key] + opt_v[key]
            #print('a',params_grad[key])
    for key,value in params.items():
        if 'b' in key:
            #print(key,value)
            opt_v[key] = momentum * opt_v[key] - (lr * np.sum(params_grad[key],axis=0))
            params[key] = params[key] + opt_v[key]
            #print('a',params_grad[key])
    return params

def relu(x):
    return np.maximum(0, x)

def derivative_relu(x):
    relu_grad = np.zeros_like(x)
    relu_grad[x>=0] = 1
    return relu_grad

def loss_diagram(loss):
    plt.title('Learning curve',fontsize = 18)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.plot(np.arange(loss.shape[0]),loss,linewidth=1.5)
    plt.show()

def main():
    option = input('Choose a dataset to test. 1 : linear, 2 : xor ---> ')
    optimize = input('Choose a optimizer to test. 1 : SGD, 2 : Momentum ---> ')
    if option == '1':
        x,y = generate_linear(n=100)
    else:
        x,y =generate_XOR_easy()
    i=0
    params,opt_v = setparams(100,100,x)
    loss = np.array([])
    while True:
        i=i+1
        output , params = forward(params,x)
        y_pred = np.copy(output) 
        y_pred[y_pred > 0.5] = 1
        y_pred[y_pred <= 0.5] = 0
        loss = np.append(loss,np.asscalar(MSE(output,y,i)))
        if accuracy(y_pred,y)>99.9:
            #print('epoch:',i)
            break
        params_grad = backward(params,output,y,y_pred)
        if optimize == '1':
            params = back_propogate(params,params_grad,lr=0.01)
        else:
            params = optimizer_bp(params,params_grad,opt_v,lr=0.01)
    print('Prediction')
    for k in range(output.shape[0]):
        print('Iter : ', k , '\t|\t' , 'Ground Truth : ' , y[k] ,'|\tprdiction: ',output[k],'\t|')
    MSE(output,y,i)
    print('Accuracy : ',accuracy(y_pred,y))
    show_result(x,y,y_pred)
    loss_diagram(loss)

if __name__ == '__main__':
    main()