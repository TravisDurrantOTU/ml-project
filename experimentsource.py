import modelsource as m
import matplotlib.pyplot as plt
import numpy as np
import time

def run_experiment(test_quantity, args, graph="F"):
    match test_quantity:
        case "filters":
             return filter_experiment(args, graph)
        case "conv_neigh":
            return conv_neigh_experiment(args, graph)
        case "max_neigh":
            return max_neigh_experiment(args, graph)
        case "opt":
            return opt_experiment(args, graph)
        case "activation":
            return activation_experiment(args, graph)
        case "epochs":
            return epochs_experiment(args, graph)
        case "batch_size":
            return batch_size_experiment(args, graph)
        case "validation_split":
            return validation_split_experiment(args, graph)
        case "learning_rate":
            return learning_rate_experiment(args, graph)
        case _:
            print("invalid use of command")

#expected args:
#(iteration_start, iteration_end)
def filter_experiment(args, graph):
    # want to store: n, time, test_loss, test_acc
    data = np.zeros((args[1]-args[0]+1, 4))

    for i in range(args[1]-args[0]+1):
        print(f'Current filter count: {args[0]+i}\nRemaining steps: {args[1]-args[0]-i}')
        start = time.time()
        bin, test_loss, test_acc, bin2 = m.train_model(filters=args[0]+i)
        elapsed = time.time()-start
        data[i,0] = i
        data[i,1] = elapsed
        data[i,2] = test_loss
        data[i,3] = test_acc
    
    if graph == "T":
        figure,axis = plt.subplots(1, 2)
        figure.set_figwidth(16)
        axis[0].plot(data[:,0], data[:,1], '*-k')
        axis[0].set_title("Time consumed against number of filters")
        axis[0].set_xlabel("Number of filters")
        axis[0].set_xticks(np.arange(0, args[1]-args[0]+1, step=1))
        axis[0].set_xticklabels(np.arange(args[0], args[1]+1, step=1))
        axis[0].set_ylabel("Time consumed in model compilation (s)")
        axis[1].plot(data[:,0], data[:,3], '*-r')
        axis[1].set_title("Test step accuracy against number of filters")
        axis[1].set_xlabel("Number of filters")
        axis[1].set_xticks(np.arange(0, args[1]-args[0]+1, step=1))
        axis[1].set_xticklabels(np.arange(args[0], args[1]+1, step=1))
        axis[1].set_yticks(np.arange(0.9, 1.0000001, step=0.01))
        axis[1].set_yticklabels(np.arange(90, 100.00001, step=1))
        axis[1].set_ylabel("Test step accuracy (%)")
        plt.savefig("./figures/exp/filter_"+str(args[0])+"-"+str(args[1]))

    return data

# expected args:
# (iteration_start, iteration_end)
def conv_neigh_experiment(args, graph):
    # want to store: n, time, test_loss, test_acc
    data = np.zeros((args[1]-args[0]+1, 4))

    for i in range(args[1]-args[0]+1):
        print(f'Current neighbourhood size: {args[0]+i}x{args[0]+i}\nRemaining steps: {args[1]-args[0]-i}')
        start = time.time()
        bin, test_loss, test_acc, bin2 = m.train_model(conv_neigh=args[0]+i)
        elapsed = time.time()-start
        data[i,0] = i
        data[i,1] = elapsed
        data[i,2] = test_loss
        data[i,3] = test_acc
    
    if graph == "T":
        figure,axis = plt.subplots(1, 2)
        figure.set_figwidth(16)
        axis[0].plot(data[:,0], data[:,1], '*-k')
        axis[0].set_title("Time consumed against square neighbourhood size (Conv2D)")
        axis[0].set_xlabel("Neighbourhood size")
        axis[0].set_xticks(np.arange(0, args[1]-args[0]+1, step=1))
        axis[0].set_xticklabels(np.arange(args[0], args[1]+1, step=1))
        axis[0].set_ylabel("Time consumed in model compilation (s)")
        axis[1].plot(data[:,0], data[:,3], '*-r')
        axis[1].set_title("Test step accuracy against neighbourhood size (Conv2D)")
        axis[1].set_xlabel("Number of filters")
        axis[1].set_xticks(np.arange(0, args[1]-args[0]+1, step=1))
        axis[1].set_xticklabels(np.arange(args[0], args[1]+1, step=1))
        axis[1].set_yticks(np.arange(0.9, 1.0000001, step=0.01))
        axis[1].set_yticklabels(np.arange(90, 100.00001, step=1))
        axis[1].set_ylabel("Test step accuracy (%)")
        plt.savefig("./figures/exp/conv-neighbourhood_"+str(args[0])+"-"+str(args[1]))
        

    return data

# expected args:
# (iteration_start, iteration_end)
def max_neigh_experiment(args, graph):
    # want to store: n, time, test_loss, test_acc
    data = np.zeros((args[1]-args[0]+1, 4))

    for i in range(args[1]-args[0]+1):
        print(f'Current neighbourhood size: {args[0]+i}x{args[0]+i}\nRemaining steps: {args[1]-args[0]-i}')
        start = time.time()
        bin, test_loss, test_acc, bin2 = m.train_model(max_neigh=args[0]+i)
        elapsed = time.time()-start
        data[i,0] = i
        data[i,1] = elapsed
        data[i,2] = test_loss
        data[i,3] = test_acc
    
    if graph == "T":
        figure,axis = plt.subplots(1, 2)
        figure.set_figwidth(16)
        axis[0].plot(data[:,0], data[:,1], '*-k')
        axis[0].set_title("Time consumed against square neighbourhood size (MaxPool)")
        axis[0].set_xlabel("Neighbourhood size")
        axis[0].set_xticks(np.arange(0, args[1]-args[0]+1, step=1))
        axis[0].set_xticklabels(np.arange(args[0], args[1]+1, step=1))
        axis[0].set_ylabel("Time consumed in model compilation (s)")
        axis[1].plot(data[:,0], data[:,3], '*-r')
        axis[1].set_title("Test step accuracy against neighbourhood size (MaxPool)")
        axis[1].set_xlabel("Number of filters")
        axis[1].set_xticks(np.arange(0, args[1]-args[0]+1, step=1))
        axis[1].set_xticklabels(np.arange(args[0], args[1]+1, step=1))
        axis[1].set_yticks(np.arange(0.9, 1.0000001, step=0.01))
        axis[1].set_yticklabels(np.arange(90, 100.00001, step=1))
        axis[1].set_ylabel("Test step accuracy (%)")
        plt.savefig("./figures/exp/max-neighbourhood_"+str(args[0])+"-"+str(args[1]))
        

    return data

# expected args:
# (list of optimizers to test)
# Available options:
# ('SGD', 
# 'Adagrad',
# 'RMSprop', 
# 'Adadelta', 
# 'Adam', 
# 'Adamax', 
# 'Nadam', 
# 'Ftrl')
def opt_experiment(args, graph):
    # want to store: n, time, test_loss, test_acc
    data = np.zeros((len(args), 4))

    for i in range(len(args)):
        print(f'Current optimizer: {args[i]}\nRemaining steps: {len(args)}')
        start = time.time()
        bin, test_loss, test_acc, bin2 = m.train_model(opt=args[i])
        elapsed = time.time()-start
        data[i,0] = i
        data[i,1] = elapsed
        data[i,2] = test_loss
        data[i,3] = test_acc
    
    if graph == "T":
        figure,axis = plt.subplots(1, 2)
        figure.set_figwidth(16)
        axis[0].plot(data[:,0], data[:,1], '*k')
        axis[0].set_title("Time consumed against chosen optimizer")
        axis[0].set_xlabel("Algorithm Name")
        axis[0].set_xticks(np.arange(len(args)))
        axis[0].set_xticklabels(args)
        axis[0].set_ylabel("Time consumed in model compilation (s)")
        axis[1].plot(data[:,0], data[:,3], '*r')
        axis[1].set_title("Test step accuracy against chosen optimizer")
        axis[1].set_xlabel("Algorithm Name")
        axis[1].set_xticks(np.arange(len(args)))
        axis[1].set_xticklabels(args)
        axis[1].set_yticks(np.arange(0.9, 1.0000001, step=0.01))
        axis[1].set_yticklabels(np.arange(90, 100.00001, step=1))
        axis[1].set_ylabel("Test step accuracy (%)")
        plt.savefig("./figures/exp/opt_"+str(args))
        

    return data
# expected args:
# (list of desired functions to test)
# note: will not change the activation function on final-dense or maxpool layers. These are immutable.
# further: this only changes the activation function used in the convolution and first-dense layers
# valid functions are: 
# ('relu', 
# 'sigmoid', 
# 'softmax', 
# 'softplus', 
# 'softsign', 
# 'tanh', 
# 'selu', 
# 'elu', 
# 'exponential', 
# 'leaky_relu', 
# 'relu6', 
# 'silu', 
# 'hard_silu', 
# 'gelu', 
# 'hard_sigmoid', 
# 'linear', 
# 'mish', 
# 'log_softmax')
def activation_experiment(args, graph):
    # want to store: n, time, test_loss, test_acc
    data = np.zeros((len(args), 4))

    for i in range(len(args)):
        print(f'Current function: {args[i]}\nRemaining steps: {len(args)}')
        start = time.time()
        bin, test_loss, test_acc, bin2 = m.train_model(activation=args[i])
        elapsed = time.time()-start
        data[i,0] = i
        data[i,1] = elapsed
        data[i,2] = test_loss
        data[i,3] = test_acc
    
    if graph == "T":
        figure,axis = plt.subplots(1, 2)
        figure.set_figwidth(16)
        axis[0].plot(data[:,0], data[:,1], '*k')
        axis[0].set_title("Time consumed against chosen function")
        axis[0].set_xlabel("Function Shorthand")
        axis[0].set_xticks(np.arange(len(args)))
        axis[0].set_xticklabels(args)
        axis[0].set_ylabel("Time consumed in model compilation (s)")
        axis[1].plot(data[:,0], data[:,3], '*r')
        axis[1].set_title("Test step accuracy against chosen function")
        axis[1].set_xlabel("Function Shorthand")
        axis[1].set_xticks(np.arange(len(args)))
        axis[1].set_xticklabels(args)
        axis[1].set_yticks(np.arange(0.9, 1.0000001, step=0.01))
        axis[1].set_yticklabels(np.arange(90, 100.00001, step=1))
        axis[1].set_ylabel("Test step accuracy (%)")
        plt.savefig("./figures/exp/functions_"+str(args))
        

    return data

#expected args:
#(iteration_start, iteration_end)
def epochs_experiment(args, graph):
    # want to store: n, time, test_loss, test_acc
    data = np.zeros((args[1]-args[0]+1, 4))

    for i in range(args[1]-args[0]+1):
        print(f'Current number of epochs: {args[0]+i}\nRemaining steps: {args[1]-args[0]-i}')
        start = time.time()
        bin, test_loss, test_acc, bin2 = m.train_model(epochs=args[0]+i)
        elapsed = time.time()-start
        data[i,0] = i
        data[i,1] = elapsed
        data[i,2] = test_loss
        data[i,3] = test_acc
    
    if graph == "T":
        figure,axis = plt.subplots(1, 2)
        figure.set_figwidth(16)
        axis[0].plot(data[:,0], data[:,1], '*-k')
        axis[0].set_title("Time consumed against number of epochs")
        axis[0].set_xlabel("Number of epochs")
        axis[0].set_xticks(np.arange(0, args[1]-args[0]+1, step=1))
        axis[0].set_xticklabels(np.arange(args[0], args[1]+1, step=1))
        axis[0].set_ylabel("Time consumed in model compilation (s)")
        axis[1].plot(data[:,0], data[:,3], '*-r')
        axis[1].set_title("Test step accuracy against number of epochs")
        axis[1].set_xlabel("Number of epochs")
        axis[1].set_xticks(np.arange(0, args[1]-args[0]+1, step=1))
        axis[1].set_xticklabels(np.arange(args[0], args[1]+1, step=1))
        axis[1].set_yticks(np.arange(0.9, 1.0000001, step=0.01))
        axis[1].set_yticklabels(np.arange(90, 100.00001, step=1))
        axis[1].set_ylabel("Test step accuracy (%)")
        plt.savefig("./figures/exp/epochs"+str(args[0])+"-"+str(args[1]))
        

    return data

#expected args:
#note: probably not a useful experiment. batch size too large just cuts your epochs down, they're very interdependent quantities
#(iteration_start, iteration_end)
def batch_size_experiment(args, graph):
    # want to store: n, time, test_loss, test_acc
    data = np.zeros((args[1]-args[0]+1, 4))

    for i in range(args[1]-args[0]+1):
        print(f'Current batch size: {args[0]+i}\nRemaining steps: {args[1]-args[0]-i}')
        start = time.time()
        bin, test_loss, test_acc, bin2 = m.train_model(batch_size=args[0]+i)
        elapsed = time.time()-start
        data[i,0] = i
        data[i,1] = elapsed
        data[i,2] = test_loss
        data[i,3] = test_acc
    
    if graph == "T":
        figure,axis = plt.subplots(1, 2)
        figure.set_figwidth(16)
        axis[0].plot(data[:,0], data[:,1], '*-k')
        axis[0].set_title("Time consumed against batch size")
        axis[0].set_xlabel("Batch size")
        axis[0].set_xticks(np.arange(0, args[1]-args[0]+1, step=1))
        axis[0].set_xticklabels(np.arange(args[0], args[1]+1, step=1))
        axis[0].set_ylabel("Time consumed in model compilation (s)")
        axis[1].plot(data[:,0], data[:,3], '*-r')
        axis[1].set_title("Test step accuracy against batch size")
        axis[1].set_xlabel("Batch size")
        axis[1].set_xticks(np.arange(0, args[1]-args[0]+1, step=1))
        axis[1].set_xticklabels(np.arange(args[0], args[1]+1, step=1))
        axis[1].set_yticks(np.arange(0.9, 1.0000001, step=0.01))
        axis[1].set_yticklabels(np.arange(90, 100.00001, step=1))
        axis[1].set_ylabel("Test step accuracy (%)")
        plt.savefig("./figures/exp/batch_"+str(args[0])+"-"+str(args[1]))
        

    return data

#expected args:
#(iteration_start, iteration_end)
#note, will be scaled down by a factor of 100.
#So the input (1,20) will test val. splits from 1% (0.01) to 20% (0.2)
def validation_split_experiment(args, graph):
    # want to store: n, time, test_loss, test_acc
    data = np.zeros((args[1]-args[0]+1, 4))

    for i in range(args[1]-args[0]+1):
        print(f'Current split: {args[0]+i}\nRemaining steps: {args[1]-args[0]-i}')
        start = time.time()
        bin, test_loss, test_acc, bin2 = m.train_model(validation_split=(args[0]+i)/100)
        elapsed = time.time()-start
        data[i,0] = i
        data[i,1] = elapsed
        data[i,2] = test_loss
        data[i,3] = test_acc
    
    if graph == "T":
        figure,axis = plt.subplots(1, 2)
        figure.set_figwidth(16)
        axis[0].plot(data[:,0], data[:,1], '*-k')
        axis[0].set_title("Time consumed against validation split")
        axis[0].set_xlabel("Validation split (%)")
        axis[0].set_xticks(np.arange(0, args[1]-args[0]+1, step=1))
        axis[0].set_xticklabels(np.arange(args[0], args[1]+1, step=1))
        axis[0].set_ylabel("Time consumed in model compilation (s)")
        axis[1].plot(data[:,0], data[:,3], '*-r')
        axis[1].set_title("Test step accuracy against validation split")
        axis[1].set_xlabel("Validation split (%)")
        axis[1].set_xticks(np.arange(0, args[1]-args[0]+1, step=1))
        axis[1].set_xticklabels(np.arange(args[0], args[1]+1, step=1))
        axis[1].set_yticks(np.arange(0.9, 1.0000001, step=0.01))
        axis[1].set_yticklabels(np.arange(90, 100.00001, step=1))
        axis[1].set_ylabel("Test step accuracy (%)")
        plt.savefig("./figures/exp/validation_"+str(args[0])+"-"+str(args[1]))
        
    
    return data

#expected args:
#(iteration_start, iteration_end)
#note: will be inputted as 1E(-i)
def learning_rate_experiment(args, graph):
    # want to store: n, time, test_loss, test_acc
    data = np.zeros((args[1]-args[0]+1, 4))

    for i in range(args[1]-args[0]+1):
        print(f'Current learning rate: 1*10^-({args[0]+i})\nRemaining steps: {args[1]-args[0]-i}')
        start = time.time()
        bin, test_loss, test_acc, bin2 = m.train_model(learning_rate=1*(10**(-args[0]-i)))
        elapsed = time.time()-start
        data[i,0] = i
        data[i,1] = elapsed
        data[i,2] = test_loss
        data[i,3] = test_acc
    
    if graph == "T":
        figure,axis = plt.subplots(1, 2)
        figure.set_figwidth(16)
        axis[0].plot(data[:,0], data[:,1], '*-k')
        axis[0].set_title("Time consumed against learning rate")
        axis[0].set_xlabel("Validation split (1e-n)")
        axis[0].set_xticks(np.arange(0, args[1]-args[0]+1, step=1))
        axis[0].set_xticklabels(np.arange(args[0], args[1]+1, step=1))
        axis[0].set_ylabel("Time consumed in model compilation (s)")
        axis[1].plot(data[:,0], data[:,3], '*-r')
        axis[1].set_title("Test step accuracy against learning rate")
        axis[1].set_xlabel("Validation split (1e-n)")
        axis[1].set_xticks(np.arange(0, args[1]-args[0]+1, step=1))
        axis[1].set_xticklabels(np.arange(args[0], args[1]+1, step=1))
        axis[1].set_yticks(np.arange(0.9, 1.0000001, step=0.01))
        axis[1].set_yticklabels(np.arange(90, 100.00001, step=1))
        axis[1].set_ylabel("Test step accuracy (%)")
        plt.savefig("./figures/exp/learn-rate_"+str(args[0])+"-"+str(args[1]))
        
    
    return data

#run_experiment('filters', (3, 25), graph='T')
#run_experiment('conv_neigh', (1, 8), graph='T')
#run_experiment('max_neigh', (1, 6), graph='T')
#run_experiment('opt', ['adagrad', 'rmsprop', 'adamax', 'adam', 'nadam', 'adadelta', 'ftrl'], graph='T')
#run_experiment('activation', ['relu', 'gelu', 'elu', 'linear', 'tanh', 'sigmoid', 'hard_sigmoid'], graph='T')
#run_experiment('epochs', (3, 40), graph='T')
#run_experiment('validation_split', (1, 20), graph='T')
#run_experiment('learning_rate', (1, 5), graph='T')
# for greedy model this was the order i picked parameters in
# epochs := 15
# opt := rmsprop
# act := relu, doesn't seem to be any meaningful effect with selections
# filters := 16
# c_n := 7
# m_n := 3
m.train_model() #0.9690
m.naive_model() #0.9840
m.greedy_model() #0.9840
