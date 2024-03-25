import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import time


### GLOBALS ###
RATE = f'lerning rate {{}}'
COLORS = ['red', 'blue', 'green', 'darkorange' , 'black' , 'purple' , 'darkblue'  , 'brown' , 'grey' ]
BATCH_SIZE =[1, 16, 128, 1024]
BEST_LERNING_RATE = 0.01
BEST_EPOCH = 50
BEST_BATCH_S = 1024
DEPTH = [1,2,6,10,6,6,6]
WIDTHS = [16,16,16,16,8,32,64]

### FUNCTIONS ###
def plot_decision_boundaries(model, X, y, title='Decision Boundaries', implicit_repr=False):
    """
    Plots decision boundaries of a classifier and colors the space by the prediction of each point.

    Parameters:
    - model: The trained classifier (sklearn model).
    - X: Numpy Feature matrix.
    - y: Numpy array of Labels.
    - title: Title for the plot.
    """
    # h = .02  # Step size in the mesh

    # enumerate y
    y_map = {v: i for i, v in enumerate(np.unique(y))}
    enum_y = np.array([y_map[v] for v in y]).astype(int)




    h_x = (np.max(X[:, 0]) - np.min(X[:, 0])) / 200
    h_y = (np.max(X[:, 1]) - np.min(X[:, 1])) / 200

    # Plot the decision boundary.
    added_margin_x = h_x * 20
    added_margin_y = h_y * 20
    x_min, x_max = X[:, 0].min() - added_margin_x, X[:, 0].max() + added_margin_x
    y_min, y_max = X[:, 1].min() - added_margin_y, X[:, 1].max() + added_margin_y
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h_x), np.arange(y_min, y_max, h_y))

    # Make predictions on the meshgrid points.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if implicit_repr:
        model_inp = np.c_[xx.ravel(), yy.ravel()]
        new_model_inp = np.zeros((model_inp.shape[0], model_inp.shape[1] * 10))
        alphas = np.arange(0.1, 1.05, 0.1)
        for i in range(model_inp.shape[1]):
            for j, a in enumerate(alphas):
                new_model_inp[:, i * len(alphas) + j] = np.sin(a * model_inp[:, i])
        model_inp = torch.tensor(new_model_inp, dtype=torch.float32, device=device)
    else:
        model_inp = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32, device=device)
    with torch.no_grad():
        Z = model(model_inp).argmax(dim=1).cpu().numpy()
    Z = np.array([y_map[v] for v in Z])
    Z = Z.reshape(xx.shape)
    vmin = np.min([np.min(enum_y), np.min(Z)])
    vmax = np.min([np.max(enum_y), np.max(Z)])

    # Plot the decision boundary.
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8, vmin=vmin, vmax=vmax)

    # Scatter plot of the data points with matching colors.
    plt.scatter(X[:, 0], X[:, 1], c=enum_y, cmap=plt.cm.Paired, edgecolors='k', s=40, alpha=0.7, vmin=vmin, vmax=vmax)

    plt.title("Decision Boundaries")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(title)
    plt.show()

def read_data(filename):
    """
    Read the data from the csv file and return the features and labels as numpy arrays.
    """

    # the data in pandas dataframe format
    df = pd.read_csv(filename)

    # extract the column names
    col_names = list(df.columns)

    # the data in numpy array format
    features = df[['long','lat']].values

    labels = None

    if (len(col_names) != 2):
        labels = df['country'].values

    return features, labels

def chart_data(data ,row_headers, column_headers):
    df = pd.DataFrame(data, index=row_headers, columns=column_headers  )
    return df

def plot_batch_size(train_lst , val_lst , test_lst , label  ):
    for i in range(len(BATCH_SIZE)):
        plt.figure()
        if i == 0:
            plt.bar(['Train' , 'Val' , 'Test'] , [train_lst[0][-1] , val_lst[0][-1] , test_lst[0][-1]], color=COLORS[0], width=0.5)
            plt.title(f'Batch Size - {BATCH_SIZE[i] }  - {label}')
            plt.show()
        else: 
            plt.plot(train_lst[i], label='Train', color=COLORS[0])
            plt.plot(val_lst[i], label='Validation', color=COLORS[1])
            plt.plot(test_lst[i], label='Test', color=COLORS[2])

            plt.title(f'Batch Size - {BATCH_SIZE[i] } - {label}') 
            plt.legend()
            plt.show()
            
    #         plt.plot(data[i], label=f'{label} - batch size {BATCH_SIZE[i]}' , color=COLORS[i])

    # if 
    # plt.figure()
    # plt.bar(BATCH_SIZE, data, color=COLORS[0], width=3)
    # plt.title(f'The batch size vs {label}')
    # plt.xlabel('batch size')
    # plt.ylabel(label)
    # #plt.xticks(BATCH_SIZE)
    # plt.legend()
    # plt.show()

    # plt.figure()
    # plt.plot(train_losses, label='Train', color='red')
    # plt.plot(val_losses, label='Val', color='blue')
    # plt.plot(test_losses, label='Test', color='green')
    # plt.title('Losses')
    # plt.legend()
    # plt.show()

    #     # Plot the  each model acorrding to batch size and epochs
    # plt.figure()
    # for i in range(len(BATCH_SIZE)):
    #     plt.bar(epochs, validation_loss_lst, color=COLORS[0], width=3)
    #     plt.plot(data[i], label=f'{label} - batch size {BATCH_SIZE[i]}' , color=COLORS[i])

    # plt.title(f'Batch Size - {label} ')
    # plt.xlabel('epochs')
    # plt.ylabel('loss')
    # plt.legend()
    # plt.show()


### 6.1.1 Task ### 
def train_model(train_data, val_data, test_data, model, lr=0.001, epochs=50, batch_size=256):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print('Using device:', device)

    trainset = torch.utils.data.TensorDataset(torch.tensor(train_data[['long', 'lat']].values).float(), torch.tensor(train_data['country'].values).long())
    valset = torch.utils.data.TensorDataset(torch.tensor(val_data[['long', 'lat']].values).float(), torch.tensor(val_data['country'].values).long())
    testset = torch.utils.data.TensorDataset(torch.tensor(test_data[['long', 'lat']].values).float(), torch.tensor(test_data['country'].values).long())

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    valloader = torch.utils.data.DataLoader(valset, batch_size=1024, shuffle=False, num_workers=0)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1024, shuffle=False, num_workers=0)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_accs = []
    val_accs = []
    test_accs = []
    train_losses = []
    val_losses = []
    test_losses = []

    for ep in range(epochs):
        model.train()
        pred_correct = 0
        ep_loss = 0.
        # perform a training iteration
        for i, (inputs, labels) in enumerate(tqdm(trainloader)):
            # move the inputs and labels to the device
            inputs, labels = inputs.to(device , dtype=torch.float), labels.to(device)

            # zero the gradients
            optimizer.zero_grad()

            # forward pass
            outputs = model(inputs)

            # calculate the loss
            ##############################
            if batch_size > 16:
                loss = criterion(outputs.squeeze(), labels)
            else:
                loss = criterion(outputs, labels)
            
            ##############################
            
            # backward pass
            loss.backward()
            # update the weights
            optimizer.step()

            # name the model outputs "outputs"
            # model_outputs = outputs
            # # and the loss "loss"
            # model_loss = loss
            
            pred_correct += (torch.argmax(outputs, dim=1) == labels).sum().item()
            ep_loss += loss.item()

        train_accs.append(pred_correct / len(trainset))
        train_losses.append(ep_loss / len(trainloader))

        model.eval()
        with torch.no_grad():
            for loader, accs, losses in zip([valloader, testloader], [val_accs, test_accs], [val_losses, test_losses]):
                correct = 0
                total = 0
                ep_loss = 0.
                # perform an evaluation iteration
                for inputs, labels in loader:

                    # move the inputs and labels to the device
                    inputs, labels = inputs.to(device , dtype=torch.float), labels.to(device)
                    # forward pass
                    outputs = model(inputs)

                    # calculate the loss
                    loss = criterion(outputs, labels)
 


                    # name the model outputs "outputs"
                    # and the loss "loss"

                    ep_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                accs.append(correct / total)
                losses.append(ep_loss / len(loader))

        print('Epoch {:}, Train Acc: {:.3f}, Val Acc: {:.3f}, Test Acc: {:.3f}'.format(ep, train_accs[-1], val_accs[-1], test_accs[-1]))

    return model, train_accs, val_accs, test_accs, train_losses, val_losses, test_losses

### 6.1.2 Questions ###
def Q_6_1_2(train_data):
    print('Q_6_1_2')
    # Train the network with learning rates of: [1., 0.01, 0.001, 0.00001]
    torch.manual_seed(42)
    np.random.seed(42)
    learning_rate = [1., 0.01, 0.001, 0.00001]
    
    output_dim = len(train_data['country'].unique())

    validation_loss_lst = []
    for lr in learning_rate:
        # initialize the model
        model = [nn.Linear(2, 16), nn.ReLU(),  # hidden layer 1
                nn.Linear(16, 16), nn.ReLU(),  # hidden layer 2
                nn.Linear(16, 16), nn.ReLU(),  # hidden layer 3
                nn.Linear(16, 16), nn.ReLU(),  # hidden layer 4
                nn.Linear(16, 16), nn.ReLU(),  # hidden layer 5
                nn.Linear(16, 16), nn.ReLU(),  # hidden layer 6
                nn.Linear(16, output_dim)  # output layer
                ]
        # convert the model to a sequential model
        model = nn.Sequential(*model)
        
        # train the model
        model, train_accs, val_accs, test_accs, train_losses, val_losses, test_losses = train_model(train_data, val_data, test_data, model, lr=lr, epochs=50, batch_size=256)
        
        # append the validation losses to the list
        validation_loss_lst.append(val_losses)

        # Plot the validation loss of each epoch for 0.01 learning rate to compere in the end
        if (lr == 0.01):
            plt.figure()
            plt.plot(train_losses, label='Train', color='red')
            plt.plot(val_losses, label='Val', color='blue')
            plt.plot(test_losses, label='Test', color='green')
            plt.title('Losses')
            plt.legend()
            plt.show()

            plt.figure()
            plt.plot(train_accs, label='Train', color='red')
            plt.plot(val_accs, label='Val', color='blue')
            plt.plot(test_accs, label='Test', color='green')
            plt.title('Accs.')
            plt.legend()
            plt.show()



    # Plot the validation loss of each epoch for each learning rate
    # x axis - epochs, y axis - loss
    
    plt.figure()
    plt.plot(validation_loss_lst[0], label=RATE.format(learning_rate[0]) , color=COLORS[0])
    plt.plot(validation_loss_lst[1], label=RATE.format(learning_rate[1]) , color=COLORS[1])
    plt.plot(validation_loss_lst[2], label=RATE.format(learning_rate[2]) , color=COLORS[2])
    plt.plot(validation_loss_lst[3], label=RATE.format(learning_rate[3]) , color=COLORS[3])
    plt.title('validation losses ')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.show()
    return validation_loss_lst
        
        

def Q_6_1_3(train_data):
    """
    This function trains the network for 100 epochs. For the following epochs: 1,5,10,20,50,100, and plota the loss over the validation.
    param : train_data : pd.DataFrame
    return : None
    """
    print('Q_6_1_3')
    torch.manual_seed(42)
    np.random.seed(42)
    output_dim = len(train_data['country'].unique())

    model = [nn.Linear(2, 16), nn.ReLU(),  # hidden layer 1
             nn.Linear(16, 16), nn.ReLU(),  # hidden layer 2
             nn.Linear(16, 16), nn.ReLU(),  # hidden layer 3
             nn.Linear(16, 16), nn.ReLU(),  # hidden layer 4
             nn.Linear(16, 16), nn.ReLU(),  # hidden layer 5
             nn.Linear(16, 16), nn.ReLU(),  # hidden layer 6
             nn.Linear(16, output_dim)  # output layer
             ]
    model = nn.Sequential(*model)

    model, train_accs, val_accs, test_accs, train_losses, val_losses, test_losses = train_model(train_data, val_data, test_data, model, lr=0.01, epochs=100, batch_size=256)
    
    
    epochs = [1,5,10,20,50,100]
    validation_loss_lst = []

    for epoch in epochs:

        # Subtract 1 because epoch indices are 0-based
        validation_loss_lst.append(val_losses[epoch - 1])

       

    # Plot the validation loss of each epoch for each learning rate
    # x axis - epochs, y axis - loss
    plt.figure()
    plt.bar(epochs, validation_loss_lst, color=COLORS[0], width=3)
    plt.title('Chosen epochs validation losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.xticks(epochs)
    plt.legend()
    plt.show()

def Q_6_1_4(train_data ):
    """
    Batch Norm - this function adds a batch norm (nn.BatchNorm1d) after each hidden layer. 
    Compares the results of the regular and modified model as before
    param : train_data : pd.DataFrame
    return : None
    """
    print('Q_6_1_4')
    torch.manual_seed(42)
    np.random.seed(42)
    output_dim = len(train_data['country'].unique())

    model = [nn.Linear(2, 16), nn.BatchNorm1d(16), nn.ReLU(),  # hidden layer 1
                nn.Linear(16, 16), nn.BatchNorm1d(16), nn.ReLU(),  # hidden layer 2
                nn.Linear(16, 16), nn.BatchNorm1d(16), nn.ReLU(),  # hidden layer 3
                nn.Linear(16, 16), nn.BatchNorm1d(16), nn.ReLU(),  # hidden layer 4
                nn.Linear(16, 16), nn.BatchNorm1d(16), nn.ReLU(),  # hidden layer 5
                nn.Linear(16, 16), nn.BatchNorm1d(16), nn.ReLU(),  # hidden layer 6
                nn.Linear(16, output_dim)  # output layer
                ]
             
    model = nn.Sequential(*model)
    model, train_accs, val_accs, test_accs, train_losses, val_losses, test_losses = train_model(train_data, val_data, test_data, model, lr=0.01, epochs=50, batch_size=256)
    
    # Plot the validation loss of each epoch for each learning rate
    plt.figure()
    plt.plot(train_losses, label='Train', color='red')
    plt.plot(val_losses, label='Val', color='blue')
    plt.plot(test_losses, label='Test', color='green')
    plt.title('Batch Norm - Losses')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(train_accs, label='Train', color='red')
    plt.plot(val_accs, label='Val', color='blue')
    plt.plot(test_accs, label='Test', color='green')
    plt.title('Batch Norm - Accs.')
    plt.legend()
    plt.show()


def test_model(model, X_test):
    """
    Calculates the time it takes the model to test.
    :param model: nn.Sequential
    :param X_test: np.array
    :return: elapsed_time: float
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Move input data to the appropriate device (CPU or GPU)
    inputs = torch.tensor(X_test, dtype=torch.float32).to(device) 

    # Measure the time for model inference
    start_time = time.time()
    outputs = model(inputs)
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time
    return elapsed_time


def Q_6_1_5(train_data , X_test):

    """Batch Size - Train the network with batch sizes of: 1, 16, 128, 1024. For a reasonable training time, 
    change the epochs number according to the table below. 
    and calculates the time(test) it toakes the mode for each bath size and epochs
     Batch Size 1 16 128 1024 
     Epochs 1 10 50 50

     Param : train_data : pd.DataFrame
     return : None
"""

    print('Q_6_1_5')
    torch.manual_seed(42)
    np.random.seed(42)
    output_dim = len(train_data['country'].unique())

    epochs = [1, 10, 50, 50]
    models = []
    train_losses_lst = []
    val_losses_lst = []
    test_losses_lst = []
    train_accs_lst = []
    val_accs_lst = []
    test_accs_lst = []
    validation_loss = []
    test_loss = []
    test_acc = []
    validation_acc = []
    
    for i in range(len(BATCH_SIZE)):
        # initialize the model
        model = [nn.Linear(2, 16), nn.ReLU(),  # hidden layer 1
                 nn.Linear(16, 16), nn.ReLU(),  # hidden layer 2
                 nn.Linear(16, 16), nn.ReLU(),  # hidden layer 3
                 nn.Linear(16, 16), nn.ReLU(),  # hidden layer 4
                 nn.Linear(16, 16), nn.ReLU(),  # hidden layer 5
                 nn.Linear(16, 16), nn.ReLU(),  # hidden layer 6
                 nn.Linear(16, output_dim)  # output layer
                 ]
        # convert the model to a sequential model
        model = nn.Sequential(*model)
    
        model, train_accs, val_accs, test_accs, train_losses, val_losses, test_losses = train_model(train_data, val_data, test_data, model, lr=0.01, epochs=epochs[i], batch_size=BATCH_SIZE[i])
        
        train_losses_lst.append(train_losses)
        # validation loss
        val_losses_lst.append(val_losses)
        validation_loss.append(val_losses[-1])
        
        # test loss
        test_losses_lst.append(test_losses)
        test_loss.append(test_losses[-1])

        train_accs_lst.append(train_accs)

        # validation accuracy
        val_accs_lst.append(val_accs)
        validation_acc.append(val_accs[-1])
        
        # test accuracy
        test_accs_lst.append(test_accs)
        test_acc.append(test_accs[-1])

        models.append(model)

    plot_batch_size(train_losses_lst , val_losses_lst , test_losses_lst , "Loss" )
    plot_batch_size(train_accs_lst , val_accs_lst , test_accs_lst , "Accracy")



    # Calculate the time it takes for each model to test
    times = []
    for mod in models:
        times.append(test_model(mod, X_test))

    print(times)
    # Plot the time it takes for each model to test
    plt.figure()
    plt.bar(BATCH_SIZE, times, color=COLORS[0], width=5)
    plt.title('Time it takes for each model to test')
    plt.xlabel('batch size')
    plt.ylabel('time')
    plt.legend()
    plt.show()
    
def model_maker(depth ,width , output_dim ):
    """
    This function creates a model with the given depth and width
    param : depth : int
    param : width : int
    param : output_dim : int
    return : list
    """
    
    model = [nn.Linear(2, width), nn.BatchNorm1d(width) , nn.ReLU()]  # hidden layer 1

    for i in range(depth -1 ):
        print('hidden layer ', i +1 )
        model.append(nn.Linear(width, width))
        model.append(nn.BatchNorm1d(width))
        model.append(nn.ReLU())
    
    # output layer
    model.append(nn.Linear(width, output_dim))
    return model


### 6.2 Evaluating MLPs Performance ###
def Q_6_2(train_data , X_test , Y_test):
    """In this section you will train several MLPs and analyze their algorithmic choices
    Train 8 classifiers for the following combinations of depth (number of hidden layers) and width (number of neurons in each hidden layer):
    Depth 1  2  6  10 6  6 6 
    Width 16 16 16 16 8 32 64  
    param : train_data : pd.DataFrame
    param : X_test : np.array
    param : Y_test : np.array
    return : list

    """
    torch.manual_seed(42)
    np.random.seed(42)

    output_dim = len(train_data['country'].unique())
    depths = [1,2,6,10,6,6,6]
    widths = [16,16,16,16,8,32,64]
    # A list to save the modes validation accuracy
    val_final_accs = []
    models = []
    for i in range(len(depths)):
        print('model number ', i)
        model = model_maker(depths[i] , widths[i] , output_dim)
        model = nn.Sequential(*model)

        model, train_accs, val_accs, test_accs, train_losses, val_losses, test_losses = train_model(train_data, val_data, test_data, model, lr=0.01, epochs=50, batch_size=256)
        val_final_accs.append((val_accs[-1] , depths[i] , widths[i]))
        models.append((model ,train_accs[-1], val_accs[-1], test_accs[-1] ))


        print("Model with depth: ", depths[i], " and width: ", widths[i])

        # Plot the validation loss of each epoch for each learning rate
        plt.figure()
        plt.plot(train_losses, label='Train', color='red')
        plt.plot(val_losses, label='Val', color='blue')
        plt.plot(test_losses, label='Test', color='green')
        plt.title(f'Losses of model with depth: {depths[i]} and width: {widths[i]}')
        plt.legend()
        plt.show()

        plt.figure()
        plt.plot(train_accs, label='Train', color='red')
        plt.plot(val_accs, label='Val', color='blue')
        plt.plot(test_accs, label='Test', color='green')
        plt.title(f'Accs. of model with depth: {depths[i]} and width: {widths[i]}')
        plt.legend()
        plt.show()
        
        plot_decision_boundaries(model, X_test, Y_test, f'Depth: {depths[i]}, Width: {widths[i]}')
        print(f'Finished training model with depth: {depths[i]} and width: {widths[i]}')

    # print the model with the highest validation accuracy depth and width
    print("The model with the highest validation accuracy is: " , max([val_final_accs[i][0] for i in range(len(val_final_accs))]))
    print("The model with the lowest validation accuracy is: ", min([val_final_accs[i][0] for i in range(len(val_final_accs))]))

    return models


def Q_6_3(models):
    """"Using only the MLPs of width 16, plot the training, validation and test accuracy of the models vs. number of hidden layers. 
    (x axis - number of hidden layers, y axis - accuracy).
    param : models : list
    return : None
    """
    width_16_models = [models[0], models[1] , models[2] , models[3]]
    depth_16 = DEPTH[:4]
 

    train_lst = []
    val_lst = []
    test_lst = []

    for i in range(len(width_16_models)):
        train_lst.append(width_16_models[i][1])
        val_lst.append(width_16_models[i][2])
        test_lst.append(width_16_models[i][3])

    plt.figure()
    plt.plot(depth_16 , train_lst , label='Train', color='red')
    plt.plot(depth_16, val_lst , label='Validation', color='blue')
    plt.plot( depth_16, test_lst , label='Test', color='green')

    plt.title('Accuracies vs Depths')
    plt.xlabel('Number of hidden layers')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


def Q_6_4(models):
    """Using only the MLPs of depth 6, 
    plot the training, validation and test accuracy of the models vs. number of neuron in each hidden layer.
      (x axis - number of neurons, y axis - accuracy).
      Param : models : list
        return : None
      """
    depth_6_models = [models[2] ,models[4], models[5] , models[6]]
    width_6 = [8, 16, 32, 64]
    train_lst = []
    val_lst = []
    test_lst = []
    for i in range(len(depth_6_models)):
        train_lst.append(depth_6_models[i][1])
        val_lst.append(depth_6_models[i][2])
        test_lst.append(depth_6_models[i][3])
        
    
    plt.figure()
    plt.plot(width_6 , train_lst, label='Train', color='red')
    plt.plot(width_6, val_lst, label='Validation', color='blue')
    plt.plot(width_6, test_lst , label='Test', color='green')
    plt.title('Accuracies vs Widths')
    plt.xlabel('Number of neurons')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

def Q_6_5_train(train_data, val_data, test_data, model, grant_layers , lr=0.001, epochs=50, batch_size=256 ):
    """This function ttains the model but this time keep the magnitude of the gradients 
    for each layer, through the training steps.
    The magnitude of of the gradients is defined by: grad magnitude = ||grad||2. 
    param : train_data : pd.DataFrame
    param : val_data : pd.DataFrame
    param : test_data : pd.DataFrame
    param : model : nn.Sequential
    param : lr : float
    param : epochs : int
    param : batch_size : int
    return : nn.Sequential, list, list, list, list, list, list
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print('Using device:', device)

    # create the datasets and dataloaders
    trainset = torch.utils.data.TensorDataset(torch.tensor(train_data[['long', 'lat']].values).float(), torch.tensor(train_data['country'].values).long())
    valset = torch.utils.data.TensorDataset(torch.tensor(val_data[['long', 'lat']].values).float(), torch.tensor(val_data['country'].values).long())
    testset = torch.utils.data.TensorDataset(torch.tensor(test_data[['long', 'lat']].values).float(), torch.tensor(test_data['country'].values).long())

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    valloader = torch.utils.data.DataLoader(valset, batch_size=1024, shuffle=False, num_workers=0)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1024, shuffle=False, num_workers=0)

    # define the loss function and the optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_accs = []
    val_accs = []
    test_accs = []
    train_losses = []
    val_losses = []
    test_losses = []

    # save the magnitude of the gradients for each layer
    grad_magnitudes = {i: [] for i in grant_layers}

    for ep in range(epochs):
        model.train()
        pred_correct = 0
        ep_loss = 0.
        # save the magnitude of the gradients for each epoch
        grad_magnitudes_epoch = {i: [] for i in grant_layers}

        # perform a training iteration
        for i, (inputs, labels) in enumerate(tqdm(trainloader)):
        
            # move the inputs and labels to the device
            inputs, labels = inputs.to(device , dtype=torch.float), labels.to(device)

            # zero the gradients
            optimizer.zero_grad()

            # forward pass
            outputs = model(inputs)

            # calculate the loss
            if batch_size > 16:
                loss = criterion(outputs.squeeze(), labels)
            else:
                loss = criterion(outputs, labels)

            # backward pass
            loss.backward()
            
            # computing and collecting the gradient magnitude for specified layers 
            for name, param in model.named_parameters():
                # Extract the layer index from the parameter name
                layer_idx = int(name.split('.')[0])

                # check if the layer is in the grant_layers
                if layer_idx in grant_layers:

                    # compute the gradient magnitude - grad magnitude = ||grad||2. 
                    #grad = torch.sqrt(torch.norm(param.grad) ** 2)
                    grad = torch.norm(param.grad) ** 2
                    # save the magnitude of the gradient for the current epoch
                    grad_magnitudes_epoch[layer_idx].append(grad.item())

            
            # update the weights
            optimizer.step()

            # calculate the number of correct predictions
            pred_correct += (torch.argmax(outputs, dim=1) == labels).sum().item()

            # calculate the loss
            ep_loss += loss.item()

        # save the avrage magnitude of the gradients for each layer
        for layer in grant_layers:
            grad_magnitudes[layer].append(np.mean(grad_magnitudes_epoch[layer]))

        # save the training accuracy and loss
        train_accs.append(pred_correct / len(trainset))
        train_losses.append(ep_loss / len(trainloader))

        # evaluate the model on the validation and test sets
        model.eval()
        with torch.no_grad():
            for loader, accs, losses in zip([valloader, testloader], [val_accs, test_accs], [val_losses, test_losses]):
                correct = 0
                total = 0
                ep_loss = 0.

                # perform an evaluation iteration
                for inputs, labels in loader:

                    # move the inputs and labels to the device
                    inputs, labels = inputs.to(device , dtype=torch.float), labels.to(device)

                    # forward pass
                    outputs = model(inputs)

                    # calculate the loss
                    loss = criterion(outputs, labels)

                    # calculate the number of correct predictions
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                    # calculate the loss
                    ep_loss += loss.item()

                accs.append(correct / total)
                losses.append(ep_loss / len(loader))
          

    return model, train_accs, val_accs, test_accs, train_losses, val_losses, test_losses , grad_magnitudes

def Q_6_5(train_data):
    """
    Monitoring Gradients - Train another model with 100 hidden layers each with 4 neurons. 
    This time, keep the magnitude of the gradients for each layer, through the training steps.
    The magnitude of of the gradients is defined by: grad magnitude = ||grad||2. 
    Train this network for 10 epochs. 
    For the layers: 0,30,60,90,95,99 plot the gradients magnitude 
    through the training epochs (average the magnitudes of each layer in every epoch)
    param : train_data : pd.DataFrame
    """
    print('Q_6_5')
    torch.manual_seed(42)
    np.random.seed(42)

    output_dim = len(train_data['country'].unique())
    # initialize the model
    model = model_maker(100 , 4 , output_dim)
    model = nn.Sequential(*model)
    grant_layers = [0,30,60,90,95,99]
    model, train_accs, val_accs, test_accs, train_losses, val_losses, test_losses , grad_magnitudes = Q_6_5_train(train_data, val_data, test_data, model, grant_layers , lr=0.01, epochs=10, batch_size=256)

    # Plot the magnitude of the gradients for each layer
    epochs = list(range(1, 11))
    plt.figure()
    for layer in grant_layers:
        plt.plot(epochs , grad_magnitudes[layer], label=f'Layer {layer}' , color=COLORS[grant_layers.index(layer)])

    plt.title('Gradients Magnitude Average acroos epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Gradients Magnitude')
    plt.legend()
    plt.show()




if __name__ == "__main__":
    # seed for reproducibility


    train_data = pd.read_csv('train.csv')
    val_data = pd.read_csv('validation.csv')
    test_data = pd.read_csv('test.csv')
    
    loss_lst_rates = Q_6_1_2(train_data)
    Q_6_1_3(train_data)

    Q_6_1_4(train_data )
    X_test, Y_test = read_data("test.csv")

    # Load the test files
    X_test, Y_test = read_data("test.csv")


    # Load the validation files
    X_validation, Y_validation = read_data("validation.csv")


    Q_6_1_5(train_data , X_test)

    models = Q_6_2(train_data , X_test , Y_test)
    
    Q_6_3(models)
    
    Q_6_4(models)

    Q_6_5(train_data)