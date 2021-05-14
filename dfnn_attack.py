# default_example.py
import argparse
import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
import tensorflow as tf
from tensorflow import keras
import kerastuner as kt
from kerastuner.tuners import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters

# Create the parser
my_parser = argparse.ArgumentParser(description='Deep learning based model building attack tool on PUF')

# Add the arguments
my_parser.add_argument('-c','--challenge',
                       metavar='path',
                       type=str,
                       help='the path to challenges CSV file')

my_parser.add_argument('-r','--response',
                       metavar='path',
                       type=str,
                       help='the path to response CSV file')
my_parser.add_argument('-f', '--features', type=str,
                       help='To convert the challenges to feature vectors')

my_parser.add_argument('-l', '--level', type=int,
                       help='level specifies the first-level test (0.5 million CRPs) or second level test ( <= 1.5 million)')


my_parser.add_argument('-v', '--verbose', type=int,
                       help='verbose=10 enables the epoch prints and verbose=0 disables it')


args = my_parser.parse_args()
print(args)
challenge_path = args.challenge
response_path = args.response
features = args.features
level = args.level
verbose = args.verbose
#challenge_path = 'XOR_APUF_Binary_chal_64_500000.csv'
#response_path = '4-xorpuf.csv'



def keras_puf_model_builder(hp):
  model = keras.Sequential()
 
  # Choose an optimal value between 10-300
  hp_units = hp.Int('units', min_value=10, max_value=300, step=30, default =10)
  model.add(keras.layers.Dense(units=hp_units,  input_shape=x_train.shape[1:], activation=hp.Choice(
                    'dense_activation',
                    values=['relu', 'tanh'],
                    default='relu'
                )))
  model.add(keras.layers.Dense(units=hp_units, activation=hp.Choice(
                    'dense_activation',
                    values=['relu', 'tanh'],
                    default='relu'
                )))
  model.add(keras.layers.Dense(units=hp_units,  activation=hp.Choice(
                    'dense_activation',
                    values=['relu', 'tanh'],
                    default='relu'
                )))
                
  model.add(keras.layers.Dense(units=hp_units,   activation=hp.Choice(
                    'dense_activation',
                    values=['relu', 'tanh'],
                    default='relu'
                )))
  model.add(keras.layers.Dense(units=hp_units,  activation=hp.Choice(
                    'dense_activation',
                    values=['relu', 'tanh'],
                    default='relu'
                )))
  model.add(keras.layers.Dense(1, activation='sigmoid'))
   
  model.compile(
            optimizer=keras.optimizers.Adam(
                hp.Float(
                    'learning_rate',
                    min_value=1e-4,
                    max_value=1e-2,
                    default=1e-3
                )
            ),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
  
  return model 

def fetch_CRP():

    if not os.path.exists(challenge_path):
        print('The path specified for challenge CSV file does not exist')
        sys.exit()
     
    if not os.path.exists(response_path):
       print('The path specified for response CSV file does not exist')
       sys.exit()
    
    npChallenge = pd.read_csv(challenge_path,header=None).values
    npResponse = pd.read_csv(response_path,header=None).values

    return npChallenge, npResponse
    
def get_parity_vectors(C):
  n=C.shape[1]
  m=C.shape[0]
  C[C==0]=-1
  parityVec=np.zeros((m,n+1))
  parityVec[:,0:1]=np.ones((m,1))
  for i in range(2,n+2):
      parityVec[:,i-1:i]=np.prod(C[:,0:i-1],axis=1).reshape((m,1))
  return parityVec

def transform_challenge(npC):

    if (features == 'parity'):
        fVec = get_parity_vectors(npC)
    elif (features == 'custom'):
        fVec = custom_transformation(npC)
    else :
         fVec = npC
         
    return fVec

     
    if not os.path.exists(response_path):
       print('The path specified for response CSV file does not exist')
       sys.exit()
    
    npChallenge = pd.read_csv(challenge_path,header=None).values.astype(np.int32)
    npResponse = pd.read_csv(response_path,header=None).values.astype(np.int32)

    return npChallenge, npResponse

def linear_attack(train_features, test_features, train_labels, test_labels,n_samples):
    # cut down the # of challenges to 50,000 for traditional ml algo such as svm and lr
    tr_f = train_features[:n_samples]
    tr_l = train_labels[:n_samples]
    te_f = test_features[n_samples:n_samples + int(1.2 * n_samples)]
    te_l = test_labels[n_samples:n_samples + int(1.2 * n_samples)]
    print("1. Linearly Separable Test")
    print('   Trying to attack with %d samples' %n_samples)
    print("    1.a SVM Classifier")
    lin_svc = svm.LinearSVC(C=1.0).fit(tr_f, tr_l)
    y_pred = lin_svc.predict(te_f)
    acc_svm = accuracy_score(te_l, y_pred)
    print('        Linear SVM Accuracy: %f\n' % accuracy_score(te_l, y_pred))
    print("    1.b Logistic Regression Classifier")
    lin_lr = LogisticRegression(random_state=0).fit(tr_f, tr_l)
    y_pred = lin_lr.predict(te_f)
    acc_lr = accuracy_score(te_l, y_pred)
    print('        Logistic Regression Accuracy: %f\n' % accuracy_score(te_l, y_pred))
    if ((acc_svm > 0.8) or (acc_lr > 0.8)):
        print('The challenges and responses are linearly separable with accuracy %f',  np.max(acc_svm,acc_lr)*100)
    elif (((acc_svm < 0.8) and (acc_svm > 0.6))  or ((acc_lr < 0.8) and (acc_lr > 0.6))):
        print("Not satisfactory to conculde linearly separabality. Maximum achieved accuracy is ", np.max(acc_svm,acc_lr))
    else :
        print(" The challenges and responses are NOT linearly separable")


def nonlinear_attack(train_features, test_features, train_labels, test_labels,n_samples):
# cut down the challenges to 50,000 for svm and lr
    tr_f = train_features[:n_samples]
    tr_l = train_labels[:n_samples] 
    te_f = test_features[n_samples:n_samples + int(1.2 * n_samples)]
    te_l = test_labels[n_samples:n_samples + int(1.2 * n_samples)]
    print("2. Non-linear Attack Test")
    print('   Trying to attack with %d samples' %n_samples)
    print("    2.a SVM with RBF Kernel Classifier")
    rbf_svc = svm.SVC(kernel='rbf').fit(tr_f, tr_l)
    y_pred = rbf_svc.predict(te_f)
    acc_svm = accuracy_score(te_l, y_pred)
    print('        SVM-RBF Accuracy: %f\n' % accuracy_score(te_l, y_pred))
    print("    2.b Multi Layer Perceptron Classifier")
    print("    \n Performing hyperparameter search in three steps (can take approx 7 to 10 hours)")
    #print("Please be patient\n. The program will perform an exhaustive grid search to choose best hyperparameters which takes some time (max upto ten hours)")
    gs_mlp = MLPClassifier(max_iter=100,verbose=verbose,early_stopping=True)
    parameter_space1 = {
    'hidden_layer_sizes': [(10),(20)#,(30),(50),(80),(100)
                           ],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    #'alpha': [0.0001, 0.01],
    #'learning_rate': ['constant','adaptive']
    }
    
    GS_CV = GridSearchCV(gs_mlp, parameter_space1, cv=5, scoring='accuracy', verbose=1)  
    GS_CV.fit(train_features[:100000], train_labels[:100000]) 
    
    y_pred = GS_CV.predict(test_features)
    acc_mlp1 = accuracy_score(test_labels, y_pred)
    print('    Till now achieved (1/3) \n    the best hyperparameters', GS_CV.best_params_)
    print('    (1/3)MLP Accuracy: %f\n' % accuracy_score(test_labels, y_pred))
    
    
    parameter_space2 = {
    'hidden_layer_sizes': [
                           (10,10),(20,20)#,(30,30),(50,50),(80,80),(100,100),
                           
                           ],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    #'alpha': [0.0001, 0.01],
    #'learning_rate': ['constant','adaptive']
    }
    
    GS_CV = GridSearchCV(gs_mlp, parameter_space1, cv=5, scoring='accuracy', verbose=1)  
    GS_CV.fit(train_features[:250000], train_labels[:250000]) 
    
    y_pred = GS_CV.predict(test_features)
    acc_mlp2 = accuracy_score(test_labels, y_pred)
    print('    Till now achieved (2/3) \n    the best hyperparameters', GS_CV.best_params_)
    print('    (2/3)MLP Accuracy: %f\n' % accuracy_score(test_labels, y_pred))
        
    parameter_space3 = {
    'hidden_layer_sizes': [(10,10,10),(20,20,20)#,(30,30,30),(50,50,50),(80,80,80),(100,100,100)
                           ],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    #'alpha': [0.0001, 0.01],
    #'learning_rate': ['constant','adaptive']
    }
    
    
    GS_CV = GridSearchCV(gs_mlp, parameter_space1, cv=5, scoring='accuracy', verbose=1)  
    GS_CV.fit(train_features, train_labels) 
    
    y_pred = GS_CV.predict(test_features)
    acc_mlp3 = accuracy_score(test_labels, y_pred)
    print('    Till now achieved (3/3) \n    the best hyperparameters', GS_CV.best_params_)
    print('    (3/3)MLP Accuracy: %f\n' % accuracy_score(test_labels, y_pred))
    
    print('    Best MLP Accuracy: %f\n' % np.max(acc_mlp3,np.max(acc_mlp2,acc_mlp1)))
            
def perform_first_level_test(train_challenges, test_challenges, train_responses, test_responses):
    
    #if user didnt specify the features, try with all known
    if( features == 'direct'):
        #try  with first direct challenges 
        print("Trying to attack using raw challenges")
        linear_attack(train_challenges, test_challenges, train_responses, test_responses,50000)
        nonlinear_attack(train_challenges, test_challenges, train_responses, test_responses,50000)
        
    
    if( features == 'parity'):
        print("Trying to attack using parity vector transformed challenges")
        train_features = get_parity_vectors(train_challenges)
        test_features = get_parity_vectors(test_challenges)
        linear_attack(train_features, test_features, train_responses, test_responses,50000)
        nonlinear_attack(train_features, test_features, train_responses, test_responses,50000)
    # add further if any other feature transformation extists for PUF modeling attacks
    
    if(features == 'custom'):
        print("Trying to attack using parity vector transformed challenges")
        #define the function 'get_custom_transformation' before running this part of code
        print(" Please define the function 'get_custom_transformation' before running this part of code")
        #train_features = get_custom_transformation(train_challenges)
        #test_features = get_custom_transformation(test_challenges)
        #linear_attack(train_features, test_features, train_labels, test_labels,50000)
   

def keras_attack(train_challenges, test_challenges, train_responses, test_responses):
    
    # Hyperparameter optimization by Random Search
    # comment the below code if you opt for other search methods
    tuner = RandomSearch(
    keras_puf_model_builder,
    objective='val_accuracy',
    max_trials=1,  # how many model variations to test?
    executions_per_trial=1)  # how many trials per variation? (same model could perform differently)
    
    # comment the below code if you opt for other search methods
    tuner = kt.Hyperband(
    keras_puf_model_builder, objective="val_accuracy", max_epochs=30, hyperband_iterations=2)
    
    

    tuner.search(x=train_challenges,
             y=train_responses,
             verbose=2, 
             epochs=100,
             batch_size=1000,
             validation_data=(x_test, y_test),
             callbacks=[tf.keras.callbacks.EarlyStopping(patience=3)])
             
             
    
    best_model = tuner.get_best_models(1)[0]
    best_hyperparameters = tuner.get_best_hyperparameters(1)[0]
    




def perform_second_level_test(train_challenges, test_challenges, train_responses, test_responses):
    
    #if user didnt specify the features, try with all known
    if( features == 'direct'):
        #try  with first direct challenges 
        print("Trying to attack using raw challenges")
        keras_attack(train_challenges, test_challenges, train_responses, test_responses)
        
        
    
    if( features == 'parity'):
        print("Trying to attack using parity vector transformed challenges")
        train_features = get_parity_vectors(train_challenges)
        test_features = get_parity_vectors(test_challenges)
        keras_attack(train_challenges, test_challenges, train_responses, test_responses)
        
    # add further if any other feature transformation extists for PUF modeling attacks
    
    if(features == 'custom'):
        print("Trying to attack using parity vector transformed challenges")
        #define the function 'get_custom_transformation' before running this part of code
        print(" Please define the function 'get_custom_transformation' before running this part of code")
        #train_features = get_custom_transformation(train_challenges)
        #test_features = get_custom_transformation(test_challenges)
        #keras_attack(train_challenges, test_challenges, train_responses, test_responses)
   
        
             
        
        
def main():
    # Fetch the CSV files and decribe the CRP
    npC, npR = fetch_CRP()
    
    # Shuffle the CRPs and divide them into training set and testing set in the ratio of 80:20

    train_challenges, test_challenges, train_responses, test_responses = train_test_split(npC, npR, test_size = 0.2, random_state = 42)
    #Perform first level modeling attack test
    if(level == 1):
        perform_first_level_test(train_challenges, test_challenges, train_responses, test_responses)
    
    if(level == 2):
        perform_second_level_test(train_challenges, test_challenges, train_responses, test_responses)


    
    
    
if __name__ == "__main__":
    main()
