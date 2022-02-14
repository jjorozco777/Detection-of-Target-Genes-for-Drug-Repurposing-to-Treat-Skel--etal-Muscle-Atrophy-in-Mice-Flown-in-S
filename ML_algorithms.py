import random
from tqdm import tqdm
import networkx as nx
from zipfile import ZipFile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)
import numpy as np
import random
import networkx as nx
from IPython.display import Image
from random import sample
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
from node2vec import Node2Vec
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score, auc, roc_curve, roc_auc_score,confusion_matrix
import pandas as pd 
import sys
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
import seaborn as sns
import matplotlib.colors as colors
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import DistanceMetric
from sklearn.neighbors import KNeighborsClassifier
import scipy as sp
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from pandas import ExcelWriter 
from pandas import ExcelFile
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
from sklearn import metrics
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, MaxPooling1D
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_iris
from numpy import unique
from keras.layers import Dropout
from keras.utils import to_categorical

pd.set_option('display.max_rows', 100)

network_data = pd.read_excel('R-AUC.xlsx', sheet_name = ['Elements', 'Connections','gene-disease'])

elements_data = network_data['Elements']
connections_data = network_data['Connections']
gene_diasease=network_data['gene-disease']


G= nx.convert_matrix.from_pandas_edgelist(connections_data,
                                               source = 'Node_1',
                                               target = 'Node_2')

print(connections_data)
print(nx.info(G))

# get a list of nodes in our graph
l = list(G.nodes())

print(l)
 
# create adjacency matrix
adj_G = pd.crosstab(connections_data.Node_1, connections_data.Node_2)
#adj_G = nx.to_numpy_matrix(G, nodelist = l)
 
print(adj_G)

print(connections_data.Node_1[0])

print(adj_G.shape[1])

# get all node pairs which don't have an edge
non_existing_edges = []


# traverse adjacency matrix
offset = 0
for i in range(adj_G.shape[0]):
   for j in range(adj_G.shape[1]):
       if i != j:
           if adj_G.iloc[i,j] == 0:
                   non_existing_edges.extend([(gene_diasease.genes[i],gene_diasease.diseases[j])])

print(len(non_existing_edges))   



nodes_2000 = sorted(random.sample(non_existing_edges, k=2000))



non_existing_edges = [(i[0],i[1]) for i in tqdm(nodes_2000) if nx.has_path(G, i[0], i[1])]

print(non_existing_edges[:5])

#Creating Dataframe of Non_Existing_Edges

df1 = pd.DataFrame(data = non_existing_edges, columns =['Node_1', 'Node_2'])
 
# create a column 'Connection' with default 0 (no-connection)
df1['Connection'] = 0
 
print(df1.head())


net_temp = connections_data.copy()
 

removable_edges_indices = []
 

ncc = nx.number_connected_components(G)
number_of_nodes = len(G.nodes)
 

for i in tqdm(connections_data.index.values):
  
      # remove a node pair and build a new graph
   G1 = nx.from_pandas_edgelist(net_temp.drop(index= i), "Node_1", "Node_2",
                                create_using=nx.Graph())
   if (nx.number_connected_components(G1) == ncc) and (len(G1.nodes) == number_of_nodes):
       removable_edges_indices.append(i)
 
       
       net_temp = net_temp.drop(index = i)


print(removable_edges_indices[:5])


# get node pairs in connection_data dataframe with indices in removable_edges_indices
df2 = connections_data.loc[removable_edges_indices]
 
# create a column 'Connection' and assign default value of 1 (connected nodes)
df2['Connection'] = 1
 
print(df2.head())

#Creating Subgraph

df1 = df1.append(df2[['Node_1', 'Node_2', 'Connection']],
                ignore_index=True)


df1=df1.astype(int)
print(df1)
df1.to_csv('df1.csv', index=False)



df3 = connections_data.drop(index=df2.index.values)

G_new = nx.from_pandas_edgelist(df3, "Node_1", "Node_2",
                               create_using=nx.Graph())

print(nx.info(G_new))




# Generating walks
node2vec = Node2Vec(G_new, dimensions=100, walk_length=16, num_walks=50)
 
# training the node2vec model
n2v_model = node2vec.fit(window=7, min_count=1)



edge_features = [(n2v_model[str(i)]+n2v_model[str(j)])
for i,j in zip(df1['Node_1'], df1['Node_2'])]
####################################################################################################################################################################
#BULDING THE MODEL


X = np.array(edge_features)  
y = df1['Connection']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

print(X_train[3])


#################################################### RANDOM FOREST ###########################################################


clf1 = RandomForestClassifier()
 
# parameters
param = {'n_estimators' : [10,50,100], 'max_depth' : [5,10,15]}
 
# model
grid_clf_acc1 = GridSearchCV(clf1, param_grid = param)
 
# train the model
grid_clf_acc1.fit(X_train, y_train)
 
print('Grid best parameter RANDOM FOREST (max. accuracy): ', grid_clf_acc1.best_params_)
print('Grid best score RANDOM FOREST (accuracy): ', grid_clf_acc1.best_score_)
 
# alternative metric to optimize over grid parameters: AUC
grid_clf_auc1 = GridSearchCV(clf1, param_grid = param, scoring = 'roc_auc')
grid_clf_auc1.fit(X_train, y_train)
predict_proba = grid_clf_auc1.predict_proba(X_test)[:,1]
 
print('Test set AUC: ', roc_auc_score(y_test, predict_proba))
print('Grid best parameter RANDOM FOREST(max. AUC): ', grid_clf_auc1.best_params_)
print('Grid best score RANDOM FOREST (AUC): ', grid_clf_auc1.best_score_)


#################################################### GRADIENT BOOST ###########################################################


clf2 = GradientBoostingClassifier()
 
# parameters
param = {'learning_rate' : [.05,.1]}
 
# model
grid_clf_acc2 = GridSearchCV(clf2, param_grid = param)
 
# train the model
grid_clf_acc2.fit(X_train, y_train)
 
print('Grid best parameter G-BOOST (max. accuracy): ', grid_clf_acc2.best_params_)
print('Grid best score G-BOOST (accuracy): ', grid_clf_acc2.best_score_)
 
# alternative metric to optimize over grid parameters: AUC
grid_clf_auc2 = GridSearchCV(clf2, param_grid = param, scoring = 'roc_auc')
grid_clf_auc2.fit(X_train, y_train)
predict_proba = grid_clf_auc2.predict_proba(X_test)[:,1]
 
print('Test set AUC: ', roc_auc_score(y_test, predict_proba))
print('Grid best parameter G-BOOST (max. AUC): ', grid_clf_auc2.best_params_)
print('Grid best score (AUC) G-BOOST: ', grid_clf_auc2.best_score_)

#################################################### NEURAL NETWORK ###########################################################
from sklearn.preprocessing import MinMaxScaler
# classifier
clf3 = MLPClassifier(max_iter=1000)

# scaling training and test sets

scaler = MinMaxScaler()
X_scaled=scaler.fit_transform(X)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
 
# parameters
param = {'hidden_layer_sizes' : [10,10,[10,10]], 'activation' : ['tanh','relu'], 'solver' : ['adam','lbfgs']}
 
# model
grid_clf_acc3 = GridSearchCV(clf3, param_grid = param)
 
# train the model
grid_clf_acc3.fit(X_train_scaled, y_train)
 
print('Grid best parameter NN (max. accuracy): ', grid_clf_acc3.best_params_)
print('Grid best score NN(accuracy): ', grid_clf_acc3.best_score_)
 
# alternative metric to optimize over grid parameters: AUC
grid_clf_auc3 = GridSearchCV(clf3, param_grid = param, scoring = 'roc_auc')
grid_clf_auc3.fit(X_train_scaled, y_train)
predict_proba = grid_clf_auc3.predict_proba(X_test_scaled)[:,1]
 
print('Test set AUC: ', roc_auc_score(y_test, predict_proba))
print('Grid best parameter NN(max. AUC): ', grid_clf_auc3.best_params_)
print('Grid best score NN (AUC): ', grid_clf_auc3.best_score_)


 
false_positive_rate,true_positive_rate,_ = roc_curve(y_test, predict_proba)
roc_auc_score = auc(false_positive_rate,true_positive_rate)

plt.plot(false_positive_rate,true_positive_rate)
plt.title(f'ROC Curve \n ROC AUC Score : {roc_auc_score}')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

probabilities=[]



#print(f' ({df1.iloc[3,0]},{df1.iloc[3,1]}) node pair features : {X[3]}')

# its position in X_train
#print(f'Index of ({df1.iloc[3,0]},{df1.iloc[3,1]}) node pair in X_train : {np.where(X_train == X[3])[0][1]}')

#print(np.where(X_train == X[3]))

for i in range(0,2932):
  #print(f' ({df1.iloc[15,0]},{df1.iloc[15,1]}) node pair features : {X[15]}')

  # its position in X_train
  #print(f'Index of ({df1.iloc[15,0]},{df1.iloc[15,1]}) node pair in X_train : {np.where(X_train == X[15])[0][1]}')
  probabilities.extend([( df1.iloc[i,0], df1.iloc[i,1], grid_clf_auc3.predict_proba(X_scaled[np.where(X== X[i])[0][1]].reshape(1,-1))[:,1]*100)])


##########################################1ConvNN#############################

model = Sequential()
model.add(Conv1D(filters=100, kernel_size=3, activation='relu', input_shape=(100,1)))
model.add(Conv1D(filters=100, kernel_size=5, activation='relu'))
model.add(Dropout(0.5))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()


y_train = to_categorical(y_train, 3)
y_test = to_categorical(y_test, 3)

'''model = Sequential()
model.add(Conv1D(1000, 10, activation="relu", input_shape=(890,1)))
model.add(Dense(500, activation="relu"))
model.add(MaxPooling1D())
model.add(Flatten())
model.add(Dense(3, activation = 'softmax'))
model.compile(loss = 'sparse_categorical_crossentropy', 
     optimizer = "adam",               
              metrics = ['accuracy'])
model.summary()'''

model.fit(X_train_scaled, y_train, batch_size=16,epochs=100, verbose=0)

acc = model.evaluate(X_test_scaled, y_test)
print("Loss:", acc[0], " Accuracy:", acc[1])

accuracy = model.evaluate(X_test_scaled, y_test, batch_size=16, verbose=0)

print('accuracy Conv1D',accuracy)
