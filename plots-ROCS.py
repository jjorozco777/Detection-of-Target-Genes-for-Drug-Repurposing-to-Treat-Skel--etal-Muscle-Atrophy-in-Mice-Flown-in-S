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
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

pd.set_option('display.max_rows', 100)

network_data = pd.read_excel('disease-drugs-nn.xlsx', sheet_name = ['Elements', 'Connections','drug-disease'])

elements_data = network_data['Elements']
connections_data = network_data['Connections']
drug_disease=network_data['drug-disease']


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
                   non_existing_edges.extend([(drug_disease.disease[i],drug_disease.drug[j])])

print(len(non_existing_edges))   

#We will randomly select 2000 non-existing edges from the vast collection of 20988

nodes_2000 = sorted(random.sample(non_existing_edges, k=2000))

'''If we can reach a node through other nodes, then we consider that these two nodes are connected by a 
well-defined path. The nodes for which a path does not exist carry a little likelihood of a connection
in near future. We can safely eliminate such nodes in our further analysis. To sort out nodes with a reachable 
connection path, we use the following loop:'''

non_existing_edges = [(i[0],i[1]) for i in tqdm(nodes_2000) if nx.has_path(G, i[0], i[1])]

print(non_existing_edges[:5])

#Creating Dataframe of Non_Existing_Edges

df1 = pd.DataFrame(data = non_existing_edges, columns =['Node_1', 'Node_2'])
 
# create a column 'Connection' with default 0 (no-connection)
df1['Connection'] = 0
 
print(df1.head())

# Create a list of all indices of the node pairs in the dataframe,
# which when removed won’t change the structure of our graph
 
# create a copy
net_temp = connections_data.copy()
 
# for storing removable edges
removable_edges_indices = []
 
# number of connected components and nodes of G
ncc = nx.number_connected_components(G)
number_of_nodes = len(G.nodes)
 
# for each node pair we will be removing a node pair and creating a new graph,
# and check if the number of connected components and the number of nodes
# are the same as the original graph
for i in tqdm(connections_data.index.values):
  
      # remove a node pair and build a new graph
   G1 = nx.from_pandas_edgelist(net_temp.drop(index= i), "Node_1", "Node_2",
                                create_using=nx.Graph())
  
      # If the number of connected components remain same as the original
      # graph we won't remove the edge
   if (nx.number_connected_components(G1) == ncc) and (len(G1.nodes) == number_of_nodes):
       removable_edges_indices.append(i)
 
       # drop the edge, so that for the next iteration the next G1
       # is created without this edge
       net_temp = net_temp.drop(index = i)


print(removable_edges_indices[:5])

'''Next, we will create a data frame df2 which contains all the removable edges. As earlier with df1, 
we create a Connection column and set its value to 1, which means the node pair has an edge between them.'''



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
#df1.to_csv('df1.csv', index=False)

'''
Now all we have to do is to get the features of the nodes from graph ‘G_new’, 
and predict the possibility of the nodes forming an edge in the future, i.e. the graph ‘G’, 
using the connection column of df1 as our target feature.
In the next cell, we have created a new data frame df3 which contains the remaining edges,
 after removing the ‘removable edges’ from the fb data frame.
'''
df3 = connections_data.drop(index=df2.index.values)

G_new = nx.from_pandas_edgelist(df3, "Node_1", "Node_2",
                               create_using=nx.Graph())

print(nx.info(G_new))

'''Model Building
To apply a machine learning algorithm, you first need to identify the features and the target 
in your dataset. In our case, the dataset is the graph you have generated in the previous step.
 We will use node2vec for extracting features.'''


# Generating walks
node2vec = Node2Vec(G_new, dimensions=100, walk_length=16, num_walks=50)
 
# training the node2vec model
n2v_model = node2vec.fit(window=7, min_count=1)

'''Generating Features of the Edges
We apply the trained node2vec model from the previous step on every node of a node pair in the df1 data frame.
We add the results and features at each edge to generate the features of an edge/node pair. We store the result
in a list called edge_features.'''

edge_features = [(n2v_model[str(i)]+n2v_model[str(j)])
for i,j in zip(df1['Node_1'], df1['Node_2'])]
####################################################################################################################################################################
#BULDING THE MODEL


X = np.array(edge_features)  
y = df1['Connection']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

print(X_train[3])
'''We will use GridSearchCV on three different algorithms:
Random Forest
Gradient Boost
MLPClassifier'''

################################################### RANDOM FOREST ###########################################################
cv_outer = KFold(n_splits=10, shuffle=True, random_state=1)
#classifier
clf1 = RandomForestClassifier()
 
# parameters
param = {'n_estimators' : [10,100,500], 'max_depth' : [5,10,15]}
 
# model
grid_clf_acc1 = GridSearchCV(clf1, param_grid = param)
 
# train the model
grid_clf_acc1.fit(X_train, y_train)
 
print('Grid best parameter RANDOM FOREST (max. accuracy): ', grid_clf_acc1.best_params_)
print('Grid best score RANDOM FOREST (accuracy): ', grid_clf_acc1.best_score_)
 
# alternative metric to optimize over grid parameters: AUC
grid_clf_auc1 = GridSearchCV(clf1, param_grid = param, scoring = 'roc_auc',cv=cv_outer, n_jobs=-1)
grid_clf_auc1.fit(X_train, y_train)
scores = cross_val_score(clf1, X_train, y_train, scoring='roc_auc', cv=cv_outer, n_jobs=-1) 

print('scores random forest',scores)
predict_proba1 = grid_clf_auc1.predict_proba(X_test)[:,1]
 
print('Test set AUC: ', roc_auc_score(y_test, predict_proba1))
print('Grid best parameter RANDOM FOREST(max. AUC): ', grid_clf_auc1.best_params_)
print('Grid best score RANDOM FOREST (AUC): ', grid_clf_auc1.best_score_)


#################################################### GRADIENT BOOST ###########################################################

# classifier
clf2 = GradientBoostingClassifier()
 
# parameters
param = {'learning_rate' : [.01,.2]}
 
# model
grid_clf_acc2 = GridSearchCV(clf2, param_grid = param)
 
# train the model
grid_clf_acc2.fit(X_train, y_train)
 
print('Grid best parameter G-BOOST (max. accuracy): ', grid_clf_acc2.best_params_)
print('Grid best score G-BOOST (accuracy): ', grid_clf_acc2.best_score_)
 
# alternative metric to optimize over grid parameters: AUC
grid_clf_auc2 = GridSearchCV(clf2, param_grid = param, scoring = 'roc_auc',cv=cv_outer, n_jobs=-1)
grid_clf_auc2.fit(X_train, y_train)
scores = cross_val_score(clf2, X_train, y_train, scoring='roc_auc', cv=cv_outer, n_jobs=-1) 

print('scores random forest',scores)
predict_proba2 = grid_clf_auc2.predict_proba(X_test)[:,1]
 
print('Test set AUC: ', roc_auc_score(y_test, predict_proba2))
print('Grid best parameter G-BOOST (max. AUC): ', grid_clf_auc2.best_params_)
print('Grid best score (AUC) G-BOOST: ', grid_clf_auc2.best_score_)

#################################################### NEURAL NETWORK ###########################################################

# classifier
clf3 = MLPClassifier(max_iter=1000)
 
# scaling training and test sets
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
 
# parameters
param = {'hidden_layer_sizes' : [10,100,[10,10]], 'activation' : ['tanh','relu'], 'solver' : ['adam','lbfgs']}
 
# model
grid_clf_acc3 = GridSearchCV(clf3, param_grid = param)
 
# train the model
grid_clf_acc3.fit(X_train_scaled, y_train)
 
print('Grid best parameter NN (max. accuracy): ', grid_clf_acc3.best_params_)
print('Grid best score NN(accuracy): ', grid_clf_acc3.best_score_)
 
# alternative metric to optimize over grid parameters: AUC
grid_clf_auc3 = GridSearchCV(clf3, param_grid = param, scoring = 'roc_auc',cv=cv_outer, n_jobs=-1)
grid_clf_auc3.fit(X_train_scaled, y_train)
scores = cross_val_score(clf3, X_train_scaled, y_train, scoring='roc_auc', cv=cv_outer, n_jobs=-1) 

print('scores random forest',scores)
predict_proba3 = grid_clf_auc3.predict_proba(X_test_scaled)[:,1]
 
print('Test set AUC: ', roc_auc_score(y_test, predict_proba3))
print('Grid best parameter NN(max. AUC): ', grid_clf_auc3.best_params_)
print('Grid best score NN (AUC): ', grid_clf_auc3.best_score_)





false_positive_rate1,true_positive_rate1,_ = roc_curve(y_test, predict_proba1)
roc_auc_score = auc(false_positive_rate1,true_positive_rate1)

false_positive_rate2,true_positive_rate2,_ = roc_curve(y_test, predict_proba2)
roc_auc_score = auc(false_positive_rate2,true_positive_rate2) 

false_positive_rate3,true_positive_rate3,_ = roc_curve(y_test, predict_proba3)
roc_auc_score = auc(false_positive_rate3,true_positive_rate3)



#################################################################################################################################################################

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


network_data = pd.read_excel('disease-drugs-nn.xlsx', sheet_name = ['Elements', 'Connections'])

elements_data = network_data['Elements']
connections_data = network_data['Connections']
print (connections_data.head(10))

graph = nx.convert_matrix.from_pandas_edgelist(connections_data,
                                               source = 'Node_1',
                                               target = 'Node_2')

#print(graph.edges)

pos = nx.spring_layout(graph)

print(nx.info(graph))

#nx.draw(graph, cmap = plt.get_cmap('rainbow'), with_labels=True, pos=pos)

#information of the graph 

n = graph.number_of_nodes()
m = graph.number_of_edges()

'''The Idea is hide a subset of node pairs, and predict their links based on the
rules defined above. We then evaluate the proportion of correct predictions 
for dense graphs, or use Area under the Curve criteria for Sparse graphs.'''

# Remove 20% of the edges
proportion_edges = 0.1
edge_subset = random.sample(graph.edges(), int(proportion_edges * graph.number_of_edges()))

# Create a copy of the graph and remove the edges
G__train = graph.copy()
G__train.remove_edges_from(edge_subset)


#plt.figure(figsize=(12,8))
#nx.draw(G__train)
#plt.gca().collections[0].set_edgecolor("#000000") # set node border color to black

edge_subset_size = len(list(edge_subset))
print("Number of edges deleted : %d" % edge_subset_size)
print("Number of edges remaining : %d" % (m - edge_subset_size))

# Make prediction using Jaccard Coefficient
pred_jaccard = list(nx.jaccard_coefficient(G__train))
score_jaccard, label_jaccard = zip(*[(s, (u,v) in edge_subset) for (u,v,s) in pred_jaccard])


#print(pred_jaccard)

# Compute the ROC AUC Score
fpr_jaccard, tpr_jaccard, _ = roc_curve(label_jaccard, score_jaccard)
auc_jaccard = roc_auc_score(label_jaccard, score_jaccard)

print('jaccard score',auc_jaccard)

# Prediction using Adamic Adar 
pred_adamic = list(nx.adamic_adar_index(G__train))
score_adamic, label_adamic = zip(*[(s, (u,v) in edge_subset) for (u,v,s) in pred_adamic])

# Compute the ROC AUC Score
fpr_adamic, tpr_adamic, _ = roc_curve(label_adamic, score_adamic)
auc_adamic = roc_auc_score(label_adamic, score_adamic)

print('adamic score',auc_adamic)

# Compute the Preferential Attachment
pred_pref = list(nx.preferential_attachment(G__train))
score_pref, label_pref = zip(*[(s, (u,v) in edge_subset) for (u,v,s) in pred_pref])

fpr_pref, tpr_pref, _ = roc_curve(label_pref, score_pref)
auc_pref = roc_auc_score(label_pref, score_pref)

print('preference score',auc_pref)
# compute resource allocation 
pred_reso = list(nx.nx.resource_allocation_index(G__train))
score_reso, label_reso = zip(*[(s, (u,v) in edge_subset) for (u,v,s) in pred_reso])

fpr_reso, tpr_reso, _ = roc_curve(label_reso, score_reso)
auc_reso = roc_auc_score(label_reso, score_reso)

print('resource score',auc_reso)

#pyplot.plot(fpr_jaccard, tpr_jaccard,  marker='.',label=('Jaccard_AUC=',auc_jaccard))
#pyplot.plot(fpr_adamic, tpr_adamic,  marker='.',label=('Adamic_AUC=',auc_adamic))
pyplot.plot(fpr_pref, tpr_pref,  marker='.',label=('Preferential Attachment '))
#pyplot.plot(fpr_reso, tpr_reso,  marker='.',label=('Resource_AUC=',auc_reso))





pyplot.plot(false_positive_rate1,true_positive_rate1,  marker='.',label=('Random Forest '))
pyplot.plot(false_positive_rate2,true_positive_rate2,  marker='.',label=('Gradient Boost'))
pyplot.plot(false_positive_rate3,true_positive_rate3, marker='.',label=('GNN'))



#plt.title('ROC AUC CURVES')
pyplot.xlabel('False positive rate')
pyplot.ylabel('True positive rate')
pyplot.legend()


'''probabilities=[]
#print(f' ({df1.iloc[3,0]},{df1.iloc[3,1]}) node pair features : {X[3]}')
# its position in X_train
#print(f'Index of ({df1.iloc[3,0]},{df1.iloc[3,1]}) node pair in X_train : {np.where(X_train == X[3])[0][1]}')
#print(np.where(X_train == X[3]))
for i in range(0,2932):
  #print(f' ({df1.iloc[15,0]},{df1.iloc[15,1]}) node pair features : {X[15]}')
  # its position in X_train
  #print(f'Index of ({df1.iloc[15,0]},{df1.iloc[15,1]}) node pair in X_train : {np.where(X_train == X[15])[0][1]}')
  probabilities.extend([( df1.iloc[i,0], df1.iloc[i,1], grid_clf_auc1.predict_proba(X[np.where(X== X[i])[0][1]].reshape(1,-1))[:,1]*100)])
 
 
print(pd.DataFrame(probabilities))
 
print(non_existing_edges) '''

plt.show()
