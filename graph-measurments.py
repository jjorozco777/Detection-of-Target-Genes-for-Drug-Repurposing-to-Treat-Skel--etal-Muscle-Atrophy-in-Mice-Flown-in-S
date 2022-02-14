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

A = nx.adjacency_matrix(G)



Adjacency=pd.DataFrame(A.todense())

writer = ExcelWriter('adjacency_Col-0_sl-pool_FLT_cellular-respose-to-stimulus.xlsx')
Adjacency.to_excel(writer,'Sheet1',index=False)
writer.save()

print(pd.DataFrame(A.todense()))

#compute the spectrum S= spectrum(A) which is the Eigen values of the graph
Spectrum=np.real(nx.adjacency_spectrum(G, weight='weight'))

spec_sorted=list(Spectrum)


spec_sorted.sort()

print("spectrum:",spec_sorted)

#spectral gap gamma(S) = first highest eigenvalue - second highest eigenvalue
max_spect=spec_sorted[len(spec_sorted)-1]

second_max_spect=spec_sorted[len(spec_sorted)-2]

spectral_gap=max_spect - second_max_spect

print("spectral gap:",spectral_gap)

#local clustering coefficient Ci, average clustering coefficient for the graph G.

clustering=nx.average_clustering(G)
local_clustering=nx.clustering(G)
#print('local clustering:',local_clustering)
#clust=pd.DataFrame(clustering)
print('local clustering:',local_clustering)
#writer = ExcelWriter('clustering_output.xlsx')
#clustering.to_excel(writer,'Sheet1',index=False)
#writer.save()
print('clustering coefficient:',clustering)

#number of cliques (fully connected subgraphs in the network)

#plt.show()

print('density of the graph:',nx.density(G))
#print('diameter:',nx.diameter(G))

#ASPL=nx.average_shortest_path_length(G)
#print("average_shortest_path_length",ASPL)

S_centrality=nx.subgraph_centrality(G)
print('subgraph_centrality',S_centrality)


#PWLD=random_powerlaw_tree(len(G), gamma=3, seed=None, tries=100)
#print('power-law distribution:',PWLD)
