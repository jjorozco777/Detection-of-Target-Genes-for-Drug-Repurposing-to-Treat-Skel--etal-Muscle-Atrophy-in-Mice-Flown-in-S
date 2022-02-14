# Pearson Correlation Code 

import argparse
import csv
import numpy
import scipy.stats as stats
import pandas as pd
from pandas import DataFrame as df 
import ast
from scipy.stats import pearsonr
import math


#Pearson correlation 


def p_value(x, y):
   
    t_stat = x*numpy.sqrt((y-2)/(1-x*x))
    p_val = stats.t.sf(numpy.abs(t_stat), y-2)*2
    return t_stat, p_val

def pearson_corr(X, Y):


    geneX = numpy.array(X).astype(numpy.float)
    #print(numA)
    geneY = numpy.array(Y).astype(numpy.float)#dtype=numpy.float32
    meanX = geneX.mean()
    print(meanX)
    meanY = geneY.mean()
    print(meanY)
    stdX = geneX.std()
    #print(stdA)
    stdY = geneY.std()
    X=list(geneX - geneX)
    Y=list(geneY - geneY)
    print(X,Y)
    z = sum(numpy.multiply(X,Y))
    print(z)
    XX=sum(numpy.multiply(X,X))
    #print('sum:',AA)
    YY=sum(numpy.multiply(Y,Y))
    #print('sum:',BB)
    corr=z/math.sqrt(XX*YY)

    return corr

'''
def get_pval(r, n):
   
    tstat = r*numpy.sqrt((n-2)/(1-r*r))
    pval = stats.t.sf(numpy.abs(tstat), n-2)*2
    return tstat, pval'''

def correlation(inputfile, signifp, outputfile):
    genes = []
    experiments = []

    with open(inputfile, 'r') as f:
        next(f)
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            genes.append(row[0])
            experiments.append(row[2:])
    numgenes = len(genes)
    numexperiments = len(experiments[0])

    
    alpha_i = signifp / (0.5*(numgenes-1)*numgenes)
   
        

    with open(outputfile, 'w') as out:
                    headings = ['InteractorA', 'InteractorB', 'Correlation', 't-statistic', 'p-value']
                    writer = csv.writer(out, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    writer.writerow(headings)
            
                    for i, gene_i in enumerate(genes):
            
                        
                        for j in range(i+1, numgenes):
            
            
            
                        
                            pearsons_r = pearson_corr(experiments[i], experiments[j])
                           
                            t_stat, p_val = p_value(pearsons_r, numexperiments)
            
                            
            
                            print(experiments[i], experiments[j])    


                                                
                    
            
                            writer = csv.writer(out, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                            writer.writerow([gene_i, genes[j], pearsons_r, t_stat, p_val])
    





correlation('prove.txt', 0.05, 'output33.csv')
