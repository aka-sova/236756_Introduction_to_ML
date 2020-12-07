
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
import numpy as np



def SeqBackSel(X_train, X_test, y_train, y_test):
    
    full = [ item+1 for item in range(16)] #backward propegation

    for m in range(6): #picked some number of removals of features
        scores= []
        for item in full:
            kn = KNeighborsClassifier( n_neighbors = 5, weights= 'uniform' ) # weights= 'distance'
            removed_cols = [ 'pcrResult{}'.format(itr+1) for itr in range(16) if itr+1 not in full]
            removed_df = X_train.drop( columns= ['pcrResult{}'.format(item)] + removed_cols ) 
            kn.fit( removed_df, y_train )
            vals = kn.predict(X_test.drop( columns=['pcrResult{}'.format(item)] + removed_cols ))
            this_score = f1_score(y_test, vals, average='weighted' ) #f_1 score
            scores.append( ( item ,this_score) )
        arr = [ scores[k][1] for k in range(len(scores))]
        amax = np.amax(np.array(arr))
        amax_idx = [ scores[l] for l in range(len(scores)) if scores[l][1]  == amax ][0][0] #indx when dropped scoring is maximal
        full = [ item for item in full if item != amax_idx]

return full
