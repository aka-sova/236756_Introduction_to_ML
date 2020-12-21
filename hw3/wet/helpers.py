import pandas as pd
import numpy as np 


def val_size( e, d, H ):
    """
    Using Hoeffding Bound + Union Bound:

    size of validation set for chosen e>0, d>0 
    is ( log(2|H|) + log(1/d) )/ 2e^2 
    that has generalization error at most optimal from H plus 2e
    with probability of 1-d 
    """

    size = np.divide( np.log( 2*H ) + np.log(np.divide(1,d)), 2*np.square(e) )

    return size 


def test_size( e, d, H ):
    """
    Using Hoeffding Bound + Union Bound:
    
    size of test set for chosen e>0, d>0 
    is ( O(1) + log(1/d) )/ 2e^2 
    that has generalization error at most optimal from H plus 2e
    with probability of 1-d 
    """

    size = np.divide( np.log(np.divide(1,d)), 2*np.square(e) )

    return size  


