import numpy as np
from itertools import ifilterfalse
import string
import matplotlib.pyplot as plt

def reorder_center(n):
    order = []
    if n%2==0:
        m = n/2
        for i in range(m):
            order.append(m-i-1)
            order.append(m+i)
    else:
        m = (n-1)/2
        order.append(m)
        for i in range(m):
            order.append(m-i-1)
            order.append(m+i+1)
    assert len(order)==n
    return np.array(order)

def array_center(arr):
    n = len(list(arr))
    idx = np.array(reorder_center(n))
    #print 'idx', idx
    new_idx = np.zeros(n, dtype=np.int32)
    new_idx[idx] = np.array(list(range(n)))
    #print 'new_idx', new_idx
    new_arr = np.array(arr)[new_idx]
    #print 'new_arr', new_arr
    return new_arr 


def unique_everseen(iterable, key=None):
    "List unique elements, preserving order. Remember all elements ever seen."
    # unique_everseen('AAAABBBCCDAABBB') --> A B C D
    # unique_everseen('ABBCcAD', str.lower) --> A B C D
    seen = set()
    seen_add = seen.add
    if key is None:
        for element in ifilterfalse(seen.__contains__, iterable):
            seen_add(element)
            yield element
    else:
        for element in iterable:
            k = key(element)
            if k not in seen:
                seen_add(k)
                yield element



class Vmatrix(object):
    def __init__(self, mat, col=None, row=None):
        self.mat = np.array(mat, dtype=np.float32)
        self.col = np.array(col)
        self.row = np.array(row)

        if col is None:
            self.col = np.array(list(range(self.mat.shape[0])))
        if row is None:
            self.row = np.array(list(range(self.mat.shape[1])))

    def T(self):
        # transpose of the matrix
        return Vmatrix(self.mat.T, self.row, self.col)

    def mat_thresh(self, thresh = None):
        # remove rows where all values are smaller than thresh, then
        # remove cols where all values are smaller than thresh
        return self.col_thresh(thresh).T().col_thresh(thresh).T()
    
    def mat_sparsify(self, thresh=None):
        # set all values of the matrix to be 1 if >= thresh, else 0
        # remove rows/cols if all 0
        sparse_mat = np.where(self.mat>=thresh, 1, 0)
        return Vmatrix(sparse_mat, self.col, self.row).mat_thresh(1)
     
    def mat_col_idx(self, idx_col):
        # make a submatrix by keep only the selected columns indices
        return Vmatrix(self.mat[idx_col,:], self.col[idx_col], self.row)
    
    def mat_col_key(self, key_col):
        # make a submatrix by keep only the selected columns keys
        idx_col = [list(self.col).index(k) for k in key_col] 
        return Vmatrix(self.mat[idx_col,:], key_col, self.row)

    def col_thresh(self, thresh = None):
        # remove rows where all values are smaller than thresh
        # this means keeping only a sublist of original cols
        keep_col = np.invert(np.all(self.mat < thresh, axis=1, keepdims=False)).nonzero()[0]
        return self.mat_col_idx(keep_col)

    def col_sort(self):
        # sort columns in descending order according to row sum
        sorted_col = sorted(range(self.mat.shape[0]), key= lambda i: np.sum(self.mat[i,:]), reverse=True) 
        return self.mat_col_idx(sorted_col)

    #def row_sort(self):
    #    # sort rows in ascending order according to col sum
    #    return self.T().col_sort().T()

    def col_cluster(self, thresh):
        # 1. X = sparsified matrix
        # 2. sorted_row = sorted row of sparsified matrix
        # 3. clustered_col = cluster columns together according to row priority
        # 4. new_row/col_order = reorder_center(sorted_row/clustered_col)
        # returns X.mat[#centered(clustered_col), centered(sorted_row)]

        ### sparsify matrix
        #print 'sparsify'
        S = self.mat_sparsify(thresh)#.debug()
        
        ### sort rows
        #print 'sort row'
        V = S.T().col_sort().T()#.debug()
        sorted_row = V.row
        
        ### cluster columns
        #print 'cluster col'
        clustered_col = []
        for i, row in enumerate(sorted_row):
            for j, val in enumerate(V.mat[:, i]):
                if val!=0: 
                    clustered_col.append(V.col[j])
        clustered_col = np.array(list(unique_everseen(clustered_col)))
        #print clustered_col 

        new_col = array_center(clustered_col)
        new_row = array_center(sorted_row)
        
        #return self.mat_col_key(new_col).T().mat_col_key(new_row).T()
        return S.mat_col_key(new_col).T().mat_col_key(new_row).T()

    def view(self):
        plt.matshow(self.mat)
        plt.show()
        
        return self 

    def debug(self, max_display=10):
        print 'mat_shape', self.mat.shape
        print 'col ', self.col
        print 'row ', self.row
        #print '    ', self.row
        for i, row in enumerate(self.mat[:max_display]):
            print '   {}'.format(self.col[i]), row
        
        return self


def test_case():
    print '================================================================'
    print 'testing transpose'
    V = Vmatrix([[1,2],[1,3],[3,3],[4,1]], col=['a','b','c','d'], row=['x','y']).debug()
    A = V.T().debug()
    print '================================================================'

    print '================================================================'
    print 'testing column selection ops'
    G = V.col_thresh(3).debug()
    U = G.col_sort().debug()
    print '================================================================'

    print '================================================================'
    print 'testing matrix threshold ops'
    X = V.mat_thresh(4).debug()
    L = V.mat_sparsify(4).debug()
    print '================================================================'


if __name__=='__main__':
    #test_case()
    alphabets = list(string.ascii_lowercase)
    print alphabets
    print array_center(alphabets)
    #V = Vmatrix(np.random.rand(10,10), col=alphabets[:10]).debug().view()
    V = Vmatrix(np.random.rand(100,100)).debug().view()
    G = V.col_cluster(0.8).debug().view()

    
