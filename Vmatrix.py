import numpy as np


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

class Vmatrix(object):
    def __init__(self, mat, col=None, row=None):
        self.mat = np.array(mat, dtype=np.float32)
        self.col = np.array(col)
        self.row = np.array(row)

        if col is None:
            self.col = np.array(list(range(self.mat.shape[0])))
        if row is None:
            self.row = np.array(list(range(self.mat.shape[1])))

    def mat_transpose(self):
        # transpose of the matrix
        return Vmatrix(self.mat.T, self.row, self.col)

    def mat_thresh(self, thresh = None):
        # remove rows where all values are smaller than thresh, then
        # remove cols where all values are smaller than thresh
        return self.col_thresh(thresh).mat_transpose().col_thresh(thresh).mat_transpose()
    
    def mat_sparsify(self, thresh=None):
        # set all values of the matrix to be 1 if >= thresh, else 0
        sparse_mat = np.where(self.mat>=thresh, 1, 0)
        return Vmatrix(sparse_mat, self.col, self.row)
     
    def mat_col(self, col):
        # make a submatrix by keep only the selected columns
        return Vmatrix(self.mat[col,:], self.col[col], self.row)

    def col_thresh(self, thresh = None):
        # remove rows where all values are smaller than thresh
        # this means keeping only a sublist of original col indices
        keep_col = np.invert(np.all(self.mat < thresh, axis=1, keepdims=False)).nonzero()[0]
        return self.mat_col(keep_col)

    def col_sort(self):
        # sort columns in ascending order according to row sum
        sorted_col = sorted(range(self.mat.shape[0]), key= lambda i: np.sum(self.mat[i,:])) 
        return self.mat_col(sorted_col)
        
    def debug(self, max_display=6):
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
    A = V.mat_transpose().debug()
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
    test_case()
