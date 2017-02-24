import numpy as np
from zrst.util import MLF
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
        if thresh is None:
            #thresh = np.mean(self.mean_col().reshape(1,-1))
            thresh = np.sort(self.mat,axis=None)[int(0.9*self.mat.size)]
            print 'set sparse threshold to ', thresh
        sparse_mat = np.where(self.mat>=thresh, 1, 0)
        return Vmatrix(sparse_mat, self.col, self.row)
     
    def mat_key(self, key_col):
        # make a submatrix by keeping only the selected columns keys
        idx_col = [list(self.col).index(k) for k in key_col] 
        return Vmatrix(self.mat[idx_col,:], key_col, self.row)

    def col_thresh(self, thresh=None):
        # remove rows where all values are smaller than thresh
        # this means keeping only a sublist of original cols
        keep_col_idx = np.invert(np.all(self.mat < thresh, axis=1, keepdims=False)).nonzero()[0]
        return self.mat_key(self.col[keep_col_idx])

    def col_sort(self, key_list=None):
        # sort columns in descending order according to row sum
        if key_list is None:
            key_list = np.sum(self.mat, axis=1, keepdims=False)
        sorted_col_idx = sorted(range(self.mat.shape[0]), key= lambda i: key_list[i], reverse=True) 
        return self.mat_key(self.col[sorted_col_idx])

    def col_clip(self, n_cut = None):
        # discards the columns with the largest/smallest row sums
        if n_cut is None:
            p_keep = .90
            n_cut = int(len(self.col)*(1-p_keep)/2)
            print 'discarding ',2*n_cut,' tokens out of ', len(self.col)
    
        if n_cut > 0:
            clip_cols = self.col_sort().col[n_cut:-n_cut]
        else:
            clip_cols = self.col_sort().col

        return self.mat_key(clip_cols)



    def mean_col(self, key_col=None):
        if key_col is None:
            key_col = self.col
        return np.mean(self.mat_key(key_col).mat, axis=0)
    
    def mean_dist(self, u):
        return np.sqrt(np.sum((self.mat-u)**2, axis=1))
         
    def col_aggr(self):
        sorted_row = self.row
        clustered_col = []
        for i, row in enumerate(sorted_row):
            for j, val in enumerate(self.mat[:, i]):
                if val!=0: 
                    clustered_col.append(self.col[j])
        clustered_col = list(unique_everseen(clustered_col))
        for key in self.col:
            if key not in clustered_col:
                clustered_col.append(key)
        return self.mat_key(np.array(clustered_col))

    def col_center(self):
        new_col = array_center(self.col)
        return self.mat_key(new_col)

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
        #sorted_row = []
        #for i, row in enumerate(sorted_row):
        for i in reorder_center(V.mat.shape[1]):
            for j in reorder_center(V.mat.shape[0]):
                if V.mat[j, i]!=0: 
                    if V.col[j] not in clustered_col:
                        if len(clustered_col)%2==0:
                            clustered_col.append(V.col[j])
                            #sorted_row.append(V.row[i])
                        else:
                            clustered_col.append(V.col[j])
                            #sorted_row.append(V.row[i])
                            #clustered_col.insert(0, V.col[j])
                            #sorted_row.insert(0, V.row[i])
        #clustered_col = np.array(list(unique_everseen(clustered_col)))
        #print clustered_col 

        new_col = clustered_col
        new_row = sorted_row

        #new_col = array_center(clustered_col)
        
        #new_row = array_center(sorted_row)
        X = S.mat_key(new_col).T().mat_key(new_row).T()
        #X = X.T().col_sort().T()
    
        #X = X.T().col_center().T()
        #X = X.col_center()
        

        
        #return self.mat_key(new_col).T().mat_key(new_row).T()
        #return S.mat_key(new_col).T().mat_key(new_row).T()
        return X

    def mat_class(self, class_list, n_rep=None):
        # class list contains keys for each class
        # this function finds the representatives from each class that standout the most
        if n_rep is None:
            n_rep = min([len(x) for x in class_list]) 
            print 'setting the number of representatives to', n_rep

        selected_cols = []    
        u_total = self.mean_col(flatten(class_list))
        for key_clust in class_list:
            u_clust = self.mean_col(key_clust)
            d_clust = self.mat_key(key_clust).mean_dist(u_clust)
            d_total = self.mat_key(key_clust).mean_dist(u_total)
            selectd_cols_clust = self.mat_key(key_clust).col_sort(d_total/d_clust).col[:n_rep]
            selected_cols.append(selectd_cols_clust)
        return self.mat_key(flatten(selected_cols))


    def view(self):
        plt.matshow(self.mat)
        plt.show()
        return self 

    def debug(self, max_display=10):
        print '    ', self.row
        for i, row in enumerate(self.mat[:max_display]):
            print '   {}'.format(self.col[i]), row
        print 'col ', self.col
        print 'row ', self.row
        print 'mat_shape', self.mat.shape
        
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

def test_order():
    alphabets = list(string.ascii_lowercase)
    print alphabets
    print array_center(alphabets)

    print reorder_center(10)

def test_rand():
    #V = Vmatrix(np.random.rand(10,10), col=alphabets[:10]).debug().view()
    V = Vmatrix(np.random.rand(100,100)).debug().view()
    G = V.col_cluster(0.8).debug().view()


def vector_occr(word_list, vocab_list):
    return np.array(map(lambda x: word_list.count(x), vocab_list),dtype=np.float32)

def matrix_from_mlf(mlf_path):
    mlf = MLF(mlf_path)
    speaker_list, vector_list = [], []
    vocab_list = mlf.tok_list
    for i, tags in enumerate(mlf.tag_list):
        speaker = '_'.join(mlf.wav_list[i].split('_')[1:3])
        vector = vector_occr(tags, vocab_list)
        if i!=0 and speaker_list[-1]==speaker:
            vector_list[-1]+=vector
        else:
            #if i!=0: print speaker_list[-1], vector_list[-1]
            speaker_list.append(speaker)
            vector_list.append(vector)
    matrix = np.array(vector_list)
    return matrix, speaker_list, vocab_list

def sort_speaker_list(speaker_list):
    speaker_class = ['f1', 'f2', 'f3', 'f4', 'f5','f6', 'f7', 'f8', 'm1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8']
    speaker_dict = {}
    for sp in speaker_list:
        gr = sp[4] + sp[2] 
        if gr not in speaker_dict:
            speaker_dict[gr] = [sp,]
        else:   
            speaker_dict[gr].append(sp)
    nested_list = []
    for cl in speaker_class:
        print 'there are {} speakers in class {}'.format(len(speaker_dict[cl]), cl)
        nested_list.append(speaker_dict[cl])
    return nested_list


#flatten = lambda l: [item for sublist in l for item in sublist]

def flatten(nested_list):
    return [item for sublist in nested_list for item in sublist]


def get_speakers(mlf_path):
    SV, speaker_list, vocab_list = matrix_from_mlf(mlf_path)
    V = Vmatrix(SV, speaker_list, vocab_list)
    spclass_list = sort_speaker_list(speaker_list)
    female_speakers = V.mat_class(spclass_list[:8], n_rep = 5).col
    male_speakers = V.mat_class(spclass_list[8:], n_rep = 5).col
    return female_speakers, male_speakers

def contrast_matrix(mlf_path, (female_speakers, male_speakers)):
    N = int(mlf_path.split('/')[-3].split('_')[0])
    M = int(mlf_path.split('/')[-3].split('_')[1])
    nn = 25
    mm = 5

    SV, speaker_list, vocab_list = matrix_from_mlf(mlf_path)
    V = Vmatrix(SV, speaker_list, vocab_list)
    #V = V.T().col_clip(mm).T()
    #V = V.T().col_sort().T()
    
    A = V.mat_key(list(female_speakers) + list(male_speakers))
    F = V.mat_key(female_speakers)
    M = V.mat_key(male_speakers)
    phn_dist = (F.mean_col()-M.mean_col())
    A = A.T().col_sort(phn_dist).T()
    

    #contrast_row = list(A.row[::2]) 
    #contrast_row = list(A.row[:nn])+list(A.row[-nn:]) 
    contrast_row = A.row


    X = A.T().mat_key(contrast_row).T()
    

    ###############AGGREGATE ROWS##################
    ag_rows = X.T().mat_sparsify().col_aggr().T().row
    X = X.T().mat_key(ag_rows).T()
    ###############################################

    return X
    #Vmatrix(X.mat**0.5, X.col,X.row).view()


if __name__=='__main__':
    mlf_path = '/home/c2tao/timit_train_matdnn_nosa/tokenizer_bnf0_mr1/{}_{}/result/result.mlf'
    #mlf_path = '/mnt/c/Users/c2tao/Desktop/Semester 18/tokenizer_bnf0_mr1/500_9/result/result.mlf'
    M = 5
    N = 100
    m_list = [3, 5, 7, 9]
    n_list = [50, 100, 300, 500] 
    female, male = get_speakers(mlf_path.format(N, M))
    mats = []
    for i in range(4):
        N = n_list[i]
        #M = m_list[i] 
        X = contrast_matrix(mlf_path.format(N, M), (female, male))
        mats.append(X)
    #X_mats = np.concatenate(mats, axis = 1)
    

    fig, axes = plt.subplots(1, 4, sharey=True)
    for i, ax in enumerate(axes):
        N = n_list[i]
        #M = m_list[i] 
        im = ax.imshow(1.0 - np.exp(-mats[i].mat*N*M*0.0005), interpolation='nearest', aspect='auto', vmin=0, vmax=1, cmap='gray')
        ax.set_xticks(np.linspace(0, len(mats[i].row), 11))
        ax.set_title('MR = 1\n n = {}\n m = {}'.format(N, M))
        plt.setp(ax.get_xticklabels(), rotation=90)
    cax = fig.add_axes([0.95, 0.1, 0.01, 0.8])
    im = fig.colorbar(im, cax=cax)
    print axes[0].get_xticks()
    print axes[0].get_yticks()
    
    dr_list = ['New England', 'Northern', 'North Midland', 'South Midland', 'Southern', 'New York City', 'Western', 'Army Brat']
    fm_list = ['Female', 'Male']
    fmdr_list = [fm+' '+dr for fm in fm_list for dr in dr_list]
    
    axes[0].set_yticks(np.arange(0+2.5, 80+2.5, 5))
    axes[0].set_yticklabels(fmdr_list, horizontalalignment='right')
    
    #plt.tight_layout()
    print fmdr_list
    plt.show()
