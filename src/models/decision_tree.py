#TODO: Decision tree model
# -----------------------------------------------------------------------------
# Gini coefficient Function:
#
#   C        
#G= 1-∑  p(i)∗p(i)
#   i=1
#          
#
# where:
#   C   = total classes
#   p(i)   = probability of picking a datapoint with class i
#
#Child impurity function:
#
# child_imp=n_Left*Gini(y_left)+n_right*Gini(y_right)
#
#Impurity_decrease(Gain)=current_gini-child_imp
#
# In DT with Gini:
#   - Gini Impurity is the probability of incorrectly classifying a randomly chosen element in the dataset if it were randomly labeled
#   - Determines the Quality of the split in our tree
#   - Quality of the split is determined by weighting the impurity of each branch by how many elements it has
# -----------------------------------------------------------------------------
#Information gain Function
#           n
#Entropy =  ∑ P*Log(P)   log base 2
#           i=1
#where P is the probability that it is a function of entropy


import numpy as np


def load_50npz():
    data = np.load("./data/features/features_cifar10_resnet18_pca50.npz")
    X_train, y_train = data["X_train"], data["y_train"]
    X_test,  y_test  = data["X_test"],  data["y_test"]
    print (f"Loaded PCA-reduced features: X_train={X_train.shape}, X_test={X_test.shape}")
    return X_train, y_train, X_test, y_test

#G= 1-∑  p(i)∗p(i)
def gini_coefficient(y, n_classes=None):
    if y.size == 0:
        return 0.0
    if n_classes is None:
        n_classes=int(y.max())+1
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    p=counts/y.size
    return 1-np.sum(p*p)

#counts each bin and returns the bin with the highest count therefore our majority class
def majority_class(y):
    counts = np.bincount(y)
    return np.argmax(counts)

 
##Impurity_decrease(Gain)=current_gini-child_imp
def best_split_feature(x_col,y,n_classes):
    order = np.argsort(x_col)
    x_sorted = x_col[order]
    y_sorted = y[order]

    diffs = np.diff(x_sorted)
    valid_split = diffs>0
    if not np.any(valid_split):
        return np.inf,None
    
    n=y_sorted.size
    left_counts = np.zeros((n, n_classes), dtype=np.int64)
    for i in range(n):
        left_count[i, y_sorted[i]]+= 1
        if i>0:
            left_counts[i]+=left_counts[i-1]

    total_counts = left_counts[-1].astype(np.float64)
    

# child_imp=n_Left*Gini(y_left)+n_right*Gini(y_right)
def best_split(X,y,n_classes):
    n_samples, n_features = X.shape
    best_feat = None
    best_thresh = None
    best_imp= np.inf

    for i in range(n_features):
        imp, thr = best_split_feature(X[:, i],y,n_classes)
        if imp < best_imp:
            best_imp = imp
            best_feat = i
            best_thresh= thr
            return best_feat,best_thresh,best_imp




def make_leaf(y):
    return{"feature":None,"threshold":None,"left":None,"right":None, "prediction":majority_class(y)}

def build_tree(X, y,depth,max_depth,n_classes,min_samples_splits=2, min_impurity_decreases=0.2):
    curr_gini=gini_coefficient(y,n_classes) #if this comes back as 0, the node is pure
    
    #Reached the max depth, can split the node any further or the node is pure
    if(depth>=max_depth) or (y.size<min_samples_splits) or(curr_gini== 0.0):
        return make_leaf(y) 
    #Otherwise we are spliting the tree by features
    feat,thr,child_imp = best_split(X,y,n_classes)
    if feat is None:
        return make_leaf(y)
    
    if curr_gini-child_imp<min_impurity_decreases:
        return make_leaf(y)


def train_decision_tree_gini(X_train, y_train, max_depth=50,min_samples_split=2,min_impurity_decrease=0.0,n_classes=None):
    if n_classes is None:
        n_classes = int(max(y_train.max(),0))+1
        tree= build_tree()
    return tree, n_classes

    
def run_pipeline():
    X_train, y_train, X_test,y_test = load_50npz()

    tree, n_classes = train_decision_tree_gini( X_train, y_train,max_depth=50,min_sample_split=2, min_impurity_decrease=0.0)

    #y_pred = predict(tree,X_test)

   #acc=accuracy(y_test, y_pred)

    return tree


if __name__ == "__main__":
    run_pipeline()