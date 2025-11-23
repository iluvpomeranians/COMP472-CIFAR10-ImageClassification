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
from sklearn.tree import DecisionTreeClassifier
from src.utils.metrics import Metrics
class DTree:
    @staticmethod
    def load_50npz():
        data = np.load("./data/features/features_cifar10_resnet18_pca50.npz")
        X_train, y_train = data["X_train"], data["y_train"]
        X_test,  y_test  = data["X_test"],  data["y_test"]
        print (f"Loaded PCA-reduced features: X_train={X_train.shape}, X_test={X_test.shape}")
        print (f"Label ranges: train[{y_train.min()}..{y_train.max()}], test[{y_test.min()}..{y_test.max()}]")
        return X_train, y_train, X_test, y_test

    #G= 1-∑  p(i)∗p(i)
    @staticmethod
    def gini_coefficient(y, n_classes=None):
        if y.size == 0:
            return 0.0
        if n_classes is None:
            n_classes=int(y.max())+1
        counts = np.bincount(y, minlength=n_classes).astype(np.float64)
        p=counts/y.size
        return 1-np.sum(p*p)

    #counts each bin and returns the bin with the highest count therefore our majority class
    @staticmethod
    def majority_class(y):
        counts = np.bincount(y)
        return np.argmax(counts)


    ##Impurity_decrease(Gain)=current_gini-child_imp
    @staticmethod
    def best_split_feature(x_col,y,n_classes):
        order = np.argsort(x_col)
        x_sorted = x_col[order]
        y_sorted = y[order]

    #finding splits where only feature value changes
        diffs = np.diff(x_sorted)
        valid_split = diffs>0
        if not np.any(valid_split):
            return np.inf,None

        n=y_sorted.size

        #class count for left side
        left_counts = np.zeros((n, n_classes), dtype=np.int64)
        for i in range(n):
            left_counts[i, y_sorted[i]]+= 1
            if i>0:
                left_counts[i]+=left_counts[i-1]

        total_counts = left_counts[-1].astype(np.float64)

        idx = np.nonzero(valid_split)[0]
        left_sizes = (idx+1).astype(np.float64)
        right_sizes=(n-(idx+1)).astype(np.float64)

        lc=left_counts[idx].astype(np.float64)
        rc=(total_counts-lc)

        #handles warnings of dividing by 0 or any other invalid operations
        with np.errstate(divide='ignore',invalid='ignore'):
            lp=lc/left_sizes[:,None]
            rp=rc/right_sizes[:,None]
            left_gini=1.0-np.sum(lp*lp,axis=1)
            right_gini=1.0-np.sum(rp*rp, axis=1)

        #weighted impurity
        weighted=(left_sizes/n)*left_gini+(right_sizes/n)*(right_gini)
        best_idx= np.argmin(weighted)
        k = idx[best_idx]

        thresh=0.5*(x_sorted[k]+x_sorted[k+1])
       # print(f"[BEST_SPLIT_FEATURE] Processing 1 feature with {len(y)} samples")
        return weighted[best_idx], thresh


    # child_imp=n_Left*Gini(y_left)+n_right*Gini(y_right)
    @staticmethod
    def best_split(X,y,n_classes):
        n_samples, n_features = X.shape
        best_feat = None
        best_thresh = None
        best_imp= np.inf
        #print(f"\n[BEST_SPLIT] Trying {n_features} features on {n_samples} samples")

        for i in range(n_features):
            imp, thr = DTree.best_split_feature(X[:, i],y,n_classes)
            if imp < best_imp:
                best_imp = imp
                best_feat = i
                best_thresh= thr
                #print(f"[BEST_SPLIT] -> Chosen feature {best_feat} with threshold {best_thresh}, impurity={best_imp:.4f}")
        return best_feat,best_thresh,best_imp

    @staticmethod
    def make_leaf(y):
        return{"feature":None,"threshold":None,"left":None,"right":None, "prediction":DTree.majority_class(y)}

    @staticmethod
    def build_tree(X, y,depth,max_depth,n_classes,min_samples_split=2, min_impurity_decrease=0.0):
        #print(f"\n[BUILD] Depth={depth}, Samples={len(y)}, Gini={DTree.gini_coefficient(y, n_classes):.4f}")
        curr_gini=DTree.gini_coefficient(y,n_classes) #if this comes back as 0, the node is pure

        #Reached the max depth, can split the node any further or the node is pure
        if(depth>=max_depth) or (y.size<min_samples_split) or(curr_gini== 0.0):
            #print(f"[STOP] Reached leaf condition (depth={depth}, samples={len(y)})")
            return DTree.make_leaf(y)


        #Otherwise we are spliting the tree by features
        feat,thr,child_imp = DTree.best_split(X,y,n_classes)
        #print(f"[SPLIT] Best feature={feat}, threshold={thr}, child_impurity={child_imp:.4f}")
        if feat is None:
            #print("[STOP] No valid feature found.")
            return DTree.make_leaf(y)


        #If the impurity is too small we just make a leaf
        if curr_gini-child_imp<min_impurity_decrease:
            #print("[STOP] Gain below threshold")
            return DTree.make_leaf(y)

        #Splitting features left and right node
        left_mask = X[:, feat]<= thr
        right_mask = np.logical_not(left_mask)
        #print(f"[MASK] Left={np.sum(left_mask)}, Right={np.sum(right_mask)}")

    #If all the samples go to the same side(repeated values)
        if not left_mask.any() or not right_mask.any():
            #print("[STOP] One side empty — making leaf.")
            return DTree.make_leaf(y)
        #recursively build the subnodes
        left = DTree.build_tree(X[left_mask],y[left_mask], depth+1, max_depth, n_classes,min_samples_split,min_impurity_decrease)
        right=DTree.build_tree(X[right_mask],y[right_mask], depth+1, max_depth, n_classes,min_samples_split,min_impurity_decrease)
        #print(f"[NODE] Finished node depth={depth}, feature={feat}, threshold={thr}")

        return{"feature": int(feat),"threshold":float(thr),"left":left,"right":right,"prediction":DTree.majority_class(y)}
    @staticmethod
    def train_decision_tree_gini(X_train, y_train, max_depth,min_samples_split=2,min_impurity_decrease=0.0,n_classes=None):
        if n_classes is None:
            n_classes = int(max(y_train.max(),0))+1
            print(f"\n[TRAIN] Training with {len(y_train)} samples, {n_classes} classes")
        tree= DTree.build_tree(X_train,y_train,depth=0, max_depth=max_depth, n_classes=n_classes,min_samples_split=min_samples_split,min_impurity_decrease=min_impurity_decrease)
        print("[TRAIN] Tree building complete.\n")
        return tree, n_classes
    @staticmethod
    def train_sklearn_decision_tree(X_train,Y_train,*,max_depth,min_samples_split,min_samples_leaf,max_features,criterion,random_state):
        clf=DecisionTreeClassifier(criterion=criterion,max_depth=max_depth,min_samples_split=min_samples_split,min_samples_leaf=min_samples_leaf,max_features=max_features,random_state=random_state)
        clf.fit(X_train, Y_train)
        return clf
    @staticmethod
    def predict_one(tree, x):
        node = tree
        while node["feature"] is not None:
            if x[node["feature"]]<=node["threshold"]:
                node = node["left"]
            else:
                node= node["right"]

        return node["prediction"]
    @staticmethod
    def predict(tree,X):
        return np.array([DTree.predict_one(tree,x) for x in X], dtype=np.int64)
    @staticmethod
    def accuracy(y_true,y_pred):
        return np.mean(y_true == y_pred)



