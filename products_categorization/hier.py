import networkx as nx
from sklearn_hierarchical_classification.classifier import HierarchicalClassifier
from sklearn.linear_model import LogisticRegression
import pickle
import time
from scipy.sparse import csr_matrix
import numpy as np
from tqdm.notebook import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn_hierarchical_classification.metrics import h_precision_score, h_recall_score
from networkx import relabel_nodes

cmap = ListedColormap(['dodgerblue', 'lightgray', 'darkorange'])

# Обертка для локальных классификаторов, позволяющая по индексам на входе - X, получать тренировочные и валидационные образцы из датасетов X_train и X_val.
class WrapBaseModel:
    def __init__(self, base_model, hiclf):
        self.model = base_model
        self.hiclf = hiclf

    def fit(self, X, y):
        self.model.fit(self.hiclf.X_train[self.indexis(X)], y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
    def decision_function(self, X):
        return self.model.decision_function(X)
    
    @property
    def classes_(self):
        return self.model.classes_

    def indexis(self, X):
        if isinstance(X, np.ma.core.MaskedArray):
            return X.astype(int).ravel()
        return X.toarray().astype(int).ravel()

# Иерархический классификатор вида local classifier per parent node
# Изменяет библиотечный HierarchicalClassifier таким образом:
# При передаче в fit например тренировочного датасета X_train, HiClassifier выбирает из него только индексы образцов и передает их HierarchicalClassifier, который строит все разбиения для локальных классификаторов. Также HiClassifier убеждается, что локальные классификаторы смогут по индексам восстановить полные образцы из X_train.
class HiClassifier(HierarchicalClassifier):
    def __init__(
        self,
        base_estimator=None,
        class_hierarchy=None,
        algorithm="lcpn",
        root=1,
        progress_wrapper=tqdm,
        path_graph=None,
        random_state=None
    ):
        if base_estimator is None:
            base_estimator = LogisticRegression(
    multi_class="multinomial", max_iter=1000, random_state=random_state
)
        
        if path_graph:
            with open(path_graph, "rb") as f:
                class_hierarchy = pickle.load(f)
                
        self.path_graph = path_graph
        self.graph_ = None
        self.X_train = None
        
        super().__init__(
            base_estimator=base_estimator,
            class_hierarchy=class_hierarchy,
            algorithm=algorithm,
            root=root,
            progress_wrapper=progress_wrapper,
        )

    def fit(self, X, y=None, rebuild_features=False):
                
        self.X_train = X
        self.clear_classifiers()
        
        if (self.X_train is not None and X.shape[0] != self.X_train.shape[0]) or rebuild_features:
            self.clear_features()

        t_start = time.time()
        super().fit(self.simplify(X), y)        
        self.time_fit = round((time.time()-t_start)/60,1)
        
        return self

    def predict(self, X):
        t_start = time.time()        
        y_pred = super().predict(X)       
        self.time_predict = round((time.time()-t_start)/60,1)       
        return y_pred
        
    def simplify(self, X):
        return csr_matrix(np.arange(X.shape[0])[:, None])

    def clear_classifiers(self):
        if self.graph_:
            for node in self.graph_.nodes:
                self.graph_.nodes[node].pop("classifier",None)
                
    def clear_features(self):
        print('clear features')
        if self.graph_:
            for node in self.graph_.nodes:
                self.graph_.nodes[node].pop("X",None)
                
    def save_graph(self, path_graph):
        self.clear_classifiers()

        with open(path_graph, "wb") as f:
            pickle.dump(self.graph_, f)

    def _build_metafeatures(self, X, y):
        return 0

    def _base_estimator_for(self, node_id):
        if callable(self.base_estimator):
            model = self.base_estimator(node_id=node_id, graph=self.graph_)
        else:
            model = super()._base_estimator_for(node_id)

        return WrapBaseModel(model, self)
    
def h_scores(y_true, y_pred, graph, beta=1., root=1, round_result=3):
    """
    Calculate the micro-averaged hierarchical hP, hR, F1-beta ("hF_{\beta}") metrics based on
    given set of true class labels and predicated class labels, and the
    class hierarchy graph.
    """
    
    mlb = MultiLabelBinarizer()
    all_classes = [
        node
        for node in graph.nodes
        if node != root
    ]
    
    mlb.fit([all_classes])

    node_label_mapping = {
        old_label: new_label
        for new_label, old_label in enumerate(list(mlb.classes_))
    }
    
    if len(y_true.shape) == 1:
        y_true = y_true[:, None]
        
    if len(y_pred.shape) == 1:
        y_pred = y_pred[:, None]
    
    y_true = mlb.transform(y_true)
    y_pred = mlb.transform(y_pred)
    graph = relabel_nodes(graph, node_label_mapping)
    
    hP = h_precision_score(y_true, y_pred, graph, root=root)
    hR = h_recall_score(y_true, y_pred, graph, root=root)
    hF1 = (1. + beta ** 2.) * hP * hR / (beta ** 2. * hP + hR)
    r = round_result
    
    return round(hP,r), round(hR,r), round(hF1,r)
    

# отображает на графике топ n вершин графа G
def draw_top_graph(G, n=20):
    fig, ax = plt.subplots(figsize=(20,12))
    top_nodes = list(nx.topological_sort(G))[:n]
    G = G.subgraph(top_nodes)
    
    colors = {1:0, 10012:1, 10014:1, 10018:1, 10020:1, 10021:1, 10003:1}
    node_colors = [colors.get(node,2) for node in list(G)]

    node_sizes = [5000 if node == 1 else 2500 for node in list(G)]
    
    nx.draw_networkx(G, pos=nx.planar_layout(G), node_color=node_colors, node_size=node_sizes,cmap=cmap)

# сокращает использование памяти для pandas DataFrame
def reduce_mem(df, silent=True):
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if str(col_type)[:5] == "float":
            c_min = df[col].min()
            c_max = df[col].max()
            if c_min > np.finfo("f2").min and c_max < np.finfo("f2").max:
                df[col] = df[col].astype(np.float16)
            elif c_min > np.finfo("f4").min and c_max < np.finfo("f4").max:
                df[col] = df[col].astype(np.float32)
            else:
                df[col] = df[col].astype(np.float64)
        elif str(col_type)[:3] == "int":
            c_min = df[col].min()
            c_max = df[col].max()
            if c_min > np.iinfo("i1").min and c_max < np.iinfo("i1").max:
                df[col] = df[col].astype(np.int8)
            elif c_min > np.iinfo("i2").min and c_max < np.iinfo("i2").max:
                df[col] = df[col].astype(np.int16)
            elif c_min > np.iinfo("i4").min and c_max < np.iinfo("i4").max:
                df[col] = df[col].astype(np.int32)
            elif c_min > np.iinfo("i8").min and c_max < np.iinfo("i8").max:
                df[col] = df[col].astype(np.int64)
        # elif col == "timestamp":
        #     df[col] = pd.to_datetime(df[col])
        # elif str(col_type)[:8] != "datetime":
        #     df[col] = df[col].astype("category")
    end_mem = df.memory_usage().sum() / 1024**2
    if not silent:
        print(f'Память ДО: {round(start_mem,1)} Мб')
        print(f'Память ПОСЛЕ: {round(end_mem,1)} Мб')
        print('Уменьшилось на',
              round(start_mem - end_mem, 2),
              'Мб (минус',
              round(100 * (start_mem - end_mem) / start_mem, 1),
              '%)')
    return