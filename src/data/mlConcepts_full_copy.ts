export const mlConcepts = [
  {
    id: "1",
    title: `Linear Regression`,
    category: `Supervised`,
    description: `Linear Regression — concise ~100-word description. Learn core ideas, when to use it, and practical tips.`,
    code: `import numpy as np
from sklearn.linear_model import LinearRegression
X = np.array([[1],[2],[3],[4],[5]])
y = np.array([2,4,6,8,10])
model = LinearRegression().fit(X,y)
print('coef=', model.coef_[0], 'intercept=', model.intercept_)
print('pred for 6 ->', model.predict([[6]])[0])
`
  },
  {
    id: "2",
    title: `Logistic Regression`,
    category: `Unsupervised`,
    description: `Logistic Regression — concise ~100-word description. Learn core ideas, when to use it, and practical tips.`,
    code: `import numpy as np
from sklearn.linear_model import LogisticRegression
X = np.array([[0.1],[0.5],[1.0],[1.5],[2.0]])
y = np.array([0,0,0,1,1])
model = LogisticRegression().fit(X,y)
print('probs:', model.predict_proba([[0.2],[1.8]]))
print('pred:', model.predict([[0.2],[1.8]]))
`
  },
  {
    id: "3",
    title: `Decision Trees`,
    category: `Reinforcement`,
    description: `Decision Trees — concise ~100-word description. Learn core ideas, when to use it, and practical tips.`,
    code: `from sklearn.tree import DecisionTreeClassifier
X = [[0,0],[1,1],[1,0],[0,1]]
y = [0,1,1,0]
clf = DecisionTreeClassifier().fit(X,y)
print('predict [1,0]:', clf.predict([[1,0]])[0])
`
  },
  {
    id: "4",
    title: `Random Forests`,
    category: `Neural`,
    description: `Random Forests — concise ~100-word description. Learn core ideas, when to use it, and practical tips.`,
    code: `from sklearn.ensemble import RandomForestClassifier
X = [[0,0],[1,1],[1,0],[0,1]]
y = [0,1,1,0]
clf = RandomForestClassifier(n_estimators=10, random_state=0).fit(X,y)
print('predict:', clf.predict([[0,0],[1,1]]))
`
  },
  {
    id: "5",
    title: `Gradient Boosting Machines`,
    category: `Supervised`,
    description: `Gradient Boosting Machines — concise ~100-word description. Learn core ideas, when to use it, and practical tips.`,
    code: `from sklearn.ensemble import HistGradientBoostingClassifier
X = [[0,0],[1,1],[1,0],[0,1]]
y = [0,1,1,0]
clf = HistGradientBoostingClassifier().fit(X,y)
print('predict:', clf.predict([[0,0],[1,1]]))
`
  },
  {
    id: "6",
    title: `Support Vector Machines`,
    category: `Unsupervised`,
    description: `Support Vector Machines — concise ~100-word description. Learn core ideas, when to use it, and practical tips.`,
    code: `from sklearn.svm import SVC
X = [[0,0],[1,1],[1,0],[0,1]]
y = [0,1,1,0]
clf = SVC(kernel='linear').fit(X,y)
print('support vectors:', clf.support_vectors_)
`
  },
  {
    id: "7",
    title: `K-Nearest Neighbors`,
    category: `Reinforcement`,
    description: `K-Nearest Neighbors — concise ~100-word description. Learn core ideas, when to use it, and practical tips.`,
    code: `from sklearn.neighbors import KNeighborsClassifier
X = [[0],[1],[2],[3]]
y = [0,0,1,1]
knn = KNeighborsClassifier(n_neighbors=3).fit(X,y)
print('predict [1.5]:', knn.predict([[1.5]]))
`
  },
  {
    id: "8",
    title: `K-Means Clustering`,
    category: `Neural`,
    description: `K-Means Clustering — concise ~100-word description. Learn core ideas, when to use it, and practical tips.`,
    code: `import numpy as np
from sklearn.cluster import KMeans
X = np.array([[1,2],[1,4],[1,0],[10,2],[10,4],[10,0]])
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
print('centers:', kmeans.cluster_centers_)
print('labels:', kmeans.labels_)
`
  },
  {
    id: "9",
    title: `Hierarchical Clustering`,
    category: `Supervised`,
    description: `Hierarchical Clustering — concise ~100-word description. Learn core ideas, when to use it, and practical tips.`,
    code: `from scipy.cluster.hierarchy import linkage, fcluster
import numpy as np
X = np.array([[1,2],[1,4],[1,0],[10,2],[10,4]])
Z = linkage(X, method='ward')
labels = fcluster(Z, t=2, criterion='maxclust')
print('labels:', labels)
`
  },
  {
    id: "10",
    title: `Principal Component Analysis`,
    category: `Unsupervised`,
    description: `Principal Component Analysis — concise ~100-word description. Learn core ideas, when to use it, and practical tips.`,
    code: `import numpy as np
from sklearn.decomposition import PCA
X = np.array([[1,2],[2,3],[3,4],[4,5]])
pca = PCA(n_components=1).fit(X)
print('explained variance ratio:', pca.explained_variance_ratio_)
print('transformed:', pca.transform(X))
`
  },
  {
    id: "11",
    title: `t-SNE`,
    category: `Reinforcement`,
    description: `t-SNE — concise ~100-word description. Learn core ideas, when to use it, and practical tips.`,
    code: `# t-SNE usually requires sklearn.manifold
from sklearn.manifold import TSNE
import numpy as np
X = np.random.RandomState(0).randn(100, 5)
emb = TSNE(n_components=2, random_state=0).fit_transform(X)
print('embedding shape:', emb.shape)
`
  },
  {
    id: "12",
    title: `DBSCAN`,
    category: `Neural`,
    description: `DBSCAN — concise ~100-word description. Learn core ideas, when to use it, and practical tips.`,
    code: `import numpy as np
from sklearn.cluster import DBSCAN
X = np.array([[1,2],[2,2],[2,3],[8,7],[8,8]])
clustering = DBSCAN(eps=1.5, min_samples=2).fit(X)
print('labels:', clustering.labels_)
`
  },
  {
    id: "13",
    title: `Gaussian Mixture Models`,
    category: `Supervised`,
    description: `Gaussian Mixture Models — concise ~100-word description. Learn core ideas, when to use it, and practical tips.`,
    code: `import numpy as np
from sklearn.mixture import GaussianMixture
X = np.vstack([np.random.normal(-2,1,(50,2)), np.random.normal(3,1,(50,2))])
gm = GaussianMixture(n_components=2, random_state=0).fit(X)
print('means:', gm.means_)
`
  },
  {
    id: "14",
    title: `Naive Bayes`,
    category: `Unsupervised`,
    description: `Naive Bayes — concise ~100-word description. Learn core ideas, when to use it, and practical tips.`,
    code: `from sklearn.naive_bayes import GaussianNB
X = [[1,2],[2,1],[1,0],[0,1]]
y = [0,0,1,1]
clf = GaussianNB().fit(X,y)
print('predict [1,1]:', clf.predict([[1,1]])[0])
`
  },
  {
    id: "15",
    title: `Ensemble Methods`,
    category: `Reinforcement`,
    description: `Ensemble Methods — concise ~100-word description. Learn core ideas, when to use it, and practical tips.`,
    code: `from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
X = [[0,0],[1,1],[1,0],[0,1]]
y = [0,1,1,0]
clf1 = LogisticRegression()
clf2 = DecisionTreeClassifier()
clf3 = SVC(probability=True)
eclf = VotingClassifier([('lr',clf1),('dt',clf2),('svc',clf3)]).fit(X,y)
print('predict:', eclf.predict([[0,0],[1,1]]))
`
  },
  {
    id: "16",
    title: `AdaBoost`,
    category: `Neural`,
    description: `AdaBoost — concise ~100-word description. Learn core ideas, when to use it, and practical tips.`,
    code: `from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
X = [[0,0],[1,1],[1,0],[0,1]]
y = [0,1,1,0]
clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=10).fit(X,y)
print('predict:', clf.predict([[0,0],[1,1]]))
`
  },
  {
    id: "17",
    title: `XGBoost`,
    category: `Supervised`,
    description: `XGBoost — concise ~100-word description. Learn core ideas, when to use it, and practical tips.`,
    code: `# XGBoost/LightGBM/CatBoost may not be available in this environment.
# Using sklearn's HistGradientBoosting as an alternative example.
from sklearn.ensemble import HistGradientBoostingClassifier
X = [[0,0],[1,1],[1,0],[0,1]]
y = [0,1,1,0]
clf = HistGradientBoostingClassifier().fit(X,y)
print('predict:', clf.predict([[0,0],[1,1]]))
`
  },
  {
    id: "18",
    title: `LightGBM`,
    category: `Unsupervised`,
    description: `LightGBM — concise ~100-word description. Learn core ideas, when to use it, and practical tips.`,
    code: `# XGBoost/LightGBM/CatBoost may not be available in this environment.
# Using sklearn's HistGradientBoosting as an alternative example.
from sklearn.ensemble import HistGradientBoostingClassifier
X = [[0,0],[1,1],[1,0],[0,1]]
y = [0,1,1,0]
clf = HistGradientBoostingClassifier().fit(X,y)
print('predict:', clf.predict([[0,0],[1,1]]))
`
  },
  {
    id: "19",
    title: `CatBoost`,
    category: `Reinforcement`,
    description: `CatBoost — concise ~100-word description. Learn core ideas, when to use it, and practical tips.`,
    code: `# XGBoost/LightGBM/CatBoost may not be available in this environment.
# Using sklearn's HistGradientBoosting as an alternative example.
from sklearn.ensemble import HistGradientBoostingClassifier
X = [[0,0],[1,1],[1,0],[0,1]]
y = [0,1,1,0]
clf = HistGradientBoostingClassifier().fit(X,y)
print('predict:', clf.predict([[0,0],[1,1]]))
`
  },
  {
    id: "20",
    title: `Regularization (L1/L2)`,
    category: `Neural`,
    description: `Regularization (L1/L2) — concise ~100-word description. Learn core ideas, when to use it, and practical tips.`,
    code: `import numpy as np
from sklearn.linear_model import Ridge, Lasso
X = np.array([[1],[2],[3],[4]])
y = np.array([2,4,6,8])
print('Ridge coef:', Ridge(alpha=1.0).fit(X,y).coef_)
print('Lasso coef:', Lasso(alpha=0.1).fit(X,y).coef_)
`
  },
  {
    id: "21",
    title: `Bias-Variance Tradeoff`,
    category: `Supervised`,
    description: `Bias-Variance Tradeoff — concise ~100-word description. Learn core ideas, when to use it, and practical tips.`,
    code: `# This is a conceptual topic. Try comparing underfit/overfit models in experiments.
print('Compare simple vs complex models and observe train/test error')
`
  },
  {
    id: "22",
    title: `Cross-Validation`,
    category: `Unsupervised`,
    description: `Cross-Validation — concise ~100-word description. Learn core ideas, when to use it, and practical tips.`,
    code: `from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
import numpy as np
X = np.arange(20).reshape(-1,1)
y = X.ravel()*2 + 1
scores = cross_val_score(LinearRegression(), X, y, cv=5)
print('CV scores:', scores)
`
  },
  {
    id: "23",
    title: `Hyperparameter Tuning`,
    category: `Reinforcement`,
    description: `Hyperparameter Tuning — concise ~100-word description. Learn core ideas, when to use it, and practical tips.`,
    code: `from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
X = [[0,0],[1,1],[1,0],[0,1]]
y = [0,1,1,0]
params = {'C':[0.1,1,10], 'kernel':['linear','rbf']}
gs = GridSearchCV(SVC(), params, cv=2).fit(X,y)
print('best params:', gs.best_params_)
`
  },
  {
    id: "24",
    title: `Grid Search`,
    category: `Neural`,
    description: `Grid Search — concise ~100-word description. Learn core ideas, when to use it, and practical tips.`,
    code: `from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
X = [[0,0],[1,1],[1,0],[0,1]]
y = [0,1,1,0]
params = {'C':[0.1,1,10], 'kernel':['linear','rbf']}
gs = GridSearchCV(SVC(), params, cv=2).fit(X,y)
print('best params:', gs.best_params_)
`
  },
  {
    id: "25",
    title: `Random Search`,
    category: `Supervised`,
    description: `Random Search — concise ~100-word description. Learn core ideas, when to use it, and practical tips.`,
    code: `from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
X = [[0,0],[1,1],[1,0],[0,1]]
y = [0,1,1,0]
params = {'C':[0.1,1,10], 'kernel':['linear','rbf']}
gs = GridSearchCV(SVC(), params, cv=2).fit(X,y)
print('best params:', gs.best_params_)
`
  },
  {
    id: "26",
    title: `Bayesian Optimization`,
    category: `Unsupervised`,
    description: `Bayesian Optimization — concise ~100-word description. Learn core ideas, when to use it, and practical tips.`,
    code: `# Bayesian Optimization requires external packages (e.g., scikit-optimize). Use GridSearch as a practical alternative here.
print('Use skopt or optuna in full Python environments')
`
  },
  {
    id: "27",
    title: `Feature Engineering`,
    category: `Reinforcement`,
    description: `Feature Engineering — concise ~100-word description. Learn core ideas, when to use it, and practical tips.`,
    code: `print('Feature engineering: create new features, e.g., ratios, log transforms, date parts, interactions')
`
  },
  {
    id: "28",
    title: `Feature Selection`,
    category: `Neural`,
    description: `Feature Selection — concise ~100-word description. Learn core ideas, when to use it, and practical tips.`,
    code: `from sklearn.feature_selection import SelectKBest, f_classif
X = [[1,2,3],[2,1,3],[1,0,4],[2,2,2]]
y = [0,1,0,1]
sel = SelectKBest(f_classif, k=2).fit(X,y)
print('scores:', sel.scores_)
`
  },
  {
    id: "29",
    title: `One-Hot Encoding`,
    category: `Supervised`,
    description: `One-Hot Encoding — concise ~100-word description. Learn core ideas, when to use it, and practical tips.`,
    code: `from sklearn.preprocessing import OneHotEncoder
import numpy as np
X = np.array([['red'], ['blue'], ['green'], ['red']])
ohe = OneHotEncoder(sparse=False).fit(X)
print(ohe.transform(X))
`
  },
  {
    id: "30",
    title: `Label Encoding`,
    category: `Unsupervised`,
    description: `Label Encoding — concise ~100-word description. Learn core ideas, when to use it, and practical tips.`,
    code: `from sklearn.preprocessing import LabelEncoder
y = ['cat','dog','cat','mouse']
le = LabelEncoder().fit(y)
print(le.transform(y))
`
  },
  {
    id: "31",
    title: `Imputation`,
    category: `Reinforcement`,
    description: `Imputation — concise ~100-word description. Learn core ideas, when to use it, and practical tips.`,
    code: `from sklearn.impute import SimpleImputer
import numpy as np
X = np.array([[1,2],[np.nan,3],[7,np.nan]])
imp = SimpleImputer(strategy='mean').fit(X)
print(imp.transform(X))
`
  },
  {
    id: "32",
    title: `Scaling and Normalization`,
    category: `Neural`,
    description: `Scaling and Normalization — concise ~100-word description. Learn core ideas, when to use it, and practical tips.`,
    code: `from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
X = np.array([[1,2],[2,4],[3,6]])
print('standardized:', StandardScaler().fit_transform(X))
print('minmax:', MinMaxScaler().fit_transform(X))
`
  },
  {
    id: "33",
    title: `Dimensionality Reduction`,
    category: `Supervised`,
    description: `Dimensionality Reduction — concise ~100-word description. Learn core ideas, when to use it, and practical tips.`,
    code: `# Example using PCA
from sklearn.decomposition import PCA
import numpy as np
X = np.random.rand(10,5)
pca = PCA(n_components=2).fit(X)
print('components shape:', pca.components_.shape)
`
  },
  {
    id: "34",
    title: `Feature Hashing`,
    category: `Unsupervised`,
    description: `Feature Hashing — concise ~100-word description. Learn core ideas, when to use it, and practical tips.`,
    code: `from sklearn.feature_extraction import FeatureHasher
data = [{'text':'hello world','count':2},{'text':'hello there','count':1}]
h = FeatureHasher(n_features=8, input_type='dict')
print(h.transform(data).toarray())
`
  },
  {
    id: "35",
    title: `Time Series Forecasting`,
    category: `Reinforcement`,
    description: `Time Series Forecasting — concise ~100-word description. Learn core ideas, when to use it, and practical tips.`,
    code: `import numpy as np
from sklearn.linear_model import LinearRegression
# naive example: predict next value as trend
X = np.arange(10).reshape(-1,1)
y = X.ravel()*0.5 + 2
model = LinearRegression().fit(X,y)
print('predict next:', model.predict([[10]])[0])
`
  },
  {
    id: "36",
    title: `ARIMA`,
    category: `Neural`,
    description: `ARIMA — concise ~100-word description. Learn core ideas, when to use it, and practical tips.`,
    code: `# ARIMA/SARIMA/Prophet require statsmodels or prophet packages; run in full Python environment. Here, use simple trend example.
print('Use statsmodels.tsa or prophet in Python environments')
`
  },
  {
    id: "37",
    title: `SARIMA`,
    category: `Supervised`,
    description: `SARIMA — concise ~100-word description. Learn core ideas, when to use it, and practical tips.`,
    code: `# ARIMA/SARIMA/Prophet require statsmodels or prophet packages; run in full Python environment. Here, use simple trend example.
print('Use statsmodels.tsa or prophet in Python environments')
`
  },
  {
    id: "38",
    title: `Prophet Model`,
    category: `Unsupervised`,
    description: `Prophet Model — concise ~100-word description. Learn core ideas, when to use it, and practical tips.`,
    code: `# ARIMA/SARIMA/Prophet require statsmodels or prophet packages; run in full Python environment. Here, use simple trend example.
print('Use statsmodels.tsa or prophet in Python environments')
`
  },
  {
    id: "39",
    title: `Recurrent Neural Networks (RNN)`,
    category: `Reinforcement`,
    description: `Recurrent Neural Networks (RNN) — concise ~100-word description. Learn core ideas, when to use it, and practical tips.`,
    code: `# Deep learning examples typically require TensorFlow or PyTorch which may not be available in Pyodide. In full environments use TensorFlow/PyTorch.
print('See TensorFlow/PyTorch examples in full Python environments')
`
  },
  {
    id: "40",
    title: `LSTM`,
    category: `Neural`,
    description: `LSTM — concise ~100-word description. Learn core ideas, when to use it, and practical tips.`,
    code: `# Deep learning examples typically require TensorFlow or PyTorch which may not be available in Pyodide. In full environments use TensorFlow/PyTorch.
print('See TensorFlow/PyTorch examples in full Python environments')
`
  },
  {
    id: "41",
    title: `GRU`,
    category: `Supervised`,
    description: `GRU — concise ~100-word description. Learn core ideas, when to use it, and practical tips.`,
    code: `# Deep learning examples typically require TensorFlow or PyTorch which may not be available in Pyodide. In full environments use TensorFlow/PyTorch.
print('See TensorFlow/PyTorch examples in full Python environments')
`
  },
  {
    id: "42",
    title: `Convolutional Neural Networks (CNN)`,
    category: `Unsupervised`,
    description: `Convolutional Neural Networks (CNN) — concise ~100-word description. Learn core ideas, when to use it, and practical tips.`,
    code: `# Deep learning examples typically require TensorFlow or PyTorch which may not be available in Pyodide. In full environments use TensorFlow/PyTorch.
print('See TensorFlow/PyTorch examples in full Python environments')
`
  },
  {
    id: "43",
    title: `Transfer Learning`,
    category: `Reinforcement`,
    description: `Transfer Learning — concise ~100-word description. Learn core ideas, when to use it, and practical tips.`,
    code: `# Deep learning examples typically require TensorFlow or PyTorch which may not be available in Pyodide. In full environments use TensorFlow/PyTorch.
print('See TensorFlow/PyTorch examples in full Python environments')
`
  },
  {
    id: "44",
    title: `Data Augmentation`,
    category: `Neural`,
    description: `Data Augmentation — concise ~100-word description. Learn core ideas, when to use it, and practical tips.`,
    code: `print('Data augmentation: e.g., flip, rotate images, add noise to augment datasets')
`
  },
  {
    id: "45",
    title: `Activation Functions`,
    category: `Supervised`,
    description: `Activation Functions — concise ~100-word description. Learn core ideas, when to use it, and practical tips.`,
    code: `import math
def relu(x): return max(0,x)
def sigmoid(x): return 1/(1+math.exp(-x))
print('relu(2)=',relu(2),'sigmoid(2)=',sigmoid(2))
`
  },
  {
    id: "46",
    title: `Backpropagation`,
    category: `Unsupervised`,
    description: `Backpropagation — concise ~100-word description. Learn core ideas, when to use it, and practical tips.`,
    code: `# Simple gradient descent for y = ax + b (least squares)
x = [1,2,3,4]
y = [2,4,6,8]
a,b=0.0,0.0
lr=0.01
for _ in range(1000):
    da = sum(2*(a*xi+b-yi)*xi for xi,yi in zip(x,y))/len(x)
    db = sum(2*(a*xi+b-yi) for xi,yi in zip(x,y))/len(x)
    a -= lr*da; b -= lr*db
print('a,b',a,b)
`
  },
  {
    id: "47",
    title: `Gradient Descent`,
    category: `Reinforcement`,
    description: `Gradient Descent — concise ~100-word description. Learn core ideas, when to use it, and practical tips.`,
    code: `# Simple gradient descent for y = ax + b (least squares)
x = [1,2,3,4]
y = [2,4,6,8]
a,b=0.0,0.0
lr=0.01
for _ in range(1000):
    da = sum(2*(a*xi+b-yi)*xi for xi,yi in zip(x,y))/len(x)
    db = sum(2*(a*xi+b-yi) for xi,yi in zip(x,y))/len(x)
    a -= lr*da; b -= lr*db
print('a,b',a,b)
`
  },
  {
    id: "48",
    title: `Stochastic Gradient Descent`,
    category: `Neural`,
    description: `Stochastic Gradient Descent — concise ~100-word description. Learn core ideas, when to use it, and practical tips.`,
    code: `# Simple gradient descent for y = ax + b (least squares)
x = [1,2,3,4]
y = [2,4,6,8]
a,b=0.0,0.0
lr=0.01
for _ in range(1000):
    da = sum(2*(a*xi+b-yi)*xi for xi,yi in zip(x,y))/len(x)
    db = sum(2*(a*xi+b-yi) for xi,yi in zip(x,y))/len(x)
    a -= lr*da; b -= lr*db
print('a,b',a,b)
`
  },
  {
    id: "49",
    title: `Adam Optimizer`,
    category: `Supervised`,
    description: `Adam Optimizer — concise ~100-word description. Learn core ideas, when to use it, and practical tips.`,
    code: `# Simple gradient descent for y = ax + b (least squares)
x = [1,2,3,4]
y = [2,4,6,8]
a,b=0.0,0.0
lr=0.01
for _ in range(1000):
    da = sum(2*(a*xi+b-yi)*xi for xi,yi in zip(x,y))/len(x)
    db = sum(2*(a*xi+b-yi) for xi,yi in zip(x,y))/len(x)
    a -= lr*da; b -= lr*db
print('a,b',a,b)
`
  },
  {
    id: "50",
    title: `Batch Normalization`,
    category: `Unsupervised`,
    description: `Batch Normalization — concise ~100-word description. Learn core ideas, when to use it, and practical tips.`,
    code: `print('Batch Normalization: conceptual techniques used in neural networks to stabilize and regularize training')
`
  },
  {
    id: "51",
    title: `Dropout`,
    category: `Reinforcement`,
    description: `Dropout — concise ~100-word description. Learn core ideas, when to use it, and practical tips.`,
    code: `print('Dropout: conceptual techniques used in neural networks to stabilize and regularize training')
`
  },
  {
    id: "52",
    title: `Initialization Methods`,
    category: `Neural`,
    description: `Initialization Methods — concise ~100-word description. Learn core ideas, when to use it, and practical tips.`,
    code: `print('Initialization Methods: conceptual techniques used in neural networks to stabilize and regularize training')
`
  },
  {
    id: "53",
    title: `Autoencoders`,
    category: `Supervised`,
    description: `Autoencoders — concise ~100-word description. Learn core ideas, when to use it, and practical tips.`,
    code: `# These models require deep learning frameworks; run full examples with TensorFlow/PyTorch in a full Python environment.
print('Use TensorFlow/PyTorch for autoencoders/VAEs/GANs')
`
  },
  {
    id: "54",
    title: `Variational Autoencoders (VAE)`,
    category: `Unsupervised`,
    description: `Variational Autoencoders (VAE) — concise ~100-word description. Learn core ideas, when to use it, and practical tips.`,
    code: `# These models require deep learning frameworks; run full examples with TensorFlow/PyTorch in a full Python environment.
print('Use TensorFlow/PyTorch for autoencoders/VAEs/GANs')
`
  },
  {
    id: "55",
    title: `Generative Adversarial Networks (GANs)`,
    category: `Reinforcement`,
    description: `Generative Adversarial Networks (GANs) — concise ~100-word description. Learn core ideas, when to use it, and practical tips.`,
    code: `# These models require deep learning frameworks; run full examples with TensorFlow/PyTorch in a full Python environment.
print('Use TensorFlow/PyTorch for autoencoders/VAEs/GANs')
`
  },
  {
    id: "56",
    title: `Reinforcement Learning Basics`,
    category: `Neural`,
    description: `Reinforcement Learning Basics — concise ~100-word description. Learn core ideas, when to use it, and practical tips.`,
    code: `# RL examples often require gym and deep learning libs; use gym and stable-baselines or RL libs in full Python environment.
print('Use OpenAI Gym and RL libs in full Python environment')
`
  },
  {
    id: "57",
    title: `Q-Learning`,
    category: `Supervised`,
    description: `Q-Learning — concise ~100-word description. Learn core ideas, when to use it, and practical tips.`,
    code: `# RL examples often require gym and deep learning libs; use gym and stable-baselines or RL libs in full Python environment.
print('Use OpenAI Gym and RL libs in full Python environment')
`
  },
  {
    id: "58",
    title: `Deep Q-Networks (DQN)`,
    category: `Unsupervised`,
    description: `Deep Q-Networks (DQN) — concise ~100-word description. Learn core ideas, when to use it, and practical tips.`,
    code: `# RL examples often require gym and deep learning libs; use gym and stable-baselines or RL libs in full Python environment.
print('Use OpenAI Gym and RL libs in full Python environment')
`
  },
  {
    id: "59",
    title: `Policy Gradients`,
    category: `Reinforcement`,
    description: `Policy Gradients — concise ~100-word description. Learn core ideas, when to use it, and practical tips.`,
    code: `# RL examples often require gym and deep learning libs; use gym and stable-baselines or RL libs in full Python environment.
print('Use OpenAI Gym and RL libs in full Python environment')
`
  },
  {
    id: "60",
    title: `Actor-Critic Methods`,
    category: `Neural`,
    description: `Actor-Critic Methods — concise ~100-word description. Learn core ideas, when to use it, and practical tips.`,
    code: `# Advanced RL; see stable-baselines3 and gym examples in full Python environment
print('Advanced RL examples require full Python environment')
`
  },
  {
    id: "61",
    title: `PPO (Proximal Policy Optimization)`,
    category: `Supervised`,
    description: `PPO (Proximal Policy Optimization) — concise ~100-word description. Learn core ideas, when to use it, and practical tips.`,
    code: `# Advanced RL; see stable-baselines3 and gym examples in full Python environment
print('Advanced RL examples require full Python environment')
`
  },
  {
    id: "62",
    title: `Model-Based RL`,
    category: `Unsupervised`,
    description: `Model-Based RL — concise ~100-word description. Learn core ideas, when to use it, and practical tips.`,
    code: `# Advanced RL; see stable-baselines3 and gym examples in full Python environment
print('Advanced RL examples require full Python environment')
`
  },
  {
    id: "63",
    title: `Multi-Armed Bandits`,
    category: `Reinforcement`,
    description: `Multi-Armed Bandits — concise ~100-word description. Learn core ideas, when to use it, and practical tips.`,
    code: `import random
arms = [0.2,0.5,0.7]
counts = [0,0,0]
rewards = [0,0,0]
for _ in range(100):
    a = random.randrange(len(arms))
    r = 1 if random.random() < arms[a] else 0
    counts[a]+=1; rewards[a]+=r
print('counts',counts,'rewards',rewards)
`
  },
  {
    id: "64",
    title: `Markov Decision Processes (MDP)`,
    category: `Neural`,
    description: `Markov Decision Processes (MDP) — concise ~100-word description. Learn core ideas, when to use it, and practical tips.`,
    code: `import random
arms = [0.2,0.5,0.7]
counts = [0,0,0]
rewards = [0,0,0]
for _ in range(100):
    a = random.randrange(len(arms))
    r = 1 if random.random() < arms[a] else 0
    counts[a]+=1; rewards[a]+=r
print('counts',counts,'rewards',rewards)
`
  },
  {
    id: "65",
    title: `Reward Shaping`,
    category: `Supervised`,
    description: `Reward Shaping — concise ~100-word description. Learn core ideas, when to use it, and practical tips.`,
    code: `print('Conceptual RL topic: balance exploration and exploitation; use epsilon-greedy strategies')
`
  },
  {
    id: "66",
    title: `Exploration vs Exploitation`,
    category: `Unsupervised`,
    description: `Exploration vs Exploitation — concise ~100-word description. Learn core ideas, when to use it, and practical tips.`,
    code: `print('Conceptual RL topic: balance exploration and exploitation; use epsilon-greedy strategies')
`
  },
  {
    id: "67",
    title: `Monte Carlo Methods`,
    category: `Reinforcement`,
    description: `Monte Carlo Methods — concise ~100-word description. Learn core ideas, when to use it, and practical tips.`,
    code: `import random, math
# estimate pi via Monte Carlo
inside=0
trials=10000
for _ in range(trials):
    x,y = random.random(), random.random()
    if x*x+y*y<=1: inside+=1
print('pi approx:',4*inside/trials)
`
  },
  {
    id: "68",
    title: `Hidden Markov Models`,
    category: `Neural`,
    description: `Hidden Markov Models — concise ~100-word description. Learn core ideas, when to use it, and practical tips.`,
    code: `# HMM/CRF require specific libraries; use hmmlearn or sklearn-crfsuite in full Python environment
print('Use hmmlearn or sklearn-crfsuite in full Python environment')
`
  },
  {
    id: "69",
    title: `Conditional Random Fields`,
    category: `Supervised`,
    description: `Conditional Random Fields — concise ~100-word description. Learn core ideas, when to use it, and practical tips.`,
    code: `# HMM/CRF require specific libraries; use hmmlearn or sklearn-crfsuite in full Python environment
print('Use hmmlearn or sklearn-crfsuite in full Python environment')
`
  },
  {
    id: "70",
    title: `Structured Prediction`,
    category: `Unsupervised`,
    description: `Structured Prediction — concise ~100-word description. Learn core ideas, when to use it, and practical tips.`,
    code: `# HMM/CRF require specific libraries; use hmmlearn or sklearn-crfsuite in full Python environment
print('Use hmmlearn or sklearn-crfsuite in full Python environment')
`
  },
  {
    id: "71",
    title: `Bayesian Networks`,
    category: `Reinforcement`,
    description: `Bayesian Networks — concise ~100-word description. Learn core ideas, when to use it, and practical tips.`,
    code: `# These require specialized libraries (pgmpy, GPy); use in full Python environment
print('Use pgmpy or GPy in full Python environment')
`
  },
  {
    id: "72",
    title: `Probabilistic Graphical Models`,
    category: `Neural`,
    description: `Probabilistic Graphical Models — concise ~100-word description. Learn core ideas, when to use it, and practical tips.`,
    code: `# These require specialized libraries (pgmpy, GPy); use in full Python environment
print('Use pgmpy or GPy in full Python environment')
`
  },
  {
    id: "73",
    title: `Gaussian Processes`,
    category: `Supervised`,
    description: `Gaussian Processes — concise ~100-word description. Learn core ideas, when to use it, and practical tips.`,
    code: `# These require specialized libraries (pgmpy, GPy); use in full Python environment
print('Use pgmpy or GPy in full Python environment')
`
  },
  {
    id: "74",
    title: `Anomaly Detection`,
    category: `Unsupervised`,
    description: `Anomaly Detection — concise ~100-word description. Learn core ideas, when to use it, and practical tips.`,
    code: `from sklearn.ensemble import IsolationForest
X = [[0],[0.1],[0.2],[10],[0.3],[0.4]]
clf = IsolationForest(random_state=0).fit(X)
print('pred:', clf.predict([[0.1],[10]]))  # -1 outlier, 1 inlier
`
  },
  {
    id: "75",
    title: `Outlier Detection`,
    category: `Reinforcement`,
    description: `Outlier Detection — concise ~100-word description. Learn core ideas, when to use it, and practical tips.`,
    code: `from sklearn.ensemble import IsolationForest
X = [[0],[0.1],[0.2],[10],[0.3],[0.4]]
clf = IsolationForest(random_state=0).fit(X)
print('pred:', clf.predict([[0.1],[10]]))  # -1 outlier, 1 inlier
`
  },
  {
    id: "76",
    title: `Similarity Metrics`,
    category: `Neural`,
    description: `Similarity Metrics — concise ~100-word description. Learn core ideas, when to use it, and practical tips.`,
    code: `import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
a = np.array([[1,0,0]])
b = np.array([[0.9,0.1,0]])
print('cosine:', cosine_similarity(a,b)[0][0])
`
  },
  {
    id: "77",
    title: `Cosine Similarity`,
    category: `Supervised`,
    description: `Cosine Similarity — concise ~100-word description. Learn core ideas, when to use it, and practical tips.`,
    code: `import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
a = np.array([[1,0,0]])
b = np.array([[0.9,0.1,0]])
print('cosine:', cosine_similarity(a,b)[0][0])
`
  },
  {
    id: "78",
    title: `Distance Metrics`,
    category: `Unsupervised`,
    description: `Distance Metrics — concise ~100-word description. Learn core ideas, when to use it, and practical tips.`,
    code: `import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
a = np.array([[1,0,0]])
b = np.array([[0.9,0.1,0]])
print('cosine:', cosine_similarity(a,b)[0][0])
`
  },
  {
    id: "79",
    title: `Evaluation Metrics (Precision/Recall)`,
    category: `Reinforcement`,
    description: `Evaluation Metrics (Precision/Recall) — concise ~100-word description. Learn core ideas, when to use it, and practical tips.`,
    code: `from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
y_true = [0,1,1,0,1]
y_pred = [0,1,0,0,1]
print('confusion:', confusion_matrix(y_true,y_pred))
print('precision', precision_score(y_true,y_pred))
print('recall', recall_score(y_true,y_pred))
print('f1', f1_score(y_true,y_pred))
`
  },
  {
    id: "80",
    title: `ROC AUC`,
    category: `Neural`,
    description: `ROC AUC — concise ~100-word description. Learn core ideas, when to use it, and practical tips.`,
    code: `from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
y_true = [0,1,1,0,1]
y_pred = [0,1,0,0,1]
print('confusion:', confusion_matrix(y_true,y_pred))
print('precision', precision_score(y_true,y_pred))
print('recall', recall_score(y_true,y_pred))
print('f1', f1_score(y_true,y_pred))
`
  },
  {
    id: "81",
    title: `Confusion Matrix`,
    category: `Supervised`,
    description: `Confusion Matrix — concise ~100-word description. Learn core ideas, when to use it, and practical tips.`,
    code: `from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
y_true = [0,1,1,0,1]
y_pred = [0,1,0,0,1]
print('confusion:', confusion_matrix(y_true,y_pred))
print('precision', precision_score(y_true,y_pred))
print('recall', recall_score(y_true,y_pred))
print('f1', f1_score(y_true,y_pred))
`
  },
  {
    id: "82",
    title: `F1 Score`,
    category: `Unsupervised`,
    description: `F1 Score — concise ~100-word description. Learn core ideas, when to use it, and practical tips.`,
    code: `from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
y_true = [0,1,1,0,1]
y_pred = [0,1,0,0,1]
print('confusion:', confusion_matrix(y_true,y_pred))
print('precision', precision_score(y_true,y_pred))
print('recall', recall_score(y_true,y_pred))
print('f1', f1_score(y_true,y_pred))
`
  },
  {
    id: "83",
    title: `Precision-Recall Curve`,
    category: `Reinforcement`,
    description: `Precision-Recall Curve — concise ~100-word description. Learn core ideas, when to use it, and practical tips.`,
    code: `from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
y_true = [0,1,1,0,1]
y_pred = [0,1,0,0,1]
print('confusion:', confusion_matrix(y_true,y_pred))
print('precision', precision_score(y_true,y_pred))
print('recall', recall_score(y_true,y_pred))
print('f1', f1_score(y_true,y_pred))
`
  },
  {
    id: "84",
    title: `Calibration`,
    category: `Neural`,
    description: `Calibration — concise ~100-word description. Learn core ideas, when to use it, and practical tips.`,
    code: `print('Calibration: calibrate predicted probabilities with sklearn.calibration.CalibratedClassifierCV')
`
  },
  {
    id: "85",
    title: `Explainable AI (XAI)`,
    category: `Supervised`,
    description: `Explainable AI (XAI) — concise ~100-word description. Learn core ideas, when to use it, and practical tips.`,
    code: `print('Use permutation importance, partial dependence, SHAP/LIME for explainability in full Python environment')
`
  },
  {
    id: "86",
    title: `SHAP`,
    category: `Unsupervised`,
    description: `SHAP — concise ~100-word description. Learn core ideas, when to use it, and practical tips.`,
    code: `# SHAP/LIME require external packages; run in full Python environment to compute explanations
print('Install shap or lime and run examples in full Python environment')
`
  },
  {
    id: "87",
    title: `LIME`,
    category: `Reinforcement`,
    description: `LIME — concise ~100-word description. Learn core ideas, when to use it, and practical tips.`,
    code: `# SHAP/LIME require external packages; run in full Python environment to compute explanations
print('Install shap or lime and run examples in full Python environment')
`
  },
  {
    id: "88",
    title: `Model Interpretability`,
    category: `Neural`,
    description: `Model Interpretability — concise ~100-word description. Learn core ideas, when to use it, and practical tips.`,
    code: `print('Use permutation importance, partial dependence, SHAP/LIME for explainability in full Python environment')
`
  },
  {
    id: "89",
    title: `Fairness in ML`,
    category: `Supervised`,
    description: `Fairness in ML — concise ~100-word description. Learn core ideas, when to use it, and practical tips.`,
    code: `print('Fairness: check for performance disparities across groups; use AIF360 or fairlearn in full environment')
`
  },
  {
    id: "90",
    title: `Model Monitoring`,
    category: `Unsupervised`,
    description: `Model Monitoring — concise ~100-word description. Learn core ideas, when to use it, and practical tips.`,
    code: `print('Model monitoring and drift detection: use statistical tests, data pipelines, and monitoring tools in production')
`
  },
  {
    id: "91",
    title: `Drift Detection`,
    category: `Reinforcement`,
    description: `Drift Detection — concise ~100-word description. Learn core ideas, when to use it, and practical tips.`,
    code: `print('Model monitoring and drift detection: use statistical tests, data pipelines, and monitoring tools in production')
`
  },
  {
    id: "92",
    title: `MLOps Basics`,
    category: `Neural`,
    description: `MLOps Basics — concise ~100-word description. Learn core ideas, when to use it, and practical tips.`,
    code: `print('Model monitoring and drift detection: use statistical tests, data pipelines, and monitoring tools in production')
`
  },
  {
    id: "93",
    title: `Docker for ML`,
    category: `Supervised`,
    description: `Docker for ML — concise ~100-word description. Learn core ideas, when to use it, and practical tips.`,
    code: `print('Docker/CI/CD/Serving: use Dockerfiles, CI pipelines, and frameworks like Seldon, KFServing in production')
`
  },
  {
    id: "94",
    title: `CI/CD for Models`,
    category: `Unsupervised`,
    description: `CI/CD for Models — concise ~100-word description. Learn core ideas, when to use it, and practical tips.`,
    code: `print('Docker/CI/CD/Serving: use Dockerfiles, CI pipelines, and frameworks like Seldon, KFServing in production')
`
  },
  {
    id: "95",
    title: `Model Serving`,
    category: `Reinforcement`,
    description: `Model Serving — concise ~100-word description. Learn core ideas, when to use it, and practical tips.`,
    code: `print('Docker/CI/CD/Serving: use Dockerfiles, CI pipelines, and frameworks like Seldon, KFServing in production')
`
  },
  {
    id: "96",
    title: `Feature Stores`,
    category: `Neural`,
    description: `Feature Stores — concise ~100-word description. Learn core ideas, when to use it, and practical tips.`,
    code: `print('Use Feast, DVC, MLflow for feature stores, data versioning, and experiment tracking in production environments')
`
  },
  {
    id: "97",
    title: `Data Versioning (DVC)`,
    category: `Supervised`,
    description: `Data Versioning (DVC) — concise ~100-word description. Learn core ideas, when to use it, and practical tips.`,
    code: `print('Use Feast, DVC, MLflow for feature stores, data versioning, and experiment tracking in production environments')
`
  },
  {
    id: "98",
    title: `Experiment Tracking (MLflow)`,
    category: `Unsupervised`,
    description: `Experiment Tracking (MLflow) — concise ~100-word description. Learn core ideas, when to use it, and practical tips.`,
    code: `print('Use Feast, DVC, MLflow for feature stores, data versioning, and experiment tracking in production environments')
`
  },
  {
    id: "99",
    title: `Privacy and Federated Learning`,
    category: `Reinforcement`,
    description: `Privacy and Federated Learning — concise ~100-word description. Learn core ideas, when to use it, and practical tips.`,
    code: `print('Privacy and federated learning require specialized frameworks like TensorFlow Federated or PySyft')
`
  },
  {
    id: "100",
    title: `Differential Privacy`,
    category: `Neural`,
    description: `Differential Privacy — concise ~100-word description. Learn core ideas, when to use it, and practical tips.`,
    code: `print('Privacy and federated learning require specialized frameworks like TensorFlow Federated or PySyft')
`
  },
  {
    id: "101",
    title: `Scaling with GPUs`,
    category: `Supervised`,
    description: `Scaling with GPUs — concise ~100-word description. Learn core ideas, when to use it, and practical tips.`,
    code: `print('GPU/TPU and optimization tips: use batching, mixed precision, and efficient data pipelines in full environments')
`
  },
  {
    id: "102",
    title: `TPUs`,
    category: `Unsupervised`,
    description: `TPUs — concise ~100-word description. Learn core ideas, when to use it, and practical tips.`,
    code: `print('GPU/TPU and optimization tips: use batching, mixed precision, and efficient data pipelines in full environments')
`
  },
  {
    id: "103",
    title: `Optimization Tricks`,
    category: `Reinforcement`,
    description: `Optimization Tricks — concise ~100-word description. Learn core ideas, when to use it, and practical tips.`,
    code: `print('GPU/TPU and optimization tips: use batching, mixed precision, and efficient data pipelines in full environments')
`
  },
  {
    id: "104",
    title: `Sparse Models`,
    category: `Neural`,
    description: `Sparse Models — concise ~100-word description. Learn core ideas, when to use it, and practical tips.`,
    code: `print('Model compression techniques like pruning and quantization require ML frameworks and toolkits')
`
  },
  {
    id: "105",
    title: `Compressed Models (Pruning/Quantization)`,
    category: `Supervised`,
    description: `Compressed Models (Pruning/Quantization) — concise ~100-word description. Learn core ideas, when to use it, and practical tips.`,
    code: `print('Model compression techniques like pruning and quantization require ML frameworks and toolkits')
`
  },
  {
    id: "106",
    title: `Graph Neural Networks (GNN)`,
    category: `Unsupervised`,
    description: `Graph Neural Networks (GNN) — concise ~100-word description. Learn core ideas, when to use it, and practical tips.`,
    code: `print('GNNs require libraries like PyTorch Geometric or DGL; run examples in full Python environment')
`
  },
  {
    id: "107",
    title: `Recommendation Systems`,
    category: `Reinforcement`,
    description: `Recommendation Systems — concise ~100-word description. Learn core ideas, when to use it, and practical tips.`,
    code: `import numpy as np
# tiny collaborative filtering via dot-product
user_vecs = np.array([[1,0],[0,1],[1,1]])
item_vecs = np.array([[1,0],[0,1]])
scores = user_vecs.dot(item_vecs.T)
print('scores:\n', scores)
`
  },
  {
    id: "108",
    title: `Collaborative Filtering`,
    category: `Neural`,
    description: `Collaborative Filtering — concise ~100-word description. Learn core ideas, when to use it, and practical tips.`,
    code: `import numpy as np
# tiny collaborative filtering via dot-product
user_vecs = np.array([[1,0],[0,1],[1,1]])
item_vecs = np.array([[1,0],[0,1]])
scores = user_vecs.dot(item_vecs.T)
print('scores:\n', scores)
`
  },
  {
    id: "109",
    title: `Content-Based Filtering`,
    category: `Supervised`,
    description: `Content-Based Filtering — concise ~100-word description. Learn core ideas, when to use it, and practical tips.`,
    code: `import numpy as np
# tiny collaborative filtering via dot-product
user_vecs = np.array([[1,0],[0,1],[1,1]])
item_vecs = np.array([[1,0],[0,1]])
scores = user_vecs.dot(item_vecs.T)
print('scores:\n', scores)
`
  },
  {
    id: "110",
    title: `Search and Information Retrieval`,
    category: `Unsupervised`,
    description: `Search and Information Retrieval — concise ~100-word description. Learn core ideas, when to use it, and practical tips.`,
    code: `print('Search/IR: use TF-IDF, BM25, inverted indexes; use Whoosh or Elasticsearch in production')
`
  },
];
