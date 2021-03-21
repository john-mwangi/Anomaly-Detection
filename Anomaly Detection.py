
# # 0. Data importation from a .mat file

# In[73]:


from scipy.io import loadmat


# In[74]:


raw_data = loadmat(file_name='../Anomaly Detection/shuttle.mat')


# In[5]:


type(raw_data)


# In[16]:


raw_data.keys()


# In[3]:


import pandas as pd
import numpy as np


# In[19]:


raw_data['X']


# In[20]:


raw_data['y']


# In[4]:


data_x = pd.DataFrame(data=raw_data['X'])


# In[6]:


data_x.head()


# In[5]:


data_y = pd.DataFrame(data=raw_data['y'])


# In[ ]:


np.random.seed(0)


# # 1. Box Plots

# ![Box_Plot.png](attachment:Box_Plot.png)

# In[29]:


import matplotlib.pyplot as plt


# In[31]:


data_x[0].plot(kind='box')
plt.show()


# In[36]:


data_x[5].plot(kind='box')
plt.show()


# In[35]:


data_x.plot(kind='box')
plt.show()


# # 2. Histogram

# In[42]:


data_x[0].plot(kind='hist')
plt.show()


# In[43]:


data_x[6].plot(kind='hist')
plt.show()


# In[74]:


# Generically define how many plots along and across
ncols = 3
nrows = int(np.ceil(len(data_x.columns) / ncols))
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 10))

# Lazy counter so we can remove unwated axes
counter = 0
for i in range(nrows):
    for j in range(ncols):

        ax = axes[i][j]

        # Plot when we have data
        if counter < len(data_x.columns):

            ax.hist(data_x[data_x.columns[counter]], bins=10, color='blue', label='{}'.format(data_x.columns[counter]))
            ax.set_xlabel('Values')
            ax.set_ylabel('Frequency')
            #ax.set_ylim([0, 5])
            leg = ax.legend(loc='upper right')
            leg.draw_frame(False)

        # Remove axis when we no longer have data
        else:
            ax.set_axis_off()

        counter += 1

plt.show()


# # 3. Clustering

# ## 3.1 DBSCAN with default parameters

# In[75]:


from sklearn.cluster import DBSCAN


# In[8]:


dbscan = DBSCAN()


# In[9]:


model_db = dbscan.fit(data_x)


# In[12]:


model_db.labels_
#Noisy values are -1 indicating all values are noisy


# In[16]:


# Number of clusters
len(np.unique(model_db.labels_))


# In[17]:


model_db_2 = DBSCAN(eps=3, min_samples=5).fit(data_x)


# In[19]:


np.unique(model_db_2.labels_)


# In[9]:


#Score the model
from sklearn.metrics import silhouette_score


# In[25]:


print('Silhouette Score: %.3f' % silhouette_score(X=data_x, labels=model_db_2.labels_, sample_size=5000))
#Due to an memory error, I set the sample size


# In[27]:


#Select outliers
len(model_db_2.labels_), len(data_x)


# In[35]:


data_x_2 = data_x


# In[28]:


data_x_2['labels_2']=model_db_2.labels_


# In[30]:


data_x_2.head()


# In[34]:


#Find the anomalies
data_x_2.loc[data_x_2['labels_2']==-1].head()


# In[47]:


#data_x.drop(labels='labels_2', axis=1, inplace=True)


# In[11]:


data_x.head()


# ## 3.2 Improve model score using hyper-parameter tuning

# In[76]:


from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV


# In[15]:



from sklearn.metrics import silhouette_score as sc

def cv_silhouette_scorer (estimator, X):
    estimator.fit(X)
    cluster_labels = estimator.labels_
    num_labels = len(set(cluster_labels))
    num_samples = len(X.index)
    
    if num_labels == 1 or num_labels == num_samples:
        return -1
    else:
        return sc(X, cluster_labels, sample_size=100)


# In[16]:


eps = range(1,10,1)
min_samples = range(1,15,1)
params = dict(eps=eps, min_samples=min_samples)
cv = [(slice(None), slice(None))]
estimator = DBSCAN()

ran_search = RandomizedSearchCV(estimator=estimator, param_distributions=params, scoring=cv_silhouette_scorer, n_iter=2)


# In[17]:


ran_search.fit(data_x)


# In[19]:


print('Best score: %.2f, Best params: %s' % (ran_search.best_score_, ran_search.best_params_))


# In[80]:


# Testing on all data
model_db_3 = DBSCAN(eps=6, min_samples=13).fit(data_x)


# In[78]:


from sklearn.metrics import silhouette_score


# In[81]:


silhouette_score(labels=model_db_3.labels_, X=data_x, sample_size=100)
#Significant improvement over default score


# In[168]:


#Number of outliers
outlier_count = list(model_db_3.labels_).count(-1)

print('Outliers: %s items, Proportion: %.3f%%' % (outlier_count, outlier_count/len(model_db_3.labels_)*100))


# In[87]:


print(np.unique(model_db_3.labels_))


# ## 3.3 Visualise the clustering

# In[115]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[116]:


core_samples_mask = np.zeros_like(model_db_3.labels_, dtype=bool)
core_samples_mask[model_db_3.core_sample_indices_] = True
labels = model_db_3.labels_
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)
X = data_x
unique_labels = set(labels)


# In[128]:


# Black removed and is used for noise instead.
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy.values[:, 0], xy.values[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=14)

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy.values[:, 0], xy.values[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

plt.title('Estimated number of clusters: %d' % n_clusters_)
#plt.legend(unique_labels)
plt.show()



# ## 3.4 Comparison against actual

# In[129]:


data_y_db = data_y


# In[131]:


data_y_db['db']=model_db_3.labels_


# In[132]:


data_y_db.head()


# In[152]:


data_y_db['match'] = (data_y_db[0]==1) & (data_y_db['db']==-1)


# In[159]:


data_y_db.head()


# In[167]:


print('Success rate: {:.3f}%'.format(list(data_y_db['match']).count(True)/len(data_y_db)*100))


# # 4. Classification

# In[7]:


# Use kNN classifier then optimise ROC
from sklearn.neighbors import KNeighborsClassifier


# In[11]:


np.random.seed(0)


# In[12]:


from sklearn.model_selection import train_test_split


# In[15]:


train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=0.2)


# In[17]:


train_x.shape, train_y.shape, test_x.shape, test_y.shape


# In[18]:


model_knn = KNeighborsClassifier().fit(train_x, train_y)


# In[9]:


from sklearn.metrics import roc_auc_score


# In[20]:


y_pred = model_knn.predict(X=test_x)


# In[21]:


roc_auc_score(y_true=test_y, y_score=y_pred)


# # 5. Text data
# 
# In this challenge, we want to find tweets that are not health related.

# ## 5.1 Data importation

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


np.random.seed(0)


# In[3]:


def _read_all_health_tweets():
    all_tweets = {}
    file = open('../Anomaly Detection/nytimeshealth.txt', mode='r', encoding='utf8') #opens in read-only mode
    lines = file.readlines()
    for index, line in enumerate(lines):
        parts = line.split(sep='|', maxsplit=2) #maxsplit puts first 2 items in quotes and the rest in a single quote
        tweet = "".join(parts[2:len(parts)]) #join from part 2
        all_tweets[index] = tweet
    
    file.close()
    return all_tweets


# In[4]:


all_tweets = _read_all_health_tweets()


# In[5]:


type(all_tweets)


# In[6]:


all_tweets


# In[7]:


all_tweets = pd.DataFrame.from_dict(data=all_tweets, orient='index')


# In[8]:


all_tweets.head()


# In[9]:


len(all_tweets)


# In[10]:


all_tweets.rename(columns={0:'Tweet'}, inplace=True)


# Our data contains 6245 tweets. The question is: can we identify tweets that are not related to Health News? Or more generally: can we identify top n outlier tweets?

# ## 5.2 Text pre-processing & vector generation

# There are different vector generation techniques: word2vec, doc2vec, tf-idf, etc. We will use doc2vec because each tweet can be considered a document (collection of words) which works well in providing context to words as opposed to frequency based vectorisation such as tf-idf.

# In[11]:


from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from sklearn import utils
from gensim.models.doc2vec import TaggedDocument, Doc2Vec #Generating vectors
from gensim.parsing.preprocessing import preprocess_string #NLP preprocessing
from tqdm import tqdm #For displaying a progress meter


# In[12]:


tweets_dict = _read_all_health_tweets()
tweets = tweets_dict.values()


# ### Fit-transform: Option 1 (Official approach)

# In[14]:


#First try the simpler official implementation
tagged_x = [TaggedDocument(preprocess_string(item), [index]) for index, item in enumerate(tweets)]
#TaggedDocument(words, tags) is a list of tagged documents where a tag is an index of a document (tweets in this case).
#NB: we only pass the values


# In[15]:


type(tagged_x)


# In[16]:


#Fit the model
model = Doc2Vec(documents=tagged_x, vector_size=100, window=2, min_count=1, workers=4)
#vector_size=dimension of the vector
#window=max distance between current and predicted word in a sentence
#min_count=ignore words with count/occurence less than this
#workers=more is better for faster training


# In[17]:


len(model.docvecs),len(model.docvecs[0])


# In[18]:


#Tranform
arr_2 = np.array([model.infer_vector(preprocess_string(item)) for index, item in enumerate(tweets)])


# In[19]:


arr_2.shape
#Each row is a tweet and in each tweet, the top 100 features are compared to that tweet


# In[20]:


arr_2


# ### Fit-transform: Option 2 (Using a neural network)

# In[52]:


#Using a neural network for training
class Doc2VecTransformer(BaseEstimator):

    def __init__(self, vector_size=100, learning_rate=0.02, epochs=20):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self._model = None
        self.vector_size = vector_size
        self.workers = 4

    def fit(self, x, y=None):
        tagged_x = [TaggedDocument(preprocess_string(item), [index]) for index, item in enumerate(x)]
        model = Doc2Vec(documents=tagged_x, vector_size=self.vector_size, workers=self.workers)

        for epoch in range(self.epochs):
            model.train(utils.shuffle([x for x in tqdm(tagged_x)]), total_examples=len(tagged_x), epochs=1)
            model.alpha -= self.learning_rate
            model.min_alpha = model.alpha

        self._model = model
        return self

    def transform(self, x):
        arr = np.array([self._model.infer_vector(preprocess_string(item))
                                     for index, item in enumerate(x)])
        return arr


# In[53]:


tweets_dict = _read_all_health_tweets()
tweets = tweets_dict.values()
pl = Pipeline(steps=[('doc2vec', Doc2VecTransformer())])
vectors_df = pl.fit(tweets).transform(tweets)
vectors_df


# In[55]:


vectors_df.shape


# ## 5.3 Feature reduction
# It is recommended to keep the Doc2Vec features to 100-300 then perform PCA to extract the most important features.

# In[21]:


from sklearn.decomposition import PCA


# In[22]:


def analyse_tweets_pca(n_pca_components, doc2vectors):
    pca = PCA(n_components=n_pca_components) #initialise
    pca_vectors = pca.fit_transform(doc2vectors)
    print('All principal components...')
    print(pca_vectors)
    for index, var in enumerate(pca.explained_variance_ratio_):
        print('Variance explained by principal component ',(index+1),'is ',var)


# In[23]:


analyse_tweets_pca(n_pca_components=10, doc2vectors=arr_2)

# ### 5.4.1 Determining optimal number of clusters

# In[24]:


from sklearn.base import BaseEstimator 
from sklearn.metrics.cluster.unsupervised import silhouette_samples, silhouette_score
from nltk.cluster import KMeansClusterer, cosine_distance 


# In[ ]:




# In[25]:


pca = PCA(n_components=2)
pca_vectors = pca.fit_transform(arr_2)


# In[26]:


#Cluster into 2 clusters
clusterer = KMeansClusterer(num_means=2, distance=cosine_distance, repeats=3)
cluster_labels = clusterer.cluster(vectors=pca_vectors, assign_clusters=True, trace=False)


# In[31]:


silhouette_score(X=pca_vectors, labels=cluster_labels, metric='cosine')


# In[34]:


#we create a function that calculates silhouette scores for a range of k then select the most optimal k
def _determine_k_for_max_silhouette_score_ (min_k, max_k, pca_vectors):
    max_silhouette_score = -1.0
    optimal_k = 2
    for current_k in range(min_k, max_k+1):
        clusterer = KMeansClusterer(num_means=current_k, distance=cosine_distance, repeats=3)
        cluster_labels = clusterer.cluster(vectors=pca_vectors, assign_clusters=True, trace=False)
        silhouette_score_k = silhouette_score(X=pca_vectors, labels=cluster_labels, metric='cosine')
        print('Silhouette score for ',current_k,' clusters (k) is ',silhouette_score_k)
        
        if silhouette_score_k > max_silhouette_score:
            max_silhouette_score = silhouette_score_k
            optimal_k = current_k
            
    return optimal_k, max_silhouette_score


# In[35]:


_determine_k_for_max_silhouette_score_(min_k=2, max_k=5, pca_vectors=pca_vectors)



# ### 5.4.2 Determining top n anomalies

# In[46]:


#We first group our data points into 2 clusters by labelling
clusterer = KMeansClusterer(num_means=2, distance=cosine_distance, repeats=3)
cluster_labels = clusterer.cluster(vectors=pca_vectors, assign_clusters=True, trace=False)


# In[49]:


def determine_anomaly_tweets_kmeans(top_n, pca_vectors, cluster_labels):
#NOTE:
#pca_vectors = PCA vectors of the top 2 PCA components only
#cluster_labels = labels of 2 clusters only

    silhouette_values = silhouette_samples(X=pca_vectors, labels=cluster_labels, metric='cosine')
#My understanding of silhouette_samples() is that it returns the silhouette score of an individual data point in a cluster
    tweet_index_silhouette_scores = [] #this will hold the tweet index & its actual sh_score
    absolute_silhouette_scores_tweet_index = [] #this will hold the tweet index & its absolute sh_score
    
    for index, sh_score in enumerate(silhouette_values):
        absolute_silhouette_scores_tweet_index.append((abs(sh_score), index))
#We'll need to sort the tweet indices and their sh_scores by the abs sh_score hence the abs sh_score comes first in the tuple
        tweet_index_silhouette_scores.append((index, sh_score))
#We'll need to retrieve tweet indices and their actual sh_score hence index comes first        
    sorted_scores = sorted(absolute_silhouette_scores_tweet_index)
#sorted() is for use in lists/tuples/strings (they're similar) when not using data frame columns
    
    top_n_silhouette_scores = []
    pca_vectors_anomalies = []
    print('Top ', top_n, ' anomalies:')
    for i in range(top_n):
        abs_sh_score, index = sorted_scores[i] #get top n of sorted absolute silhouette score i.e. those approx 0
        index_1, sh_score = tweet_index_silhouette_scores[index] #get actual sh_score of top n sorted abs
        #top_n_silhouette_scores.append((index, sh_score)) #Results are the same with this commented out. Maybe it's for plotting?
        print(tweets_dict[index]) #print actual tweet referenced by the index value
        print('PCA Vector: ',pca_vectors[index]) #print the tweet's PCA vectors
        #pca_vectors_anomalies.append(pca_vectors[index]) #Results are the same with this commented out. Maybe it's for plotting?
        print('Silhouette score: ',sh_score)
        print('.................')


# In[50]:


determine_anomaly_tweets_kmeans(5, pca_vectors, cluster_labels)


# ## 5.5 Local Outlier Factor (proximity) based approach
# This can be compared to DBSCAN since it is proximity-based. The greater the LOF the more outlier the item. Steps:
# 1. Import data
# 2. Convert to doc2vectors
# 3. Extract top 2 PCA vectors
# 4. Instantiate LOF transformer with cosine as metric and extract decision scores
# 5. Extract top 5 outliers

# In[2]:


from sklearn.decomposition import PCA
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from gensim.parsing.preprocessing import preprocess_string
from pyod.models.lof import LOF
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np


# In[3]:


np.random.seed(0)


# ### 5.5.1 Import data and return top 2 PCA vectors and tweets dictionary

# In[18]:


# create a function that imports that Twitter data and extracts top 2 components
def extract_2pca():
    all_tweets = {}
    file = open('../Anomaly Detection/nytimeshealth.txt', mode='r', encoding='utf8') #opens in read-only mode
    lines = file.readlines()
    for index, line in enumerate(lines):
        parts = line.split(sep='|', maxsplit=2) #maxsplit puts first 2 items in quotes and the rest in a single quote
        tweet = "".join(parts[2:len(parts)]) #join from part 2
        all_tweets[index] = tweet

    file.close()

    tweets_dict = all_tweets  
    tweets = tweets_dict.values() 
    all_tweets = pd.DataFrame.from_dict(data=all_tweets, orient='index')
    all_tweets.rename(columns={0:'Tweet'}, inplace=True)  

    tagged_x = [TaggedDocument(preprocess_string(item), [index]) for index, item in enumerate(tweets)]
    model = Doc2Vec(documents=tagged_x, vector_size=100, window=2, min_count=1, workers=4)
    arr_2 = np.array([model.infer_vector(preprocess_string(item)) for index, item in enumerate(tweets)])
   
    pca = PCA(n_components=2)
    pca_vectors = pca.fit_transform(arr_2)
    
    return pca_vectors, tweets_dict


# In[19]:


pca_vectors, tweets_dict = extract_2pca()


# In[22]:


pca_vectors.shape


# ### 5.5.2 Implement LOF model

# In[13]:


#Implement LOF model, extract decision scores
lof = LOF(metric='cosine') #cosine is good for measuring non-numeric distances
lof_model = lof.fit(pca_vectors)
scores = lof_model.decision_scores_


# In[32]:


max(scores)


# ### 5.5.3 Implement function to extract top 5 outliers

# In[35]:


top_n = 5
tweet_index_decision_scores = []
decision_scores_tweet_index = []

for index, score in enumerate(scores):
    tweet_index_decision_scores.append((index,score)) #store as a tuple
    decision_scores_tweet_index.append((score,index))
    
sorted_scores = sorted(decision_scores_tweet_index, reverse=True)

top_n_tweet_index_decision_scores = []
print("Top ", top_n, " anomalies")
for i in range(top_n):
    score, index = sorted_scores[i]
    top_n_tweet_index_decision_scores.append((index,score))
    print(tweets_dict[index])
    print('Decision score: ',score)
    print('.............')


# ### 5.5.4 Visualise the outliers

# In[36]:


def plot_scatter_lof(tweets_dict, tweet_index_decision_scores, top_n_tweet_index_decision_scores):
    plt.close('all')
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True)
    fig.suptitle('Decision scores vs Tweets')

    sub_plot_scatter_lof(ax=ax1, tweets_dict=tweets_dict,
                         tweet_index_decision_scores=tweet_index_decision_scores,
                         top_n_tweet_index_decision_scores=top_n_tweet_index_decision_scores, with_annotation=False)
    sub_plot_scatter_lof(ax=ax2, tweets_dict=tweets_dict,
                         tweet_index_decision_scores=tweet_index_decision_scores,
                         top_n_tweet_index_decision_scores=top_n_tweet_index_decision_scores, with_annotation=True)
    plt.show()


# In[40]:


def sub_plot_scatter_lof(ax, tweets_dict, tweet_index_decision_scores, top_n_tweet_index_decision_scores,
                     with_annotation=True):
    ax.set(xlabel='Tweet Index', ylabel='Decision Score')
    ax.scatter(*zip(*tweet_index_decision_scores))
    ax.scatter(*zip(*top_n_tweet_index_decision_scores), edgecolors='red')

    if with_annotation:
        for (index, score) in top_n_tweet_index_decision_scores:
            ax.annotate(tweets_dict[index], xy=(index, score), xycoords='data')


# In[41]:


plot_scatter_lof(tweets_dict, tweet_index_decision_scores, top_n_tweet_index_decision_scores)



# ![Anomalies2.png](attachment:Anomalies2.png)

# ## 6.1 Data preparation

# In[ ]:





# ## 6.1 Isolation Forest

# In[ ]:




