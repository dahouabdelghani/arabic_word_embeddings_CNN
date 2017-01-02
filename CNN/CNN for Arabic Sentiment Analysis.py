# -*- coding: utf-8 -*-

# Word Embeddings and Convolutional Neural Network for Arabic Sentiment Classification (CoLing 2016)
# Dahou, A., Xiong, S., Zhou, J., Haddoud, M. H., & Duan, P. Word Embeddings and Convolutional Neural Network for Arabic Sentiment Classification.

import os
os.environ['THEANO_FLAGS']='mode=FAST_RUN,device=gpu,floatX=float32'

# Parameters matplotlib
# ==================================================
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10,10) 
#%matplotlib inline
import seaborn as sns
plt.switch_backend('agg') 

# Parameters General
# ==================================================
import codecs
import csv
import keras
import sklearn
import gensim
import random
import scipy
import pydot
import commands
import glob
import numpy as np
import pandas as pd

# Parameters theano
# ==================================================
import theano
print 'using : ',(theano.config.device)

# Parameters keras
# ==================================================
from keras.utils.visualize_util import plot ## to print the model arch
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential, Graph
from keras.optimizers import SGD
from keras.layers.embeddings import Embedding
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.regularizers import l2, activity_l2
from keras.layers.core import Dense , Dropout , Activation , Merge , Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers import Embedding , LSTM

# Parameters sklearn
# ==================================================
import sklearn.metrics
from sklearn import preprocessing
from sklearn.base import BaseEstimator
from sklearn.svm import LinearSVC , SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report #get F1 score

# Parameters gensim
# ==================================================
from gensim.models.word2vec import Word2Vec
from gensim.models.doc2vec import Doc2Vec , TaggedDocument
from gensim.corpora.dictionary import Dictionary









# Model Hyperparameters
# ==================================================
#
print "Model Hyperparameters :"

embeddings_dim = 300
print "embeddings_dim = " ,embeddings_dim

filter_sizes = [3, 5, 7]
print "filter_sizes = ",filter_sizes

dropout_prob = [0.5,0.5]
print "dropout_prob = ",dropout_prob

# maximum number of words to consider in the representations
max_features = 30000
print "max_features = " ,max_features

# percentage of the data used for model training
val_split = 0.75
print "val_split = ",val_split


# number of classes
num_classes = 2
print "num_classes = " ,num_classes


# Training parameters
# ==================================================
#
num_epochs = 30
print "num_epochs = ",num_epochs


    
# Reading pre-trained word embeddings
# ==================================================
#   
print ("")
print ("Reading pre-trained word embeddings...")
embeddings = dict( )
embeddings = Word2Vec.load_word2vec_format("../Arabic_WE_model/300features_1minwords_5context", binary=True,encoding='utf8', unicode_errors='ignore')


  
def NN_nets_r0(embeddings,file_name,max_sent_len,Vectors,Linked,Round):

  # maximum length of a sentence
  print "max_sent_len = " ,max_sent_len


  # Reading csv data
  # ==================================================
  #
  print ("Reading text data for classification and building representations...")
  data = []
  data = [ ( row["text"] , row["polarity"]  ) for row in csv.DictReader(open(file_name), delimiter=',', quoting=csv.QUOTE_NONE) ]

  # Data Preparatopn
  # ==================================================
  #
  random.shuffle( data )
  train_size = int(len(data) * val_split)
  train_texts = [ txt for ( txt, label ) in data[0:train_size] ]
  test_texts = [ txt for ( txt, label ) in data[train_size:-1] ]
  train_labels = [ label for ( txt , label ) in data[0:train_size] ]
  test_labels = [ label for ( txt , label ) in data[train_size:-1] ]
  num_classes = len( set( train_labels + test_labels ) )
  tokenizer = Tokenizer(nb_words=max_features, filters=keras.preprocessing.text.base_filter(), split=" ")
  tokenizer.fit_on_texts(train_texts)
  train_sequences = sequence.pad_sequences( tokenizer.texts_to_sequences( train_texts ) , maxlen=max_sent_len )
  test_sequences = sequence.pad_sequences( tokenizer.texts_to_sequences( test_texts ) , maxlen=max_sent_len )
  train_matrix = tokenizer.texts_to_matrix( train_texts )
  test_matrix = tokenizer.texts_to_matrix( test_texts )
  embedding_weights = np.zeros( ( max_features , embeddings_dim ) )
  for word,index in tokenizer.word_index.items():
    if index < max_features:
      try: embedding_weights[index,:] = embeddings[word]
      except: embedding_weights[index,:] = np.random.uniform(-0.25,0.25,embeddings_dim)
  le = preprocessing.LabelEncoder( )
  le.fit( train_labels + test_labels )
  train_labels = le.transform( train_labels )
  test_labels = le.transform( test_labels )
  print "Classes : " + repr( le.classes_ )


  
  # CNN
  # ===============================================================================================
  #
  print ("Method = CNN for Arabic Sentiment Analysis'")
  model_variation = 'CNN-non-static'
  np.random.seed(0)
  nb_filter = embeddings_dim
  model = Graph()
  model.add_input(name='input', input_shape=(max_sent_len,), dtype=int)
  model.add_node(Embedding(max_features, embeddings_dim, input_length=max_sent_len, mask_zero=False, weights=[embedding_weights] ), name='embedding', input='input')
  model.add_node(Dropout(dropout_prob[0]), name='dropout_embedding', input='embedding')
  for n_gram in filter_sizes:
      model.add_node(Convolution1D(nb_filter=nb_filter, filter_length=n_gram, border_mode='valid', activation='relu', subsample_length=1, input_dim=embeddings_dim, input_length=max_sent_len), name='conv_' + str(n_gram), input='dropout_embedding')
      model.add_node(MaxPooling1D(pool_length=max_sent_len - n_gram + 1), name='maxpool_' + str(n_gram), input='conv_' + str(n_gram))
      model.add_node(Flatten(), name='flat_' + str(n_gram), input='maxpool_' + str(n_gram))
  model.add_node(Dropout(dropout_prob[1]), name='dropout', inputs=['flat_' + str(n) for n in filter_sizes])
  model.add_node(Dense(1, input_dim=nb_filter * len(filter_sizes)), name='dense', input='dropout')
  model.add_node(Activation('sigmoid'), name='sigmoid', input='dense')
  model.add_output(name='output', input='sigmoid')

  # Print model summary
  # ==================================================
  # 
  print(model.summary())

  # model compilation
  # ==================================================
  # 
  if num_classes == 2: model.compile(loss={'output': 'binary_crossentropy'}, optimizer='Adagrad') 
  else: model.compile(loss={'output': 'categorical_crossentropy'}, optimizer='Adagrad')

  #  model early_stopping and checkpointer
  # ==================================================
  # 
  early_stopping = EarlyStopping(patience=20, verbose=1)
  checkpointer = ModelCheckpoint(filepath= '../Weights/'+os.path.basename(file_name)+'_'+Round+'_'+Linked+'_'+model_variation+'_'+Vectors+'_weights_Ar_best.hdf5', verbose=1, save_best_only=False)

  #  model history
  # ==================================================
  # 
  hist = model.fit({'input': train_sequences, 'output': train_labels}, batch_size=32, nb_epoch=30, verbose=2, show_accuracy=True,callbacks=[early_stopping, checkpointer])


  # Evaluate model keras
  # ==================================================
  #
  print("Evaluate...")
  score, acc = model.evaluate({'input': test_sequences, 'output': test_labels},show_accuracy=True,batch_size=32)
  print('Test score:', score)
  print('Test accuracy:', acc)


  # Evaluate model sklearn
  # ==================================================
  #
  results = np.array(model.predict({'input': test_sequences}, batch_size=32)['output'])
  if num_classes != 2: results = results.argmax(axis=-1)
  else: results = (results > 0.5).astype('int32')
  print ("Accuracy-sklearn = " + repr( sklearn.metrics.accuracy_score( test_labels , results )  ))
  print 'RMSE',np.sqrt(((results - test_labels) ** 2).mean(axis=0)).mean()  # Printing RMSE
  print (sklearn.metrics.classification_report( test_labels , results ))




# Notations
# ==================================================
#
Vectors = 'Mine-vec'
Linked = 'Not-Linked'

"""
# List of datasets balanced
# ==================================================
#
list_of_files = []
files_extension = 'csv'
list_of_files = glob.glob('../Datasets/data_csv_balanced/*.'+files_extension) 
print '\n List of datasets: '
print list_of_files
print '\n'
rounds=8 #xrange count rounds-1 and start from 0, to start form 1 write xrange(1,rounds)
for i in xrange(0,rounds):
	
	Round='round-'+str(i)

	print "\n||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
	print '\n processing DATEsets For '+ Round
	print "\n||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
	for file_name in list_of_files:
	  



	  if file_name == '../Datasets/data_csv_balanced/LABR-balanced-not-linked-800.csv':

	    
	    print "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
	    print '\n processing (REVIEWS) ... '+file_name+' '+Round
	    print "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"


	    max_sent_len = 767
	    globals()['NN_nets_r'+str(i)](embeddings,file_name,max_sent_len,Vectors,Linked,Round)

	    
	  elif file_name== '../Datasets/data_csv_balanced/Tweets_ieee-balanced-not-linked.csv':
	    print "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
	    print '\n processing  (TWEETER)... '+file_name+' '+Round
	    print "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"

	    max_sent_len = 174
	    globals()['NN_nets_r'+str(i)](embeddings,file_name,max_sent_len,Vectors,Linked,Round)
	    
	  elif file_name== '../Datasets/data_csv_balanced/ASTD-balanced-not-linked.csv':
	    print "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
	    print '\n processing (TWEETER) ... '+file_name+' '+Round
	    print "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"

	    max_sent_len = 40
	    globals()['NN_nets_r'+str(i)](embeddings,file_name,max_sent_len,Vectors,Linked,Round)
	    
	  elif file_name== '../Datasets/data_csv_balanced/GS-LREC-balanced-not-linked.csv':
	    print "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
	    print '\n processing (TWEETER) ... '+file_name+' '+Round
	    print "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"

	    max_sent_len = 40
	    globals()['NN_nets_r'+str(i)](embeddings,file_name,max_sent_len,Vectors,Linked,Round)
	    
	  elif file_name== '../Datasets/data_csv_balanced/MOV-balanced-not-linked.csv':
	    print "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
	    print '\n processing (REVIEWS)  ... '+file_name+' '+Round
	    print "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"

	    max_sent_len = 2335
	    globals()['NN_nets_r'+str(i)](embeddings,file_name,max_sent_len,Vectors,Linked,Round)
	    
	  elif file_name == '../Datasets/data_csv_balanced/RES-balanced-not-linked.csv':
	    print "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
	    print '\n processing (REVIEWS)  ... '+file_name+' '+Round
	    print "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"

	    max_sent_len = 539
	    globals()['NN_nets_r'+str(i)](embeddings,file_name,max_sent_len,Vectors,Linked,Round)
	    
	  elif  file_name== '../Datasets/data_csv_balanced/PROD-balanced-not-linked.csv':
	    print "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
	    print '\n processing (REVIEWS)  ... '+file_name+' '+Round
	    print "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"

	    max_sent_len = 105
	    globals()['NN_nets_r'+str(i)](embeddings,file_name,max_sent_len,Vectors,Linked,Round)
	    
	  elif  file_name== '../Datasets/data_csv_balanced/HTL-balanced-not-linked.csv':
	    print "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
	    print '\n processing (REVIEWS)  ... '+file_name+' '+Round
	    print "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"

	    max_sent_len = 1066
	    globals()['NN_nets_r'+str(i)](embeddings,file_name,max_sent_len,Vectors,Linked,Round)
	    
	  elif  file_name== '../Datasets/data_csv_balanced/ATT-balanced-not-linked.csv':
	    print "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
	    print '\n processing (REVIEWS)  ... '+file_name+' '+Round
	    print "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"

	    max_sent_len = 245
	    globals()['NN_nets_r'+str(i)](embeddings,file_name,max_sent_len,Vectors,Linked,Round)

"""
		
# List of datasets unbalanced
# ==================================================
#

list_of_files = []
files_extension = 'csv'
list_of_files = glob.glob('../Datasets/data_csv_unbalanced/*.'+files_extension) 
print '\n List of datasets: '
print list_of_files
print '\n'
rounds=1 #xrange count rounds-1 and start from 0, to start form 1 write xrange(1,rounds)


for i in xrange(0,rounds):
	
	Round='round-'+str(i)

	print "\n||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
	print '\n processing DATEsets For '+ Round
	print "\n||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
	for file_name in list_of_files:
	  



	  if file_name == '../Datasets/data_csv_unbalanced/LABR-unbalanced-not-linked-800.csv':

	    
	    print "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
	    print '\n processing (REVIEWS) ... '+file_name+' '+Round
	    print "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"


	    max_sent_len = 882
	    globals()['NN_nets_r'+str(i)](embeddings,file_name,max_sent_len,Vectors,Linked,Round)

	    
	  elif file_name== '../Datasets/data_csv_unbalanced/Tweets_ieee-unbalanced-not-linked.csv':
	    print "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
	    print '\n processing  (TWEETER)... '+file_name+' '+Round
	    print "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"

	    max_sent_len = 174
	    globals()['NN_nets_r'+str(i)](embeddings,file_name,max_sent_len,Vectors,Linked,Round)
	    
	  elif file_name== '../Datasets/data_csv_unbalanced/ASTD-unbalanced-not-linked.csv':
	    print "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
	    print '\n processing (TWEETER) ... '+file_name+' '+Round
	    print "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"

	    max_sent_len = 30
	    globals()['NN_nets_r'+str(i)](embeddings,file_name,max_sent_len,Vectors,Linked,Round)
	    
	  elif file_name== '../Datasets/data_csv_unbalanced/GS-LREC-unbalanced-not-linked.csv':
	    print "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
	    print '\n processing (TWEETER) ... '+file_name+' '+Round
	    print "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"

	    max_sent_len = 40
	    globals()['NN_nets_r'+str(i)](embeddings,file_name,max_sent_len,Vectors,Linked,Round)
	    
	  elif file_name== '../Datasets/data_csv_unbalanced/MOV-unbalanced-not-linked.csv':
	    print "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
	    print '\n processing (REVIEWS)  ... '+file_name+' '+Round
	    print "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"

	    max_sent_len = 2335
	    globals()['NN_nets_r'+str(i)](embeddings,file_name,max_sent_len,Vectors,Linked,Round)
	    
	  elif file_name == '../Datasets/data_csv_unbalanced/RES-unbalanced-not-linked.csv':
	    print "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
	    print '\n processing (REVIEWS)  ... '+file_name+' '+Round
	    print "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"

	    max_sent_len = 539
	    globals()['NN_nets_r'+str(i)](embeddings,file_name,max_sent_len,Vectors,Linked,Round)
	    
	  elif  file_name== '../Datasets/data_csv_unbalanced/PROD-unbalanced-not-linked.csv':
	    print "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
	    print '\n processing (REVIEWS)  ... '+file_name+' '+Round
	    print "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"

	    max_sent_len = 234
	    globals()['NN_nets_r'+str(i)](embeddings,file_name,max_sent_len,Vectors,Linked,Round)
	    
	  elif  file_name== '../Datasets/data_csv_unbalanced/HTL-unbalanced-not-linked.csv':
	    print "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
	    print '\n processing (REVIEWS)  ... '+file_name+' '+Round
	    print "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"

	    max_sent_len = 1110
	    globals()['NN_nets_r'+str(i)](embeddings,file_name,max_sent_len,Vectors,Linked,Round)
	    
	  elif  file_name== '../Datasets/data_csv_unbalanced/ATT-unbalanced-not-linked.csv':
	    print "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
	    print '\n processing (REVIEWS)  ... '+file_name+' '+Round
	    print "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"

	    max_sent_len = 568
	    globals()['NN_nets_r'+str(i)](embeddings,file_name,max_sent_len,Vectors,Linked,Round)
