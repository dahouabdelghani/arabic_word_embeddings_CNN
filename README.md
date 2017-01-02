Word Embeddings and Convolutional Neural Network for Arabic Sentiment Classification (CoLing 2016)
--------------

In this paper, a scheme of Arabic sentiment classification, which evaluates and detects the sentiment polarity from Arabic
reviews and Arabic social media, is studied.  We investigated in several architectures to build a quality neural word embeddings using a 3.4 billion words corpus from a collected 10 billion words web-crawled corpus.   Moreover,  a convolutional neural network trained on top of pre-trained Arabic word embeddings is used for sentiment classification to evaluate the quality of these word embeddings.

Paper: https://www.aclweb.org/anthology/C/C16/C16-1228.pdf

Project Layout
--------------

Subdirectories:

- Arabic_WE_eval - Arabic Word Embeddings models evaluation using Arabic word analogies
- Arabic_WE_model - Arabic Word Embeddings models
- CNN - Convolutional Neural Network to train and evalute Arabic sentiment classification task
- Weights - Best weights from training will be saved in this folder
- Datasets - Datasets for training and evaluation
	- data_csv_balanced
	- data_csv_unbalanced

Requirements
--------------
- Keras 0.3.3
- Theano
- Cuda

Script is running on GPU


Output example on ASTD unbalanced dataset
----------------------------------------------------------------------

                    precision    recall  f1-score   support

                 0       0.82      0.85      0.83       362
                 1       0.66      0.61      0.64       178
        avg / total      0.77      0.77      0.77       540



Citation
------------------
Dahou, A., Xiong, S., Zhou, J., Haddoud, M. H., & Duan, P. Word Embeddings and Convolutional Neural Network for Arabic Sentiment Classification.

