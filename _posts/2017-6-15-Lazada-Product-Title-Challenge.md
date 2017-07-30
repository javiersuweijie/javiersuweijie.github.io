---
layout: post
title: Notes on Lazada Product Title Challenge
---

Lazada came up with a data science challenge to categorise their product titles. I joined the competition to get exposed to some experience in a data science project. This is my first time entering a competition like that and I'm quite excited to try all the different ways to tackle the problem. This blog post is to consolidate my thoughts about the project and to document my progress for future reference. Here we go.

### Approach
Looking through a few of the submissions on Kaggle, most winning entries use a variation of an ensemble method. I refered some papers to formulate a quick plan on a few models to test on.

### Data Exploration
The advice I received on starting a data science project was to first understand the data set. The following was the given set of data and a few examples of each of the rows.

Columns           | Description
------------      | -------------
country           | The country where the product is marketed, with three possible values: my for Malaysia, ph for Philippines, sg for Singapore
sku_id            | Unique product id, e.g., "NO037FAAA8CLZ2ANMY"
title             | Product title, e.g., "RUDY Dress"
category_lvl_1    | General category that the product belongs to, e.g., "Fashion"
category_lvl_2    | Intermediate category that the product belongs to, e.g., "Women"
category_lvl_3    | Specific category that the product belongs to, e.g., "Clothing"
short_description | Short description of the product, which may contain html formatting, e.g., `<ul> <li>Short Sleeve</li> <li>3 Colours 8 Sizes</li> <li>Dress</li> </ul>`
price             | Price in the local currency, e.g., "33.0".  When country is my, the price is in Malaysian Ringgit.  When country is sg, the price is in Singapore Dollar.  When country is ph, the price is in Philippine Peso.
product_type      | It could have three possible values: local means the product is delivered locally, international means the product is delivered from abroad, NA means not applicable.

Intuitively, it is not clear how some of the columns like price, country or shipping type would affect the clarity or conciseness of the product title. Using sklearn and pandas, I ran a simple logistic regression separately for each of the columns to validate my hypothesis. The dataset was heavy imbalanced with very little unclear titles. (the mean for clarity labels: 0.94, conciseness labels: 0.68) I oversampled the dataset to better balance the classes. 

feature      | clarity  | conciseness
---          | ---      | ---
price        | 0.676383 | 0.680447
title_length | 0.632987 | 0.552437
short_description_length | 0.692496 | 0.700537
country      | 0.692390 | 0.686330
category 1   | 0.479342 | 0.578741
category 2   | 0.490488 | 0.591455
category 3   | 0.473209 | 0.559848


By themselves, these basic features doesn't seem like they are good enough signals to predict neither the clarity nor conciseness of a title. I combined `price`, `title length`, `description length`, `categories 1,2,3` and ran a multi-dimension logistic regression. The result is as follows:

* clarity: 0.473671
* conciseness: 0.484296

Since most of the features cannot really give us information on the labels, we will need to clean the data and use a little more advanced features to capture the structure of the titles as signals. For the cleaning of data, I refered to the following snippet of code. It is simple and works relatively well together with a list of 10000 common used words from Google. I modified it to also resolve words that are joint together by mistake (e.g. blueshort -> blue & short). I realised that a lot of the titles have either model/serial numbers and dimensions. To reduce the size of the feature space in the later models, all words with numerics were replaced with a `#` symbol. 

## Models

### N-grams based Term Frequency with Ridge Regression + Random Forest
After cleaning the data, I used a simple TF based model to vectorise each title. The result is a large sparse matrix that counts the frequency of the n-gram terms in the title. To reduce the size of the matrix, only the top 16th percentile was chosen after using chi-squared test. For char 3-gram, the resulting width of the matrix was reduced from 26374 to 4220.

I tested out with char 3,5,7,9-grams and word 1,3-grams. The following are the respective results.


feature set     | clarity (tf)  | conciseness (tf)  | clarity (tf-idf)  | conciseness (tf-idf) 
------------    | ----------    | ---------         | --------------    | --------
random baseline | 0.324347      | 0.654909          | 0.324347          | 0.654909
char 3gram      | 0.402636      | 0.381939          | 0.410360          | 0.369954
char 5gram      | 0.401437      | 0.366141          | 0.395779          | 0.357437
char 7gram      | 0.414279      | 0.357683          | 0.386608          | 0.349552
char 9gram      | 0.387128      | 0.358778          | 0.376302          | 0.361365
word 1gram      | 0.393545      | 0.383846          | 0.410250          | 0.381326
word 3gram      | 0.390756      | 0.406796          | 0.387831          | 0.414158

It seems like this model did worst on the clarity than just randomly guessing. This is likely to do with having heavily imbalanced training set (90% of the titles were clear). In fact, by just predicting that titles are clear, we can get a score of 0.2379. However, for conciseness, it did way better than randomly guessing. 

Next, I tried to do what most Kagglers do to beat the leaderboard... Stacking! I randomly split the dataset into three parts. A (70%) B (20%) C (10%). 

Firstly, by training a few models on A and predicting labels for B, I fed the B predictions into another Ridge Regression model that is trained on the labels of B. Then finally validating the B predictions with C. The results are promising with rmse for clarity at **0.221783** and conciseness at **0.338390**. Running the pipeline with a cleaned data set gave an improvement to clarity (0.204356) but conciseness (0.338385) did not change much.

With my current instance on linode, K-Nearest Neighbour and SVM both either takes too long or runs out of memory too quickly. I tried to reduce the dimensionality of the sparse matrix with latent semantic analysis. This did not offer much improvement to the overall model. I also tried using the inverse term frequency to weight the frequencies. Clarity scores increased but conciseness decreased.

I ended up training all the feature sets on both ridge regression and random forest, picked the top 8 best performing models to feed into an ensemble. Although the split testing score was good, the score on the validation set was pretty bad.

* clarity: 0.21342
* conciseness: 0.33056

Possible things to explore:
1. Comparing the title with the short description. If description contains something that the title does not have, maybe the title in unclear?
2. Including categorical data as features. Possibily at the ensemble stage using a random forest.

### Long-Short Term Memory Network

The n-gram model works great but it does not take semantics into account. Taking an unconcise title as an example, `Women Canvas Navy Style ID Credit Card Bag Girls Coin Bags Purse (Orange)`, we see multiple mentions of the term `purse` but in different ways like `coin bag`, `card bag`. My hunch is that maybe by using a better representation of words, we can capture these repeating concepts in a title. I downloaded a pre-trained GloVe model and ran it through a LSTM network with just 1 hidden layer. I decided to train both clarity and conciseness at the same thinking that the network will capture the relationships between the two labels.

The results were not spectacular but nevertheless it was close to the ridge regression model above. Changing the hyperparameters did not change the result by a large extent so I stuck with a hidden layer of 50 units. 

* clarity: 0.2130
* conciseness: 0.3555

From the data exploratory phase, I found out that categories play some role in predicting clarity. I decided to incorporate this information into a more complex network. I used keras to generate the following diagram.

![LSTM with categories](https://raw.githubusercontent.com/javiersuweijie/javiersuweijie.github.io/master/images/lstm-2.png)

* clarity: 0.206639
* Conciseness: 0.333616

This model performed the best so far on the validation set. 

Possible things to explore:
1. Compare the performance with a tuned Bi-directional LSTM.
2. Use embeddings with a larger dimension. I wasn't able to try due to a lack of memory.

### Titles Pre-processing

I wanted to test if we can make use of some information by tagging the words in the title to a certain class (e.g. product, model, specifications, brand...), we can then derive more information on how to label them. After a quick research on training a POS tagger, I decided to build one using Conditional Random Fields. The idea is simple, predicting the state (class) of a token (a word in the title) by using the previous and next state together with some observable features like length of token, number of symbols, number of numericals and whether the word belonged to one of the 1000 commonly used words. I quickly hacked up a UI so that I can label the titles daily on my way to work. 

![Correcting predicted tags](https://github.com/javiersuweijie/javiersuweijie.github.io/blob/master/images/tagger.gif?raw=true)

After iteratively labelling 4000 titles (9% of the samples), I used the model to predict the classes the rest of the tokens. With this new feature set, I trained a RandomForest model to test signal strength in predicting the final labels. I counted the number of occurrence of each class and also how similar the product terms are to other product terms. WordNet similarity was used here. The results were a little disappointing.

* clarity: 0.257165
* concisness: 0.490752

Although it did better than randomly guessing, it's not really that much better. Will have to come back to this again in the future as this seems like the best way to advice users on what type of information they are missing. 

Possible things to explore:
1. Try including other features like category and length.
2. Regularise some of the terms such as colours, brands and sizes before training the CRF model.

### Convolution Neural Network

The last model I tried was a simple convolution neural network. I read a few papers on how CNNs worked well for sentence classification problems. Intuitively, due to the effectives of representing locality information, CNNs can capture the relationships of a string of words well. Using a similar approach to the RNN model, I used GloVe to generate the word embeddings. The next layers are 3,4,5 wide filters followed by 1-max-pooling. These configurations were recommended by [this paper](https://arxiv.org/abs/1510.03820) and they do work well. However, instead of using 100 filter maps as recommended, I only used 4 for each filter. I realised that the model tends to overfit with this dataset and reducing the number of parameters actually help to improve the validation scores significantly. 

I also tried to experiment between static and non-static embeddings. Again, likely due to overfitting, the non-static embeddings did slightly worse on the cross validation. The CNN model trains a lot faster than the RNN model and I would definitely try it right from the get go in the future. Lastly, I also decided to train clarity and conciseness on two different models. Because of the smaller number of parameters, the model likely couldn't fit both classes at once. Since the model trains fast, splitting into two wasn't that painful. The following are the validation scores for the CNN model.

* clarity: 0.212603
* conciseness: 0.337787

Possible things to explore:
1. Deeper network with more regularisation/dropouts to reduce overfitting.

### Final Results

At this point in time, I realised I have spent too much time on this competition. I decided to average the two predictions from the LSTM and CNN models for the final submission. Although I did not make to top 10, considering that this was my first "Kaggle" competition I think I learnt quite a lot. Will definitely participate in others after this to improve my intuition when it comes to data and language problems. 

Tools that I used:
1. AWS GPU Compute Instances - Highly recommend for laptop users like myself. Together with a configured image, I can spin up and run experiements in minutes.
2. Jupyter - Also another highly recommended tool when working with AWS instances.
3. sklearn - Enough said.
4. python-crftools - Conditional random field implementation wrapper for python
5. keras with tensorflow - Deep learning in less than 20 lines of code.

Final Rank: 
* clarity:
* conciseness: 
