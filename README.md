# Isis tweet detection - A new approach by contextualized sentence representation
Setup: 
install requirements.txt 

First, precompute embeddings by setting 
pre_compute_embeddings = True 

For running the prediction process, based on the precomputed embeddings, set: pre_compute_embeddings = False

Dataset: https://www.kaggle.com/datasets/fifthtribe/how-isis-uses-twitter

## Classifying ISIS tweets as pro-Isis 

A few approaches could be considered: 
 - Probabilistic model (GMM for instance) based on the clustered embeddings representation (created by SBERT) of each tweet for each class and assessing the hypothesis that a tweet belongs to one of the probabilistic models P(t/model_isis) > P(t/model_non_isis).
 - Transfer learning for downstream task of a semantic sentence similarity. This semantically sentence similarity-based model creates representation in high dimension feeding a nonlinear binary classifier (MLP), based on the SBERT embeddings, for instance. SBERT can be a good candidate since it was trained contrastively to maintain similarity between sentences with “similar meaning”. Since the embeddings aren’t necessarily linearly separable in the latent space, an additional MLP-based classifier containing non-linear elements is added to transfer the learning to the downstream task. 
  - Generative-based approach and retrieval one, are also an option
Approach No.2 was adopted here since it is fully trainable using the SBERT model as a pre-trained backbone. 

A textual similarity model such as SBERT (Bi-Encoder) was trained to get maximal cosine similarity for semantic similar sentences and also used for clustering purposes. 
Disclaimer: many SBERT models are under various challenges such as MTEB one. Hence picking the right model is also an issue to be further optimized.

I've chosen one of the top models, mostly used, in the leaderboard, all-MiniLM-L6-v2, which can be used for tasks like clustering or semantic search and yields an embedding size of 384. it is 5 times faster and still offers good quality and is very popular.

It supports a sequence length of 256 tokens (WordPiece) which is equivalent to ~160 words. It doesn’t necessarily tell that the meaning is similar among ISIS tweets hence the contextualized, SBERT, embeddings aren’t supposed to be separable.

SBERT models are trained under text clustering (MTEB) challenges (based on weakly supervised data: stack-exchange,...), and its performance isn’t perfect though adopted by many applications

Data sets of Isis tweets have 17410 entries with a maximal length of tweet is 160, which is supported by the SBERT model chosen. For larger tweets, a different model should be considered or a different way of handling them.


## Data handling and EDA

### Data split
Split data to test set (10%) and the remaining to train/cross-validation(CV). CV for optimizing hyperparameters and architecture structure. K-fold CV of 5-folds was taken creating 20%/80% split between train and validation. Training the final model based on all the data is a good practice (Todo).
The train/val/test split was stratified by the username as I found that each username has its way of expressing and hence avoiding leakage. Essentially, a username as a feature isn’t good practice for generalization, though it prejudge the username for good.

### Data skewness
Positive class is the minority: positive has the support of 17410 while negative has 123343, which reflects real-life situations. therefore the best way is to measure AP (precision-recall curve) where the Precision will reflect the majority misclassified as positive (minority class). Further balancing the batch by resampling with replacement of the minority (WeightedRandomSampler())


### Preprocessing

Processing by removing noninformative emojis (deEmojify()) and converting/ removing abbreviations  (removes OMG and LOL ). A further processing should be taken w/ should be converted further to “with”

The description field isn’t supported all the time and further thoughts should be carried how to exploit it

Misc symbols
Keywords:  @War_Reports 
Hashtags:  #Aleppo
www links such as: https://t.co/y1WWfRJdTp/s/AqAU  most don’t exist anymore hence it may not generalize well based on that.
Tokens like \ud83d\ude34 appear in the random tweets

Short tweets like 'R08o/s/q8SJ' or ‘his’ or 'A B' could be noisy and non-informative but maybe a coded message  between Isis fanboys 

Despite the noisy tokens, the model can generalize well and the cleaning of the noise could be bootstraped by observing the maximal loss-based examples over the training set. It should be demonstrated, however. 

## Discussion, limitations and further recommendations

- Training over 8 epochs over the 5 folds and evaluating over the test-set was taken
- AP over validation was averaged over the folds to get AP=90%. It was translated to a test set (AP=96%) for examining generalization
- Results are high and can stem from the fact that the test-set isn't too representative (discussed later)
- The learning curve shows signs of near overfitting which can tell that stratification was over the right attribute, other-wise the training and validation set would have been correlated
- 
### Actions to explore further given more time
 - Determine the right threshold for the classifier, trading off precision/recall or FN/FP implicitly. We can see that the threshold is nearly the same over the validation and test set.   
 - Error analysis for understanding upon what examples the model failed to predict, extracting noisy tweets. 
 - Time precedence of the tweets wasn't examined 
 - Assess early stopping supporting implicit regularization by averaging over the folds
 -- A nested CV for randomizing new test-set each time is also an option
 - For assessing the final model performance, a good practice is to train 3 models, over the entire training-set, with different seeds and averages over the test-set AP.
 - Few of the tweets are in Arabic hence:
 -  Considering another multi-lingual model such as “distiluse-base-multilingual-cased-v1”
 -  Translating to English by language translator, with the expense of non-accuracy of the translation
 - Using the place information as a prior, by a place recognition model, out of the cities/countries list and attributing to a terror act
 - Further optimizing hyperparameters(learning rates, weights decay, dropout, …) and architecture structure (more layers, different activation functions,..)
 - using the username as a prior, though it is indicative for that set, can be overfitting and will be hard to generalize to unseen tweets
 - Validate if the existence of web links, and short tweets inside the tweets improves the KPI or not
Further processing should be taken over abbreviations such as “w/” should be converted further to “with”
 - Examine the generalization vs. a holdout set such as tweets from later periods: a year or two after the terror attack in Paris
 - Though there is a chance of not generalizing to recent tweets from the same reason explained, we may consider incorporating N-gram based approach, or NER, to gain more confidence by Identifying more frequently or names of prominent clergy:  "Anwar Awlaki", "Ahmad Jibril", "Ibn Taymiyyah", "Abdul Wahhab". Examples of clergy that they hate the most: "Hamza Yusuf", "Suhaib Webb", "Yaser Qadhi", "Nouman Ali Khan", "Yaqoubi".

![learning_curve_fold_0_](https://github.com/hanochk/Isis_tweet_detection/assets/8217391/f19d44f1-8659-42cf-a46f-1d4cf270cbe4)

![learning_curve_fold_1_](https://github.com/hanochk/Isis_tweet_detection/assets/8217391/e1a10e0d-6f98-4b6e-a4fd-4b1724e6c9e3)

![Isis tweets classifier validation-set fold 0p_r_curve](https://github.com/hanochk/Isis_tweet_detection/assets/8217391/f0441141-af1a-4c69-ac48-db7e71a96279)


![Isis tweets classifier test-set p_r_curve](https://github.com/hanochk/Isis_tweet_detection/assets/8217391/76dde5c9-5bc3-495f-8121-8236a0eb03b7)
