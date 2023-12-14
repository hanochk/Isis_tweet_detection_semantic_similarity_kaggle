# Isis_tweet_detection
Setup: 
install requirements.txt 

First precompute embeddings by setting 
pre_compute_embeddings = True 

For running the prediction process, based on the precomputed embeddings, set: pre_compute_embeddings = False

**Classifying ISIS tweets as pro-Isis 

A few approaches could be considered: 
Probabilistic model (GMM for instance) based on the clustered embeddings representation (created by SBERT) of each tweet for each class and assessing the hypothesis that a tweet belongs to one of the probabilistic models P(t/model_isis) > P(t/model_non_isis).
Semantically sentence similarity-based model that creates representation in high dimension feeding a nonlinear binary classifier (MLP), based on the SBERT embeddings, for instance. SBERT can be a good candidate since it was trained contrastively to maintain similarity between sentences with “similar meaning”. Since the embeddings aren’t necessarily linearly separable in the latent space, an additional MLP-based classifier containing non-linear elements as well, is added. 
Generative-based approach with LLM containing a large context window that can have in-context learning-based prompt with many tweet examples, assessing if a  given tweet from the test set is an Isis based. However, generating confidence needs to be addressed. 

Approach No.2 was adopted here.

A textual similarity model such as SBERT (Bi-Encoder) was trained to get maximal cosine similarity for semantic similar sentences and also used for clustering purposes. 

Disclaimer: many SBERT models are under various challenges such as MTEB one. Hence picking the right model is also an issue to be further optimized.

I've chosen one of the top models, mostly used, in the leaderboard, all-MiniLM-L6-v2, which can be used for tasks like clustering or semantic search and yields an embedding size of 384. it is 5 times faster and still offers good quality and very popular.

It supports a sequence length of 256 tokens (WordPiece) which is equivalent to ~160 words. It doesn’t necessarily tell that the meaning is similar among ISIS tweets hence the contextualized, SBERT, embeddings aren’t supposed to be separable.

SBERT models are trained under text clustering (MTEB) challenges (based on weakly supervised data: stack-exchange,...), and its performance isn’t perfect though adopted by many applications

Data sets of Isis tweets have 17410 entries with a maximal length of tweet is 160, which is supported by the SBERT model chosen. For larger tweets, a different model should be considered or a different way of handling them.


**Data handling and EDA

***Data split
Split data to test set (10%) and the remaining to train/cross-validation(CV). CV for optimizing hyperparameters and architecture structure. K-fold CV of 5-folds was taken creating 20%/80% split between train and validation. Training the final model based on all the data is a good practice (Todo).
The train/val/test split was stratified by the username as I found that each username has its way of expressing and hence avoiding leakage. Essentially, a username as a feature isn’t good practice for generalization, though it prejudge the username for good.

***Data skewness
Positive class is the minority: positive has the support of 17410 while negative has 123343, which reflects real-life situations. therefore the best way is to measure AP (precision-recall curve) where the Precision will reflect the majority misclassified as positive (minority class). Further balancing the batch by resampling with replacement of the minority (WeightedRandomSampler())


***Preprocessing

Processing by removing noninformative emojis (deEmojify()) and converting/ removing abbreviations  (removes OMG and LOL ). A further processing should be taken w/ should be converted further to “with”

The description field isn’t supported all the time and further thoughts should be carried how to exploit it

Misc symbols
Keywords:  @War_Reports 
Hashtags:  #Aleppo
www links such as: https://t.co/y1WWfRJdTp/s/AqAU  most don’t exist anymore hence it may not generalize well based on that.
Tokens like \ud83d\ude34 appear in the random tweets

Short tweets like 'R08o/s/q8SJ' or ‘his’ or 'A B' could be noisy and non-informative but maybe a coded message  between Isis fanboys 

Despite the noisy tokens, the model can generalize well and the cleaning of the noise could be bootstraped by observing the maximal loss-based examples over the training set. It should be demonstrated, however. 


**Discussion and limitations

- AP over validation was averaged over the folds (AP=90%) and was translated to a test set for examining generalization

***Actions to explore further given more time
 - Referring to the time sequence of the tweets
 - Assess early stopping supporting implicit regularization by averaging over the folds
 -- A nested CV for randomizing new test-set each time is also an option
 - For having more better model performance estimation we need to train 3 models with different seeds and average over the test performance 
 - Few of the tweets are in Arabic hence better to consider another multi-lingual model such as “distiluse-base-multilingual-cased-v1”
 - Places recognition out of the cities/countries list and attributing to a terror act
 - Further optimizing hyperparameters(learning rates, weights decay, dropout, …) and architecture structure (more layers, different activation functions,..)
 - Building a  classifier based on the username, though it is indicative for that set it is an overfitting and will be hard to generalize to unseen tweets
 - Validate if the existence of web links, and short tweets inside the tweets improves the KPI or not
Further processing should be taken over abbreviations such as “w/” should be converted further to “with”
 - Examine the generalization vs. a holdout set such as tweets from later in time, a year or two after the terror attack in Paris
 - Consider Incorporating N-gram based approach or NER to gain more confidence by Identifying more frequently or names of prominent clergy:  "Anwar Awlaki", "Ahmad Jibril", "Ibn Taymiyyah", "Abdul Wahhab". Examples of clergy that they hate the most: "Hamza Yusuf", "Suhaib Webb", "Yaser Qadhi", "Nouman Ali Khan", "Yaqoubi".




