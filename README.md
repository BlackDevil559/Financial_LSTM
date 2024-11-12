## **Predicting Stocks with Financial News with LSTM, Differential Privacy and Sentiment Analysis**


---

Team member: Bhumesh Gaur and Satyam Patil

## General Terms in Research Paper

### 1.1	Differential Privacy

Differential Privacy is a technique used in data collection that allows companies such as Apple to use your information to understand large scale trends of data, while at the same time protecting your individual information from being discovered. It involves the injection of noise to your data to protect your individual information, while at the same time allowing overall trends of the data to become apparent. The main reason to use it is to understand trends of the data, while at the same time protecting individual’s privacy from which the data is obtained.

To summarize it, “nothing about an individual should be learnable from the database that cannot be learned without access to the database.” 

### 1.2 ARMA

Initially described in Peter Whittle’s 1951 PhD Thesis, ARMA [or Autoregressive-Modeling-Average Model] is a time series model that describes the evolution of a system as the sum of two polynomials: a polynomial for autoregression [AR for short], and a polynomial for moving average [MA for short], hence the name ARMA.

The autoregression polynomial essentially says that at any arbitrary time t, the state of the system is a linear combination of the previous p states. The moving average polynomial is described as a linear combination of the previous q errors, and the current error to the system.

Contrary to a Markov Chain, which only depends on the state of the present to predict the future, ARMA provides a means to understand the evolution of the state of a system as a function of its previous states.

### 1.3 LSTM

An LSTM (Long Short-Term Memory) network is a type of recurrent neural network (RNN) designed to capture long-term dependencies in sequence data, addressing the limitations of traditional RNNs when learning from sequences of data.

#### Key Components of LSTMs
LSTMs have a unique architecture consisting of memory cells that maintain a cell state across time steps. Each cell in an LSTM network has three types of gates—input, forget, and output gates—which allow the model to store, read, and discard information as necessary, controlling the flow of information through the cell:

Forget Gate: Decides what information should be removed from the cell state. It looks at the current input and previous hidden state and outputs a number between 0 and 1, where 0 means "completely forget" and 1 means "completely keep."

Input Gate: Determines what information will be added to the cell state. It uses a sigmoid function to control which values are updated and a tanh function to create a vector of new candidate values.

Output Gate: Decides what the next hidden state should be (this is used as output and feeds into the next cell). It controls the information that flows to the next time step or layer.

## The Structure of the Model

The model starts off with two ARMA models. One ARMA model is for stock price [or any desire financial parameter] at a collection of successive time points [the training data]. The test data are the remaining time points from the dataset. Another ARMA model is for the sentiment of the stock market / stock parameters [from financial news], and that similarly is partitioned into train and test data. Our overall model is thus a linear combination of these two ARMA models, plus a constant.

We now have an optimization problem. We want to find the parameters of the linear combination that minimize the sum of the squares of the errors at each time point [between our ARMA model and the training data]. This is where LSTM comes in. Incorporating this into the loss function, we use the LSTM to train our temporal data and optimize the network from which we can do predictions. The model is shown below.

<p align="center">
  <img src="https://user-images.githubusercontent.com/45545175/110247305-37ba1880-7f31-11eb-836e-9e1d09365fa4.png" />
</p>


Note that in the model above, the Joint ARMA Model data goes into the loss function that is used in training the LSTM. Now the missing link, which we haven’t shown above, is the Differential Privacy component. Data on each dimension is injected with noise, which is drawn from a normal distribution, each with different mean and variance.

## Implementing the Model

Here, we put the codes and apply it with our dataset:

### 3.1 Loading Dataset

- Historical Stock Prices: This data contains minute-level historical prices in the past 5 years for 86 companies. 
- News articles: This data contains 29630 news articles, each of which corresponds to a specific company (e.g., Apple Inc.). The news articles are categorized by company name. Each article includes four fields: “title”, “full text”, “URL” and “publish time”. There are 81 companies in total, all the companies are included in data 1.

### 3.2 Feature Engineering

Find the missing data, we can fill it with anything

```
df[df.isnull().any(axis=1)]
df_missing_percentage=df.isnull().sum()/df.shape[0] *100
df=df.fillna('missing')
```

### 3.3 Simplifying Sentiment Analysis using VADER in Python (on Social Media Text)

Sentiment Analysis, or Opinion Mining, is a sub-field of Natural Language Processing (NLP) that tries to identify and extract opinions within a given text. The aim of sentiment analysis is to gauge the attitude, sentiments, evaluations, attitudes and emotions of a speaker/writer based on the computational treatment of subjectivity in a text.

VADER (Valence Aware Dictionary and sEntiment Reasoner) is a lexicon and rule-based sentiment analysis tool that is specifically attuned to sentiments expressed in social media. VADER uses a combination of A sentiment lexicon is a list of lexical features (e.g., words) which are generally labelled according to their semantic orientation as either positive or negative. For more details, please refer to: [sentiment analysis](https://medium.com/analytics-vidhya/simplifying-social-media-sentiment-analysis-using-vader-in-python-f9e6ec6fc52f)

After this, you will generate the sentiment score for each company, like this:

<p align="center">
  <img src="https://user-images.githubusercontent.com/45545175/110325511-0d259980-7fdd-11eb-91fe-651ee10ebb3a.png" />
</p>

### 3.4 Boeing Stock Prediciton

Specifically, we select Boeing (BA) and predict its stock price.  
Here, we import the stock price for BA.

```
sp = pd.read_csv("path/historical_price/BA_2015-12-30_2021-02-21_minute.csv")#,index_col=0)
sp=pd.DataFrame(sp)
```
And the sentiment analysis for BA and group it by days.

```
dff1=df[(df.company=='BA')]
dff1g=dff1.groupby(['published_date']).agg(['mean'])
```
Next, we need to fuse these two information together by days. As the stock price is based on minutes, we choose the last day point as the close price for each day.
```
for i in range(0,d1.shape[0]):
    t=d1['published_date'][i]
    timeStruct = time.strptime(t, "%m/%d/%Y") 
    d1['published_date'][i] = time.strftime("%Y-%m-%d", timeStruct) 

sp1=sp.copy()
sp1.drop_duplicates(subset=['Date'], keep='last', inplace = True)
```
Then, Union these two tables together:
```
date_union_1=pd.DataFrame(columns=('idx','date','mean_compound','comp_flag'))
sp_len=sp1.shape[0]
d_len=d1.shape[0]
d=d1.copy()
for i in range(0,sp_len):
    idx=i
    date=sp1['Date'][i]
    j=0
    t=0
    while j<d_len:
        if sp1['Date'][i]==d['published_date'][j]:
            mean_compound=d['compound'][j]
            comp_flag=1
            t=1
            break
        j=j+1
    if t==0:
        mean_compound=0
        comp_flag=0
    
    date_union_1=date_union_1.append(pd.DataFrame({'idx':[idx],
                                                  'date':[date],
                                                  'mean_compound':[mean_compound],
                                                  'comp_flag':[comp_flag]
        
    }),ignore_index=True)
```

### 3.5 Add Noise to the Data

Next, we add some noise to the sentiment score, to make the model more robust。 The noise variance is calculated and added to the score.


### 3.6 Train LSTM model on stock prediciton from a single company (BA)

After we prepare the sentiment score with noise and the stock price. We can start building the model!

We split the dataset into 85% for training and the rest 15% for validation. 

Finally, we build the model here, which contains three LSTM layers and one dense layer. The loss function is mean square error and the optimizer is adam:
```
model = Sequential()
model.add(LSTM(neurons, input_shape=(input_timesteps, input_dim), return_sequences = True))
model.add(Dropout(drop_out))
model.add(LSTM(neurons,return_sequences = True))
model.add(LSTM(neurons,return_sequences =False))
model.add(Dropout(drop_out))
model.add(Dense(dense_output, activation='linear'))
# Compile model
model.compile(loss='mean_squared_error',
                optimizer='adam')
# Fit the model
model.fit(x_train,y_train,epochs=20,batch_size=batch_size)
```

### 3.7 Evaluation on the Testing Dataset


(1) For the single company prediction, we finally get MSE: 57.95, Accuracy: 0.972 and mean error percent: 0.028.
The results demonstrates the good stock predicitons if we use the news from that company.

<p align="center">
  <img src="https://user-images.githubusercontent.com/45545175/110559023-16138980-8109-11eb-9f3f-bc62bd35cb24.png" />
</p>

And we plot the predicitons below:
<p align="center">
  <img src="https://user-images.githubusercontent.com/45545175/110556640-8e2b8080-8104-11eb-81aa-a8082f08ca34.png" />
</p>

(2) Train LSTM model on stock prediciton from this specific company but no noise

Next, we want to see if the stock predicitons of BA would be affected if we don't add noise to the sentiment score.
Here are the results

<p align="center">
  <img src="https://user-images.githubusercontent.com/45545175/110983892-fb6f2980-832f-11eb-8d86-3893931c9456.png" />
</p>
And we plot the predicitons below:
<p align="center">
  <img src="https://user-images.githubusercontent.com/45545175/110983937-0c1f9f80-8330-11eb-9a26-4088cf249227.png" />
</p>

Compared with the previous result, it shows that adding noise to the model have a better predictions of the stock price, which means that adding noise makes the model more generalizable. 

(3) Train LSTM model on stock prediciton from all companies.
Furthermore, we want to see if the stock predicitons of BA would be increased if we use the news from all other companies.

Here are the results:
<p align="center">
  <img src="https://user-images.githubusercontent.com/45545175/110558867-d9e02900-8108-11eb-9ec6-a7509b00b898.png" />
</p>
And we plot the predicitons below:
<p align="center">
  <img src="https://user-images.githubusercontent.com/45545175/110558961-fe3c0580-8108-11eb-84df-7764cfc39abb.png" />
</p>

The comparisons demonstrate that the performance of the stock predicitons decreases if we add in sentiment scores from other companies.

(4) Train LSTM model on stock prediciton without sentiment score.
Finally, we trained the model only with the stock price from previous days without the sentiment score.

Here are the results:
<p align="center">
  <img src="https://user-images.githubusercontent.com/45545175/110560120-0a28c700-810b-11eb-8f18-9bc693d7dab4.png" />
</p>
And we plot the predicitons below:
<p align="center">
  <img src="https://user-images.githubusercontent.com/45545175/110560162-22004b00-810b-11eb-946e-d29a1f02c238.png" />
</p>

The results demonstrate that adding additional sentiment scores from that specific company did increase the model performance. 

To have a better understanding of these different methods, we plot all the predictions together:

<p align="center">
  <img src="https://user-images.githubusercontent.com/45545175/110560900-6213fd80-810c-11eb-9b77-9108b3247c85.png" />
</p>



References


[1] https://arxiv.org/pdf/1912.10806.pdf

[2] https://www.nature.com/articles/s42256-019-0112-6

[3] https://blog.cryptographyengineering.com/2016/06/15/what-is-differential-privacy/

[4] https://www.microsoft.com/en-us/research/publication/differential-privacy/?from=http%3A%2F%2Fresearch.microsoft.com%2Fpubs%2F64346%2Fdwork.pdf

[5] https://en.wikipedia.org/wiki/Autoregressive%E2%80%93moving-average_model

[6] https://colah.github.io/posts/2015-08-Understanding-LSTMs/




