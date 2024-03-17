import re
import pandas as pd
import regex
import datetime
import emojis
import regex
import emoji
import multidict as multidict
import numpy as np
import os
import re
from PIL import Image
from os import path
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import operator

def date_time(s):
    
    pattern = '^([0-9]+)(\/)([0-9]+)(\/)([0-9]+), ([0-9]+):([0-9]+)[ ]?(AM|PM|am|pm)? -'
    result = regex.match(pattern, s)
    if result:
        return True
    return False

def find_author(s):
    s = s.split(":")
    if len(s)>=2:
        return True
    else:
        return False

def getDatapoint(line):
    splitline = line.split(' - ')
    dateTime = splitline[0]
    # date, time = dateTime.split(", ")
    message = " ".join(splitline[1:])
    if find_author(message):
        splitmessage = message.split(": ")
        author = splitmessage[0]
        message = " ".join(splitmessage[1:])
    else:
        author= "None"
    return dateTime, author, message
data = []
conversation = 'whatsapp.txt'
with open(conversation, encoding="utf-8") as fp:
    fp.readline()
    messageBuffer = []
    date, time, author = None, None, None
    while True:
        line = fp.readline()
        if not line:
            break
        line = line.strip()
        if date_time(line):
            if len(messageBuffer) > 0:
                data.append([dateTime, author, ' '.join(messageBuffer)])
            messageBuffer.clear()
            dateTime, author, message = getDatapoint(line)
            messageBuffer.append(message)
        else:
            messageBuffer.append(line)

# data
df = pd.DataFrame(data, columns=['dateTime', 'Author', 'Message'  ])
# df
df['dates']=pd.to_datetime(df['dateTime'])
# df

df['year'] = df['dates'].dt.year
df['month'] = df['dates'].dt.month_name()

df['date'] = df['dates'].dt.day

df['hour'] = df['dates'].dt.hour
df['minutes'] = df['dates'].dt.minute

df['day']=df['dates'].dt.strftime('%A')

df['words'] = df['Message'].apply(lambda n: len(n.split()))

# df['Author'].value_counts().head()

user=df.Author.unique()

df2 = df.copy()
df2 = df2[df2.Author != "None"]
top10df = df2.groupby("Author")["Message"].count().sort_values(ascending=False)
# top10df
top_user_bychat=top10df.to_dict()
# top_user_bychat

import plotly.express as px
fig = px.histogram(y=list(top_user_bychat.values()),x=list(top_user_bychat.keys()))
fig.update_layout(title='Number of messages per users',title_x=0.5,xaxis_title='Users',yaxis_title='No of messages')
# fig.show()
fig.update_layout({
    'plot_bgcolor': 'rgba(0,0,0,0)',
    'paper_bgcolor': 'rgba(0,0,0,0)'})
fig.write_html('./static/fig1.html')
topMedia = df[df.Message == '<Media omitted>'].groupby('Author').count().sort_values(by="Message", ascending = False).head(10)
topMedia.drop(columns=['dates','year','month','hour','date','minutes','day','words','dateTime'],inplace=True)
topMedia.rename(columns={"Message": "media_sent"}, inplace=True)

media=topMedia.to_dict()
media['media_sent']

import plotly.graph_objects as go

labels = list(media['media_sent'].keys())
values = list(media['media_sent'].values())
fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.5)])
fig.update_layout({
    'plot_bgcolor': 'rgba(0,0,0,0)',
    'paper_bgcolor': 'rgba(0,0,0,0)'})
fig.update_layout(title='Number of media sent per users',title_x=0.5)

# fig.show()
fig.write_html('./static/fig2.html')

df3 = df.copy()
df3['message_count'] = [1] * df.shape[0]  
grouped_by_time = df3.groupby('hour').sum().sort_values(by = 'hour',ascending=True).head(24)
grouped_by_time.drop(columns=['date','year','date','minutes','words'],inplace=True)
group_time=grouped_by_time.to_dict()
# group_time
# df3

most_active_hours={}
for i in range(0,24):
  if i in group_time['message_count'].keys():
    most_active_hours[str(i)] = group_time['message_count'][i]
  else:
    most_active_hours[str(i)] = 0
# most_active_hours

x = list(most_active_hours.keys())
y=  list(most_active_hours.values())
fig = go.Figure(data=go.Scatter(x=x, y=y))
fig.update_layout(title='Activity by Hours',title_x=0.5,xaxis_title='Hour',yaxis_title='No of messages')
fig.update_layout({
    'plot_bgcolor': 'rgba(0,0,0,0)',
    'paper_bgcolor': 'rgba(0,0,0,0)'})
# fig.show()
fig.write_html('./static/fig3.html')

days = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
# grouped_by_day = df3.groupby('day').sum().reset_index()[['day', 'message_count']]
grouped_by_day = df3.groupby('day').sum().sort_values(by = 'day',ascending=True)
grouped_by_day.drop(columns=['hour','date','year','minutes','words'],inplace=True)
temp=grouped_by_day.to_dict()
# temp

most_active_days={}
for i in days:
  if i in temp['message_count'].keys():
    most_active_days[i] = temp['message_count'][i]
most_active_days

import plotly.graph_objects as go

fig = go.Figure(go.Barpolar(
    r=list(most_active_days.keys()),
    theta=list(most_active_days.values()),
   # width=[20,15,10,20,15,30,15,],
   # marker_color=["#E4FF87", '#709BFF', '#709BFF', '#FFAA70', '#FFAA70', '#FFDF70', '#B6FFB4'],
   # marker_line_color="black",
    marker_line_width=2,
   opacity=0.8
))
fig.update_layout(
    template=None,
    polar = dict(
        radialaxis = dict(range=[0, 6]),
        angularaxis = dict()
    ),
    title='Number of messages by day'
)
fig.update_layout({
    'plot_bgcolor': 'rgba(0,0,0,0)',
    'paper_bgcolor': 'rgba(0,0,0,0)'})


# fig.show()
fig.write_html('./static/fig4.html')

# (most_active_days.keys())

months = ['January','February','March','April','May','June','July','August','September','October','November','Decemeber']
# grouped_by_day = df3.groupby('day').sum().reset_index()[['day', 'message_count']]
grouped_by_month = df3.groupby('month').sum().sort_values(by = 'month',ascending=True)
grouped_by_month.drop(columns=['hour','date','year','minutes','words'],inplace=True)
temp_month=grouped_by_month.to_dict()
# temp_month

most_active_month={}
for i in months:
  if i in temp_month['message_count'].keys():
    most_active_month[i] = temp_month['message_count'][i]
  else:
    most_active_month[i]=0
# most_active_month

# import kaleido
# !pip install  kaleido

# import kaleido

import plotly.express as px
import plotly.io as pio

fig = px.bar(y=list(most_active_month.values()), x=list(most_active_month.keys()), text_auto='.2s')
fig.update_layout(title='Number of messages per Month',title_x=0.5,yaxis_title='No of messages')
fig.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
fig.update_layout({
    'plot_bgcolor': 'rgba(0,0,0,0)',
    'paper_bgcolor': 'rgba(0,0,0,0)'})
# fig.show()
fig.write_html('./static/fig5.html')

def split_count(text):
    emoji_list = []
    data = regex.findall(r'\X', text)
    for word in data:
        emojii = emojis.get(word)
        emoji_list.extend([emojis.decode(is_emoji) for is_emoji in emojii])
    return emoji_list
df3['emoji_list'] = df['Message'].apply(split_count)
df3

total_emojis_list = list(set([a for b in df3.emoji_list for a in b]))
total_emojis = len(total_emojis_list)
# print(total_emojis)

from collections import Counter
total_emojis_list = list([a for b in df3.emoji_list for a in b])
emoji_dict = dict(Counter(total_emojis_list))
# emoji_dict

emoji_final={}
for i in emoji_dict.keys():
  emoji_final[emojis.encode(i)] = emoji_dict[i] 
# emoji_final

real = (dict(sorted(emoji_final.items(), key=operator.itemgetter(1), reverse=True)[:5]))
labels = list(real.keys())
values = list(real.values())
fig = go.Figure(data=[go.Pie(labels=labels, values=values, textinfo='label+percent',insidetextorientation='radial')])
fig.update_layout(title='Number of emojis',title_x=0.5)
# fig.show()
fig.update_layout({
    'plot_bgcolor': 'rgba(0,0,0,0)',
    'paper_bgcolor': 'rgba(0,0,0,0)'})
fig.write_html('./static/fig6.html')

emoji_per_user={}
l = df3.Author.unique()
for i in range(len(l)):
  dummy_df = df3[df3['Author'] == l[i]]
  total_emojis = list([emojis.encode(a) for b in dummy_df.emoji_list for a in b])
  emoji_dict_individual = dict(Counter(total_emojis))
  emoji_per_user[l[i]]=(emoji_dict_individual)
# emoji_per_user

words=[]
df_wordcloud = df.copy()
def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)
df_wordcloud['Message']=df_wordcloud['Message'].apply(remove_urls)
for mess in df_wordcloud['Message']:
    if(mess=='<Media omitted>'):
      continue
    else:
      words.extend(mess.split())

# print(words)

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
stop_words = set(stopwords.words('english'))
nostops=[]
for w in words:
    if w not in stop_words:
        nostops.append(w)

# nostops



from collections import Counter
word_cloud = Counter(nostops)
word_cloud

def makeImage(text):
    alice_mask = np.array(Image.open("static\\b1.png"))
    wc = WordCloud(background_color="white", max_words=1000, mask=alice_mask)
    # generate word cloud
    wc.generate_from_frequencies(text)

    # show
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    # plt.show()
    plt.savefig('./static/fig7.png')



makeImage(word_cloud)


from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
sentiment = SentimentIntensityAnalyzer()

pd.options.mode.chained_assignment = None 
df_sent = df[df.Message != '<Media omitted>']
df_sent = df_sent[df_sent['Message'].map(len) > 10]
df_sent.drop(columns=['dates','year','month','hour','date','minutes','day','words','dateTime'],inplace=True)


df_sent['Negative']=df_sent['Message'].map(lambda text: sentiment.polarity_scores(text)["neg"])
df_sent['Neutral']=df_sent['Message'].map(lambda text: sentiment.polarity_scores(text)["neu"])
df_sent['Positive']=df_sent['Message'].map(lambda text: sentiment.polarity_scores(text)["pos"])
df_sent['Compound']=df_sent['Message'].map(lambda text: sentiment.polarity_scores(text)["compound"])

hf = df_sent.sort_values(by = ["Negative"])
hf = hf.tail(5)
neg = (hf["Author"]+" : "+hf["Message"]).tolist()

hf = df_sent.sort_values(by = ["Positive"])
hf = hf.tail(5)
p = (hf["Author"]+" : "+hf["Message"]).tolist()


hf = df_sent.sort_values(by = ["Neutral"])
hf = hf.tail(5)
neu = (hf["Author"]+" : "+hf["Message"]).tolist()


final_sentiments = []
for i in p:
  final_sentiments.append(i)
for i in neg:
  final_sentiments.append(i)
for i in neu:
  final_sentiments.append(i)

df2['Dates'] = pd.to_datetime(df2['dates'].dt.strftime('%Y-%m-%d'))
df2['Time'] = pd.to_datetime(df2['dates']).dt.time
df2['Dates'] = pd.to_datetime(df2['dates']).dt.date
df2['Time'] = pd.to_datetime(df2['dates']).dt.time
# df2['Dates'] = df2['Dates'].dt.strftime('%Y-%m-%d')
dat = "2022-10-04"
dtt = datetime.datetime.strptime(dat,"%Y-%m-%d").date()
# dtt= datetime.date(dtt)
print(type(df2['Dates']))
dfDate=df2[df2['Dates']>=dtt]

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib


model = joblib.load('cyberbullying_detection_model.pkl')
# Prediction
input_text = [["skinny bitch you"],["happy birthday"],["fuck off"],["these christians never change"],["you are an idiot"]]
off=[]
for i in range(len(input_text)):
  prediction = model.predict(input_text[i])

  if prediction[0] == 1:
    off.append(input_text[i])

import nltk
nltk.download('opinion_lexicon')

import nltk
from nltk.corpus import opinion_lexicon
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize

positive_wds = set(opinion_lexicon.positive())
negative_wds = set(opinion_lexicon.negative())
def sentiment_analysis(df,start,end):
  start_date = datetime.datetime.strptime(start,"%Y-%m-%dT%H:%M").date()
  start_time = datetime.datetime.strptime(start,"%Y-%m-%dT%H:%M").time()
  end_date = datetime.datetime.strptime(end,"%Y-%m-%dT%H:%M").date()
  end_time = datetime.datetime.strptime(end,"%Y-%m-%dT%H:%M").time()
  dfDate=df[df['Dates']>=start_date]
  dfDate=dfDate[dfDate['Dates']<=end_date]
  dfDate=dfDate[dfDate['Time']>=start_time]
  dfDate=dfDate[dfDate['Time']<=end_time]
  dfDate=dfDate[dfDate['Message'] != '<Media omitted>']
  def score_sent(sent):
    """Returns a score btw -1 and 1"""
    sent = [e.lower() for e in sent if e.isalnum()]
    total = len(sent)
    pos = len([e for e in sent if e in positive_wds])
    neg = len([e for e in sent if e in negative_wds])
    if total > 0:
      return (pos - neg) / total
    else:
      return 0
  def score_review(review):
    sentiment_scores = []
    sents = sent_tokenize(review)
    for sent in sents:
      wds = word_tokenize(sent)
      sent_scores = score_sent(wds)
      sentiment_scores.append(sent_scores)
    return sum(sentiment_scores) / len(sentiment_scores)
  dfDate['scores'] = dfDate['Message'].apply(lambda x:score_review(x))
  def score_to_rating(value):
    if value > 0.2:
      return 2
    if value <= 0.2 and value >= -0.2:
      return 1
    else:
      return 0
  def score_to_senti(value):
    if value > 0.2:
      return "Positive"
    if value <= 0.2 and value >= -0.2:
      return "Neutral"
    else:
      return "Negative"
  dfDate['sentiment'] = dfDate['scores'].apply(lambda x:score_to_rating(x))
  dfDate['senti'] = dfDate['scores'].apply(lambda x:score_to_senti(x))
  count = list(dfDate['sentiment'])
  value = Counter(count)
  Negative = value[0] if value[0] else -1
  Positive = value[2] if value[2] else -1
  Neutral = value[1] if value[1] else -1
  if(Negative==Positive==Neutral):
    finalSentiment='Neutral'
  elif(Negative>Positive and Negative>Neutral):
    finalSentiment='Negative'
  elif(Positive>Neutral and Positive>Negative):
    finalSentiment='Positive'
  elif(Neutral>Positive and Neutral>Negative):
    finalSentiment='Neutral'
  else:
    finalSentiment='Neutral'
  poStatements=dfDate[dfDate['sentiment']==2]
  hf = poStatements.sort_values(by = ["scores"])
  hf = hf.tail(5)
  p = hf["Message"].tolist()
  negStatements=dfDate[dfDate['sentiment']==0]
  hf = negStatements.sort_values(by = ["scores"])
  hf = hf.tail(5)
  neg = hf["Message"].tolist()
  neuStatements=dfDate[dfDate['sentiment']==1]
  hf = neuStatements.sort_values(by = ["scores"])
  hf = hf.tail(5)
  neu = hf["Message"].tolist()
  if len(p)==0:
    p.append("There are no Postive messages")
  if len(neg)==0:
    neg.append("There are no Negative messages")
  if len(neu)==0:
    neu.append("There are no Neutral messages")
  real=dfDate['senti'].tolist()
  real=Counter(real)
  labels = list(real.keys())
  values = list(real.values())
  fig = go.Figure(data=[go.Pie(labels=labels, values=values, textinfo='label+percent',insidetextorientation='radial')])
  fig.update_layout(title='Sentiment',title_x=0.5)
  fig.update_layout({
    'plot_bgcolor': 'rgba(0,0,0,0)',
    'paper_bgcolor': 'rgba(0,0,0,0)'})
  # fig.show()
  fig.write_html('./static/fig8.html')
  return finalSentiment,p,neg,neu
dfNone = df.copy()
dfNone = dfNone[dfNone.Author == "None"]
def conversation_users(df):
    users = []
    df = df.groupby('Author').count().reset_index()
    for index, row in df.iterrows():
        users.append(row['Author'])
    users.remove('None')
    return users
def group_name(dfNone):
  grpname = dfNone['Message'][0]
  def findGname(x):
    if 'changed the subject' in x:
      gname = x
      return gname
  grpname1=dfNone['Message'].apply(findGname)
  grpname1 = grpname1.replace(to_replace='None', value=np.nan).dropna()
  if grpname1.size==0:
    grpnamef = grpname
    flag=1
  else:
    grpnamef= grpname1.iloc[-1]
    flag=0
  if flag==0:
    gname=re.findall('to ".+"',grpnamef)
    grpnamef=gname[0].replace("to ",'')
    grpnamef=grpnamef.strip('"')
    return grpnamef
  else:
    gname = re.findall('".+"',grpnamef)
    grpnamef=gname[0].strip('"')
    return grpnamef

startDate=df['dateTime'][0]
endDate=df['dateTime'][len(df)-1]
totalMessages=df2['Message'].count()
mediacount=df[df['Message'] == '<Media omitted>'].shape[0]
emojiCount=sum(emoji_final.values())
mostEmoji=max(zip(emoji_final.values(), emoji_final.keys()))[1]
users = conversation_users(df)
grpname = group_name(dfNone)
uniqueEmoji=len(emoji_final)
def dataInsights():
  return {'users':users,
          'userCount':len(users),'grpname':grpname,'startDate':startDate,'endDate':endDate,'totalMessages':totalMessages,'mediacount':mediacount,'emojiCount':emojiCount,'mostEmoji':mostEmoji
          ,'uniqueEmoji':uniqueEmoji}
