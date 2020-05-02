import urllib.request
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import time
import newspaper
from newspaper import Article
from tqdm import tqdm
import re
import json
import nltk
from nltk.corpus import stopwords
from nltk.tag.perceptron import PerceptronTagger
import os
import pytz
import datetime
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')


#Init variables
url = "https://news.google.com/topics/CAAqJggKIiBDQkFTRWdvSUwyMHZNRGx6TVdZU0FtVnVHZ0pKVGlnQVAB?hl=en-IN&gl=IN&ceid=IN%3Aen"
tz = pytz.timezone('Asia/Kolkata')
global prev
global prev1
global heads
global subs
heads =[]
subs = []
prev = []
prev1 =[]

with open("./NTUSD-Fin/NTUSD_Fin_word_v1.0.json", "r") as f:
    data = f.read()
    NTUSD = json.loads(data)

word_sent_dict = {}
for i in range(len(NTUSD)):
    word_sent_dict[NTUSD[i]["token"]] = NTUSD[i]["market_sentiment"]

stop_words = set(stopwords.words('english'))
stop_word= ['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}', '@', '#']

def remove_stopwords(data):
    sentence_token = [s.split(' ') for s in data] 
    idx = 0
    for sentence in sentence_token:
        clean_sentence_token = []
        for word in sentence:
            #if word not in list(stop_words):
            word= ''.join(c for c in word if c not in stop_word)
            if word != '':
                clean_sentence_token.append(word.lower())
        sentence_token[idx] = clean_sentence_token
        idx = idx + 1
    return sentence_token


def get_score(sentence):
    sent_value = []
    #print(sentence)
    sentence_tagged = np.array(nltk.pos_tag(sentence))
    #print(sentence_tagged)
    for tagged in sentence_tagged:
        word = tagged[0]

        #print(word)
    
        #get NTUSD dict score
        try: 
            dict_score = word_sent_dict[word]
        except:
            dict_score = 0.0
            
    
        
        word_score = np.array([dict_score], dtype=float)
        sent_value.append(word_score)
    #print(sent_value)
    return np.average(np.array(sent_value), axis=0)



def NLP(text,check):

    if(check == 'head'):
        sentence = remove_stopwords([text])
        test_pred = get_score(sentence[0])
        #head_score.append(test_pred)
        return test_pred

    elif(check == 'sub'):
        sentence = remove_stopwords([text])
        test_pred = get_score(sentence[0])
        #sub_score.append(test_pred)
        return test_pred


def pre(x):
    if(pd.isnull(x) == True):
        return datetime.datetime.now(tz)
    else:
        return x


def find_word(text):
    result = []
    search1 = ['surge', 'acquisitions', 'IPO']
    text1 = text.split('.')

    for i in search1:

        for line in text1:
            if i in line:
                #print(text.index(line))
                #print(i +' was found in ' + str(address.strip()))
                #print(' ')
                #result.append(text.index(line))
                result.append(line)
                #print(line)

    return result




def Summarize(Article,check):

    try:

        Article.download() 
        Article.parse()
        Article.nlp()
        a = Article.text
        res = find_word(a)
        score = NLP(Article.title,check)
        return Article.title, Article.publish_date, res, Article.summary,score

    except newspaper.ArticleException:
        return np.nan ,np.nan, np.nan,np.nan,np.nan



def main():

    try:
        os.remove('MainNews.csv')
        os.remove('SubNews.csv')
        os.remove('Main_news.json')
        os.remove('Sub_news.json')
    except OSError:
        pass

    while(True):


        summ = []
        Subsumm = []
        Maindate = []
        Subdate =[]
        MainLink =[]
        SubLink =[]
        MainTitle = []
        SubTitle = []
        head_score = []
        sub_score = []
        

        time.sleep(20)
        page=urllib.request.Request(url,headers={'User-Agent': 'Mozilla/5.0'}) 
        infile=urllib.request.urlopen(page).read()
        print('This might take a while, Please wait for loader!') 
        soup = BeautifulSoup(infile,'lxml')
        section = soup.find("div", {"jsname": "esK7Lc"})
        for a in tqdm(section,ncols=90):
            try:
                
                #Main news
                head = a.find('h3', class_='ipQwMb ekueJc gEATFF RD0gLb')
                heading = head.text
                link = head.find('a', class_= 'DY5T1d')
                link = link['href']
                Address = 'https://news.google.com'+link[1:]
                

                #Sub new
                sub = a.find('h4', class_='ipQwMb ekueJc gEATFF RD0gLb')
                sub_heading = sub.text
                sublink = sub.find('a', class_= 'DY5T1d')
                sublink = sublink['href']
                SubAddress = 'https://news.google.com'+sublink[1:]


                if((heading in prev)):
                    continue
                
                else:
                    prev.append(heading)

            
        #This Except will be only called when there is no sub news    
            except AttributeError:

                if((heading in prev)):
                    continue
                
                else:
                    prev.append(heading)
                    article = Article(Address.strip(), language="en")
                    title , date , search , summary, score = Summarize(article,'head')
                    MainTitle.append(title)
                    Maindate.append(date)
                    heads.append(search)
                    summ.append(summary)
                    head_score.append(score)
                    MainLink.append(Address)
                    continue

            

            #MAIN NEWS EXTRACT
            article = Article(Address.strip(), language="en")
            title , date , search , summary,score = Summarize(article,'head')
            MainTitle.append(title)
            Maindate.append(date)
            heads.append(search)
            summ.append(summary)
            head_score.append(score)
            MainLink.append(Address)


            #SUB NEWS EXTRACT
            Subarticle = Article(SubAddress.strip(), language="en")
            title1 , date1 , Subsearch , summary1, score1 = Summarize(Subarticle,'sub')
            SubTitle.append(title1)
            Subdate.append(date1)
            subs.append(Subsearch)
            Subsumm.append(summary1)
            sub_score.append(score1)
            SubLink.append(SubAddress)


        #Main News Table
        MainNews ={'Title': MainTitle, 'Summary': summ, 'Date':Maindate, 'URL': MainLink, 'Score': list(map(float,head_score))}
        Maindf = pd.DataFrame(MainNews)
        Maindf['Date'] = Maindf['Date'].apply(pre)
        Maindf.drop_duplicates(keep='first', inplace=True)
        Maindf.dropna(inplace=True)
        Maindf.to_csv('MainNews.csv',mode='a',index=False)
        

        #Sub News Table
        SubNews ={'Title': SubTitle, 'Summary': Subsumm, 'Date':Subdate, 'URL': SubLink, 'Score': list(map(float,sub_score))}
        Subdf = pd.DataFrame(SubNews)
        Subdf['Date'] = Subdf['Date'].apply(pre)
        Subdf.drop_duplicates(keep='first', inplace=True)
        Subdf.dropna(inplace=True)
        Subdf.to_csv('SubNews.csv', mode='a',index=False)
        

        df = pd.read_csv('MainNews.csv')
        df.drop_duplicates(keep='first', inplace=True)
        df.to_json('Main_news.json',orient='records',date_format = 'iso')


        df1 = pd.read_csv('SubNews.csv')
        df1.drop_duplicates(keep='first', inplace=True)
        df1.to_json('Sub_news.json',orient='records',date_format = 'iso')


        #Main News search print
        main = []
        subs1 = []
        
        for i in range(len(heads)):
            if(heads[i] != []):
                if(str(heads[i]) != 'nan' and heads[i] != heads[i-1]):
                    main.append(heads[i])

        for x in subs:
            if(x != []):
                if(str(x) != 'nan'):
                    subs1.append(x)


        with open('Main_search.json', 'w') as json_file:
            json.dump(main, json_file)

        with open('Sub_search.json', 'w') as json_file:
            json.dump(subs1, json_file)
        

        t = 40
        while t>=0:
            mins,secs = (00,t)
            timer = '{:02d}:{:02d}'.format(mins,secs)
            #print('Countdown: ' + timer, end = '\r')
            time.sleep(1)
            t -= 1

    
if __name__ == "__main__":
    main()


