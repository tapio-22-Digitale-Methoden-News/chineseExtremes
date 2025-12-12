import mysecrets
import os
import sys
import math

import pandas as pd

from pathlib import Path
import os.path
import glob

import aiohttp
import asyncio
import requests
from urllib.parse import urlparse
import json
import time
import smtplib
import random
import hashlib
import glob
from difflib import SequenceMatcher


import datetime
#from datetime import timezone
from dateutil import parser
##from datetime import date, timedelta, datetime, timezone

from CreLanguageTranslate.LanguageTranslate import LanguageTranslate 

DATA_PATH = Path.cwd()

dtNow = datetime.datetime.fromtimestamp(int(time.time()), datetime.UTC)
dtLastMonth = datetime.datetime.fromtimestamp(int(time.time())-60*60*24*30, datetime.UTC)
##dtNow = datetime.datetime.fromtimestamp(int(time.time()) )

'''
topicColors = {'Thunderstorm':'#53785a', 'Flood':'#030ba1', 'Storm':'#b3b2b1', 'Storm Surge':'#834fa1', 'Flash Flood':'#02b5b8',
               'Tsunami':'#690191', 'Drought':'#edc291', 'Earthquake':'#870007', 'Landslide':'#572c03', 'Cold Wave':'#a7e9fa', 'Heat Wave':'#c40202',
               'Tropical Cyclone':'#4f4f4f', 'Volcano':'#b83202', 'Snow Avalanche':'#deddd5', 'unknown':'#d60d2b'  
               }
'''

def getAge(dateString):
    #print(dateString)
    dateString = str(dateString) 
    dateString = dateString[0:19]+'+00:00'
    #dateString
    today = datetime.datetime.now(datetime.UTC)
    timeDate = -1
    pubDate = None
    try:
        pubDate = parser.parse(dateString)
    except:
        #print('date parse error 1')
        e=1
    if(not pubDate):
      try:
        pubDate = parser.isoparse(dateString)
      except:
        #print('date parse error 2')
        e=2   
    if(pubDate):
        timeDate = today - pubDate
        timeDate = timeDate.days 
    return timeDate

##keywordsFields = ["keyword","language","topic","topicColor","keywordColor","limitPages","ratioNew"]
termsFields = ["index","module", "topic", "color", "feed", "term", "created", "country", "ratio", "counter", "pages", "language", "ipcc", "continent"]
termsDF = pd.read_csv(DATA_PATH / 'terms.csv', delimiter=',')  #,index_col='keyword'
termsDF = termsDF.sort_values(by=['ratio'], ascending=False)  


unsearchedTerms = termsDF
unsearchedTerms['unsearched'] = termsDF['ratio'] - termsDF['counter'] - 0.25*termsDF['pages']
unsearchedTerms = unsearchedTerms.sort_values(by=['unsearched'], ascending=False) 
rows20 = int(math.ceil(unsearchedTerms.shape[0]/5))
unsearchedTerms = unsearchedTerms.head(rows20)
print('11111111111111111111111111111')
print(unsearchedTerms)

def getNewsFiles():
    fileName = './cxsv/news_????_??.csv'
    files = glob.glob(fileName)
    return files  

def getNewsDFbyList(files):    
    newsDF = pd.DataFrame(None)
    for file in files:
        #print(file)
        df = pd.read_csv(file, delimiter=',')
        if(newsDF.empty):
            newsDF = df
        else:
            newsDF = pd.concat([newsDF, df])
    if(not newsDF.empty):
        newsDF = newsDF.sort_values(by=['published'], ascending=True)        
    return newsDF 

def getNewsDF():
    files = getNewsFiles()
    newsDF = getNewsDFbyList(files)
    return newsDF     

newsDf = getNewsDF()
if(not newsDf.empty):
  newsDf['age'] = newsDf['published'].apply(
    lambda x: 
        getAge(x)
  )
  newsDf = newsDf[(newsDf.age>0) & (newsDf.age < 60)]

keywordsNewsDF = pd.DataFrame(None) 
if(not newsDf.empty):
  keywordsNewsDF = newsDf.groupby('term').count()
  keywordsNewsDF = keywordsNewsDF.drop(columns = ['language','index','topic','feed','country', 'ipcc', 'continent'])
  print(keywordsNewsDF)


keywordsNewsDF2 = pd.DataFrame(None) 
if(not keywordsNewsDF.empty):
  keywordsNewsDF2 = pd.merge(termsDF, keywordsNewsDF, how='left', left_on=['term'], right_on=['term'])
  print(keywordsNewsDF2)
  keywordsNewsDF2['counting'] = keywordsNewsDF2['title'].fillna(0)
  keywordsNewsDF2['counting'] = keywordsNewsDF2['counting'] - keywordsNewsDF2['ratio']
  keywordsNewsDF2 = keywordsNewsDF2.sort_values(by=['counting'], ascending=True)  

rows20 = int(math.ceil(keywordsNewsDF2.shape[0]/5))
keywordsNewsDF2 = keywordsNewsDF2.head(rows20)
print(keywordsNewsDF2)   

rows20 = int(math.ceil(termsDF.shape[0]/5))
termsDF3 = termsDF.head(rows20)
print(termsDF3)


searchWords = dict(zip(termsDF.term.values, termsDF.language.values))

#print(termsDF)
#print(searchWords)
#print(termsDF.sample() )


stopDomains = ["www.mydealz.de", "www.techstage.de", "www.nachdenkseiten.de", "www.amazon.de", "www.4players.de", "www.netzwelt.de", "www.nextpit.de",
               "www.mein-deal.com", "www.sparbote.de", "www.xda-developers.com" "www.pcgames.de", "blog.google", "www.ingame.de", "playstation.com",
               "www.pcgameshardware.de", "9to5mac.com", "roanoke.com", "billingsgazette.com", "richmond.com", "www.rawstory.com", "slate.com",
               "www.computerbild.de", "www.giga.de", "www.heise.de", "www.chip.de", 
               "consent.google.com"
                ]


#https://github.com/theSoenke/news-crawler/blob/master/data/feeds_de.txt

                  
def dataIsNotBlocked(data):
    for blocked in stopDomains: 
        if blocked in data['domain']:
            return False
    return True         

#replace/drop: "https://www.zeit.de/zustimmung?url="  

#get url data (inq)  -> check if keyword in title|description    and url equal
#see 'https://www.stern.de/panorama/weltgeschehen/news-heute---ocean-viking--rettet-mehr-als-40-menschen-aus-dem-mittelmeer-30598826.html'



lt = LanguageTranslate()

def translateData(data):
   print(['translate from', data['language']]) 
   anyText = str(data['title']) + '. ' + str(data['description'])
   if('de'==data['language']):
       data['de'] = anyText
       data['en'] = lt.getTranslatorByLanguage('de','en').translate(anyText)
       data['la'] = lt.getTranslatorByLanguage('de','la').translate(anyText)
   if('en'==data['language']):
       data['en'] = anyText
       data['de'] = lt.getTranslatorByLanguage('en','de').translate(anyText)
       data['la'] = lt.getTranslatorByLanguage('en','la').translate(anyText)
   if('' == data['language']):
       data['language'] = 'xx'
       data['de'] = ''
       data['en'] = ''
       data['la'] = '' 
       return data
   if(not data['language'] in ['en','de','','xx','pl']):
       sourceLanguage = data['language']
       data['de'] = lt.getTranslatorByLanguage(sourceLanguage,'de').translate(anyText)
       data['en'] = lt.getTranslatorByLanguage(sourceLanguage,'en').translate(anyText)
       data['la'] = lt.getTranslatorByLanguage(sourceLanguage,'la').translate(anyText)
   return(data) 


collectedNews = {}

def addNewsToCollection(data):
    global collectedNews
    pubDate = parser.parse(data['published'])
    fileDate = 'news_'+pubDate.strftime('%Y_%m')+'.csv'
    if(fileDate in collectedNews):
      if(not data['url'] in collectedNews[fileDate]):
        if(not 'archive' in data):
           data = archiveUrl(data)
           try:
              data = translateData(data)
           except Exception as X:
              print(["article translation went wrong: ",  data,X])   
           else:  
              collectedNews[fileDate][data['url']] = data
        return True
    return False

def storeCollection():
    global collectedNews
    print("Inside store")
    #cols = ['url', 'valid', 'domain', 'title', 'description', 'image', 'published', 'archive', 'content', 'quote', 'language','term', 'topic', 'feed', 'country', 'ipcc', 'continent']
    cols = ['published', 'topic', 'term', 'domain', 'language', 'valid', 'de', 'title', 'description', 'en', 'la', 'url', 'image', 'archive', 'content', 'quote', 'added', 'feed', 'country', 'ipcc', 'continent']
    for dateFile in collectedNews:
        df = pd.DataFrame.from_dict(collectedNews[dateFile], orient='index', columns=cols)
        df.index = df['url'].apply( lambda x: hashlib.sha256(x.encode()).hexdigest()[:32])  
        df = removeDuplicates(df)
        #df.to_csv(DATA_PATH / dateFile, index=True) 
        if(not os.path.exists(DATA_PATH / 'cxsv')):
            os.mkdir(DATA_PATH / 'cxsv')
        print(["store file: ", dateFile])    
        df.to_csv(DATA_PATH / 'cxsv' / dateFile, index_label='index') 
    collectedNews = {}

# self.randomWordsDF = pd.DataFrame.from_dict(self.randomWords, orient='index', columns=self.randomBase.keys())  
# self.randomWordsDF.to_csv(DATA_PATH / self.category / "cxsv" / ("words_random_"+str(self.randomSize)+".csv"), index=True)


#https://web.archive.org/save/https://translate.google.com/translate?sl=de&tl=en&u=
#https://web.archive.org/save/https://translate.google.com/translate?sl=de&tl=en&u=https://www.nikos-weinwelten.de/beitrag/weinbau_reagiert_auf_den_klimawandel_abschied_vom_oechsle_hin_zur_nachhaltigkeit/


# https://docs.aiohttp.org/en/stable/client_reference.html
# 
async def saveArchive(saveUrl):
    async with aiohttp.ClientSession() as session:
      async with session.get(saveUrl, timeout=10) as response:    #120
        print("x")   

async def getArchives(urlList):
    async with aiohttp.ClientSession() as session:
      async with session.get(saveUrl) as response:
        print("x")   

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


def checkDuplicates(dict1, data2):
    quote2 = str(data2['domain']) + ' ' + str(data2['title']) + ' ' + str(data2['description'])
    md52 = hashlib.md5(quote2.encode('utf-8')).hexdigest() 
    for url1 in dict1:
        data1 = dict1[url1]
        quote1 = str(data1['domain']) + ' ' + str(data1['title']) + ' ' + str(data1['description'])   
        md51 = hashlib.md5(quote1.encode('utf-8')).hexdigest()
        if(md52 == md51):
            return True 
        day1 = '1970-01-01'
        if(len(str(data1['published']))>5):
          #print(data1['published'])
          #print(data1)
          pubDate1 = parser.parse(data1['published'])
          day1 = pubDate1.strftime('%Y-%m-%d')
        groupTxt1 = str(data1['domain']) +  ' ' + day1
        group1 = hashlib.md5(groupTxt1.encode('utf-8')).hexdigest()  

        day2 = '1970-01-01'
        if(len(str(data2['published']))>5):
          pubDate2 = parser.parse(data2['published'])
          day2 = pubDate2.strftime('%Y-%m-%d')
        groupTxt2 = str(data2['domain']) +  ' ' + day2
        group2 = hashlib.md5(groupTxt2.encode('utf-8')).hexdigest()  
        if(group1 == group2):
          quote1 = str(data1['title']) + ' ' + str(data1['description'])
          quote2 = str(data2['title']) + ' ' + str(data2['description'])
          similarity = similar(quote1,quote2)
          if(similarity>0.8):
            return True
    return False


def removeDuplicates(df1):
    df1['md5'] = ''
    df1['group'] = ''
    df1['similarity'] = 0.0
    df1 = df1.sort_values(by=['published'], ascending=True)

    for index, column in df1.iterrows():
        quote = str(column['domain']) + ' ' + str(column['title']) + ' ' + str(column['description'])
        md5 = hashlib.md5(quote.encode('utf-8')).hexdigest()
        df1.loc[index,'md5'] = md5
        day = '1970-01-01'
        if(len(str(column['published']))>5):
          pubDate = parser.parse(column['published'])
          day = pubDate.strftime('%Y-%m-%d')
         
        groupTxt = str(column['domain']) +  ' ' + day
        group = hashlib.md5(groupTxt.encode('utf-8')).hexdigest()  
        df1.loc[index,'group'] = group

    df1 = df1[~df1.md5.duplicated(keep='first')]  

    for index1, column1 in df1.iterrows():
        quote1 = str(column1['title']) + ' ' + str(column1['description']) 
        df2 = df1[df1['group']==column1['group']]
        for index2, column2 in df2.iterrows():
            if(column1['md5']>column2['md5']):
                quote2 = str(column2['title']) + ' ' + str(column2['description'])
                similarity = similar(quote1,quote2)
                if(similarity > df1.loc[index1,'similarity']):
                    df1.loc[index1,'similarity'] = similarity

    df3 = df1[df1['similarity']<0.8]
    df3 = df3.drop(columns=['md5', 'group', 'similarity'])
    df3 = df3.sort_values(by=['added','valid','published'], ascending=True)
    return df3


def archiveUrl(data):
    #timetravelDate = datetime.datetime.strptime(data['published'], '%Y-%m-%d %H:%M:%S').strftime('%Y%m%d')
    #pubDate = datetime.datetime.fromisoformat(data['published'])
    #pubDate = parser.isoparse(data['published'])
    timetravelDate = '19700101'
    pubDate = None
    try:
        pubDate = parser.parse(data['published'])
    except:
        print('date parse error 1')
    if(not pubDate):
      try:
        pubDate = parser.isoparse(data['published'])
      except:
        print('date parse error 2')   
    if(pubDate):
        timetravelDate = pubDate.strftime('%Y%m%d')
    #timetravelUrl = 'http://timetravel.mementoweb.org/api/json/'+timetravelDate+'/'+data['url']
    timetravelUrl = 'http://archive.org/wayback/available?url='+data['url']+'&timestamp='+timetravelDate
    try:
        print(["try request", timetravelUrl])
        page = requests.get(timetravelUrl, timeout=60)
        if page.status_code == 200:
            content = page.content
            print(content)
            if(content):
                #print(content)
                jsonData = json.loads(content)
                if(jsonData and ('archived_snapshots' in jsonData)):
                  snapshots = jsonData['archived_snapshots']
                  if('closest' in snapshots):
                    closest = snapshots['closest']
                    if('200'==closest['status']):
                      data['archive'] = closest['url']
                      if('1970-01-01T00:00:00' == data['published']):
                        ts = closest['timestamp']
                        tsNew = ts[0:4]+'-'+ts[4:6]+'-'+ts[6:8]+'T'+ts[8:10]+':'+ts[10:12]+':'+ts[12:14]
                        print(['new ts',ts,tsNew])
                        data['published'] = tsNew
    except:
#    except Exception as e:    
#    except json.decoder.JSONDecodeError as e:    
#    except requests.exceptions.RequestException as e:  
        e = sys.exc_info()[0]
        print("not archived yet")
        saveUrl = 'https://web.archive.org/save/' + data['url'] # archive.org
        #saveUrl = 'https://archive.is/submit/'
        #saveUrl = 'https://archive.ph/submit/'

        ##  pip3 install aiohttp
        try:
           loop = asyncio.get_event_loop()
           loop.run_until_complete(saveArchive(saveUrl))
        except:
           e2 = sys.exc_info()[0]
           print("something more went wrong (timeout/redirect/...)")            

        #async with aiohttp.ClientSession() as session:
        #    async with session.get(saveUrl) as response:
        #        print(await response.status())        
        '''
        try:
            page = requests.get(saveUrl, timeout=240)  # archive.org
            #page = requests.post(saveUrl, data = {'url':data['url']}, timeout=240)
            if page.status_code == 200:
                print('archived!')
        except requests.exceptions.RequestException as e2:
            print("not archivable: " + data['url'])
        '''    
    return data 

def extractData(article, language, keyWord, topic, feed, country, ipcc, continent):
    title = article['title']
    description = article['description']
    url = article['url']
    domain = None
    if(url):   
     #later use list...
     url = url.replace('https://www.zeit.de/zustimmung?url=', '')
     url = url.replace('%3A', ':')
     url = url.replace('%2F', '/')                
     domain = urlparse(url).netloc
    else:
     print(['no url', article])
    image = None
    if('urlToImage' in article): 
        image = article['urlToImage']

    published = '1970-01-01T00:00:00'
    if('publishedAt' in article):    
        published = article['publishedAt']
    content = article['content']
    hashStr = hashlib.sha256(url.encode()).hexdigest()[:32]
    data = {'url':url, 'valid':0, 'domain':domain,'published':published, 'description':description, 'title':title, 'added':str(dtNow), 'hash':hashStr,
            'image':image, 'content':content, 'quote':'', 'language': language, 'term':keyWord, 'topic':topic, 'feed':feed, 'country':country, 'ipcc':ipcc, 'continent':continent}
    return data  

def checkKeywordInQuote(keyword, quote, case=True, anyKey=False):
    keyword = keyword.replace("+","").replace("-","")
    keywords = keyword.strip("'").split(" ")
    if(not case):
        keywords = keyword.strip("'").lower().split(" ")
        quote = quote.lower()
    if(anyKey):
      allFound = False
      for keyw in keywords:
        allFound = allFound or (keyw in quote)    
    else:
      allFound = True
      for keyw in keywords:
        allFound = allFound and (keyw in quote)  

    return allFound

def checkArticlesForKeywords(articles, termsDF, seldomDF, language, keyWord, topic, feed, country, ipcc, continent):
    termsLangDF = termsDF[termsDF['language']==language]
    foundArticles = []
    for article in articles:
      data = extractData(article, language, keyWord, topic, feed, country, ipcc, continent)
      searchQuote = str(data['title']) + " " + str(data['description'])
      fullQuote = str(data['content'])
      foundKeywords = []
      foundColumns = []
      found = False
      valid = 0.1
      for index2, column2 in termsLangDF.iterrows(): 
         keyword = column2['term']
         if(keyword.strip("'") in searchQuote):
             foundKeywords.append(keyword) 
             foundColumns.append(column2) 
             found = True
             valid = max(valid,0.9)
         allFound = checkKeywordInQuote(keyword, searchQuote, case=True)
         if(allFound):
             foundKeywords.append(keyword) 
             foundColumns.append(column2) 
             found = True
             valid = max(valid,0.8)
         allFound = checkKeywordInQuote(keyword, searchQuote, case=False)
         if(allFound):
             foundKeywords.append(keyword) 
             foundColumns.append(column2) 
             found = True
             max(valid,0.7)
      # add seldom keywords twice if
      if(not seldomDF.empty):
       keywordsSeldomLangDF = seldomDF[seldomDF['language']==language]
       for index2, column2 in keywordsSeldomLangDF.iterrows(): 
         keyword = column2['term']
         allFound = checkKeywordInQuote(keyword, searchQuote, case=True) 
         if(allFound):
             foundKeywords.append(keyword) 
             foundColumns.append(column2) 
             found = True
      if(not found):
        for index2, column2 in termsLangDF.iterrows(): 
           allFound = checkKeywordInQuote(keyword, fullQuote, case=True)
           if(allFound):
             foundKeywords.append(keyword) 
             found = True
             max(valid,0.6) 
      if(not found):
        for index2, column2 in termsLangDF.iterrows(): 
           allFound = checkKeywordInQuote(keyword, fullQuote, case=True, anyKey=True)
           if(allFound):
             foundKeywords.append(keyword) 
             foundColumns.append(column2) 
             found = True
             max(valid,0.2) 
      data['valid'] = valid
      if(valid>0.15):
        foundKeywords.append(keyWord) 
        anyColumn = random.choice(foundColumns)
        data['term'] = anyColumn['term']
        data['country'] = anyColumn['country']
        data['ipcc'] = anyColumn['ipcc']
        data['continent'] = anyColumn['continent']
        data['feed'] = anyColumn['feed']
        data['topic'] = anyColumn['topic']
        foundArticles.append(data)
      else:
        data['term'] = keyWord
        #foundArticles.append(data)

    return foundArticles

def filterNewAndArchive(articles):
    global collectedNews
    newArticles = []
    startTime = time.time()
    for data in articles:
        ##data = extractData(article, language, keyWord) 
        if (dataIsNotBlocked(data)):
            pubDate = parser.parse(data['published'])
            fileDate = 'news_'+pubDate.strftime('%Y_%m')+'.csv'
            if(not fileDate in collectedNews):
                if(os.path.isfile(DATA_PATH / 'cxsv' / fileDate)):
                    df = pd.read_csv(DATA_PATH / 'cxsv' / fileDate, delimiter=',',index_col='index')
                    df = df[~df.index.duplicated(keep='first')]
                    collectedNews[fileDate] = df.to_dict('index')
                else:
                    collectedNews[fileDate] = {}
            if(not data['url'] in collectedNews[fileDate]):
              if(not checkDuplicates(collectedNews[fileDate], data)):
                data = archiveUrl(data)
                newArticles.append(data)
        if((time.time() - startTime) > 60*10):
            return newArticles        
    return newArticles

def getNewsFiles(state='harvest'):
    fileName = './cxsv/news_????_??.csv'
    if(state):
        fileName = './cxsv/news_'+state+'_????_??.csv'
    files = glob.glob(fileName)
    return files  

def getLatestFileAge():
    minAge = 1E6
    now = time.time()
    for fileName in getNewsFiles(state=None):
        print([os.path.getatime(fileName),os.path.getctime(fileName),os.path.getmtime(fileName)])
        modifyDate = os.path.getmtime(fileName)
        fileAge = now-modifyDate
        print(fileAge)
        if(fileAge<minAge):
            minAge = fileAge
    return minAge        


def inqRandomNews(maxCount=1):
  #global termsDF
  global termsDF3
  global unsearchedTerms
  global keywordsNewsDF2

  apiKey = os.getenv('NEWSAPI_KEY')
  if(apiKey == '1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7'): 
      print('Please set newsapi.org key in file: mysecrets.py');
      return None
  while(maxCount>0):
    maxCount -= 1
    rndKey = termsDF.sample()
    randomNumber = random.random()
    if(termsDF3.ratio.max()>0.77):
      randomNumber = 0.5   
    if(unsearchedTerms.ratio.max()>0.750):    #0.765(?)  #0.759:36;3; 0.758:55;21 , 0.757:83;47 , 0.75:215
      randomNumber = 0.1   
    ## randomNumber = 0.1 # unsearched first
    #randomNumber = 0.5 # succesors first
    #randomNumber = 0.7
    #randomNumber = 0.99 # seldoms first

    print(['randomNumber: ',randomNumber])
    if(not keywordsNewsDF2.empty):
      if(randomNumber>0.85):
        print("DF2 seldoms")
        rndKey = keywordsNewsDF2.sample()
      if(randomNumber>0.95):
        print("DF2 last")
        rndKey = keywordsNewsDF2.head(1).sample()
    if(not termsDF3.empty):
      if(randomNumber<0.75): 
        print("DF3 successors")
        rndKey = termsDF3.sample()
      if(randomNumber<0.55):
        print("DF3 first")
        rndKey = termsDF3.head(1).sample()
    if(not unsearchedTerms.empty):
      if(randomNumber<0.45):
        print("unsearched")
        rndKey = unsearchedTerms.sample()
      if(randomNumber<0.2):
        print("unsearched first")
        rndKey = unsearchedTerms.head(1).sample()
    #if FoundAny: newLimit = minimum(currPage+1,limitPage)
    #if foundNothing:  newLimit = maximum(1,random.choice(range(currPage-1,limitPage-1)))

    ## cheat for now!     
    ### keywordEmptyDF = termsDF[termsDF['keyword']=="'Eva Kaili'"]
    ### rndKey = keywordEmptyDF.sample()
    ## rm in final version
    ### rndKey = termsDF3.head(1).sample()

    #keyWord = random.choice(searchWords)
    #language = 'de'
    #language = 'en'   
    #language = 'fr' 
    ##  crc = rndKey['crc'].iloc[0]
    crc = rndKey['index'].iloc[0]
    keyWord = rndKey['term'].iloc[0]
    feed = rndKey['feed'].iloc[0]
    topic = rndKey['topic'].iloc[0]
    country = rndKey['country'].iloc[0]
    ipcc = rndKey['ipcc'].iloc[0]
    continent = rndKey['continent'].iloc[0]
    language = rndKey['language'].iloc[0]
    limitPages = int(round(rndKey['pages'].iloc[0]))
    ratioNew = rndKey['ratio'].iloc[0]
    currPage = random.choice(range(1,limitPages+1))  
    newLimit = max(1,random.choice(range(currPage-1,limitPages)))
    newCounter = 1+rndKey['counter'].iloc[0]
    currRatio = ratioNew
          
    print([keyWord, language, 'P:' ,limitPages, 'R:' ,ratioNew, 'C:' ,newCounter])
    if(not 'xx'==language):
        nLang = language
        if('ja'==nLang):
          nLang = 'jp'
        sort = random.choice(['relevancy', 'popularity', 'publishedAt'])
        pageSize = 33
        print('keyword: '+keyWord+'; Page: '+str(currPage))
        # https://newsapi.org/docs/endpoints/everything
        url = ('https://newsapi.org/v2/everything?'+
            #"q='"+keyWord+"'&"
            "q="+keyWord+"&"
            'pageSize='+str(pageSize)+'&'
            'language='+language+'&'
            'page='+str(currPage)+'&'
            'sortBy='+sort+'&'
            'apiKey='+apiKey
            #'excludeDomains=www.zeit.de,www.reuters.com'
            )
            
            # sortBy=relevancy   : relevancy, popularity, publishedAt
        response = requests.get(url)
        response.encoding = response.apparent_encoding
        
        foundNew = False
        if(response.text):
            jsonData = json.loads(response.text)
            if ('ok'==jsonData['status']):
             currRatio = 0
             if(jsonData['totalResults']>0):
              currRatio = jsonData['totalResults']/1E7
              if(len(jsonData['articles']) > 0):
                #print(['found it', len(jsonData['articles']), maxCount])
                maxCount = 0
                currRatio += len(jsonData['articles'])/1E3
                deltaLimit = 0
                #newLimit = limitPages
                if(len(jsonData['articles']) > 20):
                  deltaLimit += 1  
                  #newLimit = max(currPage+1,limitPages)                
                print('#found Articles: '+str(len(jsonData['articles'])))
                checkedArticles = checkArticlesForKeywords(jsonData['articles'], termsDF, keywordsNewsDF2,language, keyWord, topic, feed, country, ipcc, continent)
                print('#checked Articles: '+str(len(checkedArticles)))
                print("archive first")
                newArticles = filterNewAndArchive(checkedArticles)
                print('#new Articles: '+str(len(newArticles)))
                 
                if(len(newArticles) in [1,2]):     
                    print("sleep")   
                    #time.sleep(60)
                    time.sleep(10)
                print("add to collection")


                currRatio += len(newArticles)/len(jsonData['articles'])
                if(currRatio>0.5):
                    deltaLimit += 1
                    #newLimit = max(currPage+2,limitPages)
                newLimit =  min(3,max(currPage+deltaLimit,limitPages))
                print(['currRatio',currRatio,'currPage: ',currPage,' limitPages: ',limitPages,' deltaLimit: ',deltaLimit,' new Limit: ', newLimit])  

                for data in newArticles:
                    if (dataIsNotBlocked(data)):                    
                        #print(str(keyWord)+': '+str(title)+' '+str(url))
                        print(["addNewsToCollection: ",data])
                        if(addNewsToCollection(data)):
                            foundNew = True
                            print(["+++added"])  
                        else:
                            print(["---not added"])    
                #print(["collectedNews: ",collectedNews])            
                if(foundNew):     
                    maxCount  -= (2 + len(newArticles))  
                    storeCollection()
            else:
              print(response.text)
              if(jsonData['code'] == 'maximumResultsReached'):
                deltaLimit = -1
                maxCount = 0
                newLimit =  max(1,currPage+deltaLimit)
              # {"status":"error","code":"maximumResultsReached","message":"You have requested too many results. Developer accounts are limited to a max of 100 results. You are trying to request results 100 to 150. Please upgrade to a paid plan if you need more results."}
    #print(rndKey.index)
    #termsDF.at[rndKey.index, 'limitPages'] = newLimit    
    termsDF.loc[termsDF['index'] == crc, 'pages'] = newLimit 
    termsDF.loc[termsDF['index'] == crc, 'counter'] = newCounter 
    termsDF.loc[termsDF['index'] == crc, 'ratio'] = currRatio*0.15+ratioNew*0.85

    unsearchedTerms.loc[unsearchedTerms['index'] == crc, 'unsearched'] = -1E9
    unsearchedTerms.loc[unsearchedTerms['index'] == crc, 'ratio'] = -1E9
    unsearchedTerms = unsearchedTerms.sort_values(by=['unsearched'], ascending=False) 

    #    if(not termsDF3.empty):
    termsDF3.loc[termsDF3['index'] == crc, 'ratio'] = -1E9
    termsDF3 = termsDF3.sort_values(by=['ratio'], ascending=False) 

    if(not keywordsNewsDF2.empty):
      keywordsNewsDF2.loc[keywordsNewsDF2['index'] == crc, 'ratio'] = -1E9
      keywordsNewsDF2 = keywordsNewsDF2.sort_values(by=['ratio'], ascending=False) 

    print(['xxx','crc',crc,'currRatio',currRatio,'ratioNew',ratioNew,'currPage: ',currPage,' limitPages: ',limitPages,' new Limit: ', newLimit])  

        
      

#b'{"status":"ok","totalResults":1504,
# "articles":[{"source":{"id":null,"name":"heise online"},
#              "author":"Stefan Krempl",
#              "title":"Wissenschaftler: Klimawandel tr\xc3\xa4gt zum Starkregen bei\xe2\x80\x8b",
#              "description":"Die Wolkenbr\xc3\xbcche mit katastrophalen Folgen geh\xc3\xb6ren so nicht mehr zur \xc3\xbcblichen Wetter-Varianz, meinen Wissenschaftler. Sie fordern einen Umbau der Infrastruktur.",
#              "url":"https://www.heise.de/news/Wissenschaftler-Klimawandel-traegt-zum-Starkregen-bei-6140856.html",
#              "urlToImage":"https://heise.cloudimg.io/bound/1200x1200/q85.png-lossy-85.webp-lossy-85.foil1/_www-heise-de_/imgs/18/3/1/3/9/7/8/0/Ueberschwemmung-c06f751f2932e14b.jpeg",
#              "publishedAt":"2021-07-16T15:06:00Z",
#              "content":"Nach den langen und heftigen Regenf\xc3\xa4llen Mitte der Woche treten immer mehr katastrophale Folgen zutage: Die Zahl der Toten w\xc3\xa4chst, allein in Rheinland-Pfalz und Nordrhein-Westfalen sind \xc3\xbcber 100 Mens\xe2\x80\xa6 [+4840 chars]"}


amount = 4
if(len(termsDF)>50):
  amount = 8
if(len(termsDF)>150):
  amount = 16
inqRandomNews(amount)



termsDF['age'] = termsDF['created'].apply(
  lambda x: 
      getAge(x)
)

termsDF['magick'] = (20-termsDF['age'])/60+termsDF['ratio']/3+(8-termsDF['counter'])/24+termsDF['pages']/10
termsDF = termsDF[(termsDF.magick>0.25)]

#termsDF = termsDF.sort_values(by=['topic','keyword'])
termsDF = termsDF.sort_values(by=['ratio'], ascending=False)
print('9999999999999999999999999999')

print(termsDF)
termsDF.to_csv(DATA_PATH / 'terms.csv', columns=termsFields,index=False)  


