import mysecrets

import os
import sys
import codecs
import io
import requests
import pandas as pd
from pathlib import Path
import datetime
#from datetime import timezone
from dateutil import parser
from datetime import date, timedelta, datetime, timezone

from CreGeoReference.GeoReference import GeoReference
from CreLanguageTranslate.LanguageTranslate import LanguageTranslate 

#from deep_translator import GoogleTranslator

DATA_PATH = Path.cwd()

topicColors = {'Thunderstorm':'#53785a', 'Flood':'#030ba1', 'Storm':'#b3b2b1', 'Storm Surge':'#834fa1', 'Flash Flood':'#02b5b8',
               'Tsunami':'#690191', 'Drought':'#edc291', 'Earthquake':'#870007', 'Landslide':'#572c03', 'Cold Wave':'#a7e9fa', 'Heat Wave':'#c40202',
               'Tropical Cyclone':'#4f4f4f', 'Volcano':'#b83202', 'Snow Avalanche':'#deddd5', 'unknown':'#d60d2b'  
               }

topicsFields = ["module", "topic", "feed", "term", "created", "added", "language", "ratio", "location", "latitude", "longitude"]
keywordsFields = ["keyword","language","topic","topicColor","keywordColor","limitPages","ratioNew"]
termsFields = ["index","module", "topic", "color", "feed", "term", "created", "country", "ratio", "counter", "pages", "language", "ipcc", "continent"]
topicsDict = {}

# MOVE TO myparameters.py (no secrets only)
MAX_IMPORTS = 10

TARGET_LANGUAGE = os.getenv('EXTREME_LANGUAGE')
if(TARGET_LANGUAGE == 'xx'): 
   print('Please set EXTREME_LANGUAGE in file: mysecrets.py');
   sys.exit("Please set EXTREME_LANGUAGE in file: mysecrets.py")

lt = LanguageTranslate()

def doTranslate(column, targetLanguage, tmpTopic = True, lowerCase=True):
  tmpSource = column['term']
  if(tmpTopic):
    tmpSource = column['topic']+ ': ' + column['term']
  if(lowerCase):
    tmpSource = tmpSource.lower()
  tmpTerm = lt.getTranslatorByLanguage(column['language'],targetLanguage).translate(tmpSource)
  if (not isinstance(tmpTerm, str)):
    print(['translation failed',tmpSource,tmpTerm])
    tmpTerm = '' 
  if(tmpTopic):
    tmpArray = []
    if(':' in tmpTerm):
      tmpArray = tmpTerm.split(':', 1)
    if('：' in tmpTerm):
      tmpArray = tmpTerm.split('：', 1) 
    if(len(tmpArray)>1):
       tmpTerm = tmpArray[1]
  return tmpTerm.strip()

def importTerms(maxImports=10, targetLanguage='de'):
    termsDF = pd.DataFrame(None)
    gf = GeoReference(local=True)

    countriesForLanguage = gf.getCountriesNameByLanguage(targetLanguage)
    print(countriesForLanguage)
    if(os.path.isfile(DATA_PATH / 'terms.csv')):
      termsDF = pd.read_csv(DATA_PATH / 'terms.csv', delimiter=',')  #,index_col='keyword'
      termsDF = termsDF.sort_values(by=['ratio'], ascending=False)  

    ##print(termsDF.head())

    ghToken = os.getenv('EXTREME_GH_TOKEN')
    if(ghToken == 'ghp_1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f'): 
        print('Please set EXTREME_GH_TOKEN in file: mysecrets.py');
        return None

    stream=requests.get('https://raw.githubusercontent.com/newsWhisperer/extremes/main/csv/topics.csv', headers={"Authorization":"token "+ghToken}).content
    topicsDF=pd.read_csv(io.StringIO(stream.decode('utf-8')), delimiter=',')

    print(topicsDF)

    topicsDF['ratio'] *= topicsDF['country'].apply(
      lambda x: max(0.5,countriesForLanguage[x]) if (x in countriesForLanguage) else 0.4 
    )
    topicsDF = topicsDF.sort_values(by=['ratio'], ascending=False) 
    topicsDF['pages'] = 1
    topicsDF['counter'] = 0

    topicsDF['color'] = topicsDF['topic'].apply(
      lambda x: topicColors[x] if (x in topicColors) else topicColors['unknown'] 
    )
    topicsDF['created'] = topicsDF['added']

    ##print(topicsDF.head())

    countingImports = 0
    for index, column in topicsDF.iterrows():
      if(termsDF.empty or not column['index'] in list(termsDF['index'])):
        if(countingImports<maxImports):
          if(not column['language'] == targetLanguage):
            #column['term'] = GoogleTranslator(source=column['language'], target=targetLanguage).translate(text=column['term'])
            column['term'] = doTranslate(column, targetLanguage) #use context of topic
          column['language'] = targetLanguage
          if(len(column['term'])>1): 
            #termsDF = termsDF.append(column, ignore_index=True)
            columnDF = pd.DataFrame.from_records([column], columns=list(column.keys()))
            #print(columnDF)
            if(termsDF.empty): 
              termsDF = columnDF
            else:
              termsDF = pd.concat([termsDF,columnDF])
          #print(termsDF)

          countingImports += 1



    termsDF = termsDF.sort_values(by=['ratio'], ascending=False)
    ##print(termsDF.head())
    termsDF.to_csv(DATA_PATH / 'terms.csv', columns=termsFields,index=False)  
 

importTerms(MAX_IMPORTS, TARGET_LANGUAGE)
