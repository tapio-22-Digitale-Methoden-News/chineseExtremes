import mysecrets
import pandas as pd

from pathlib import Path
import os.path
import io
#import requests
import glob

import time
import datetime
from dateutil import parser

# pip3 install spacy
# python3 -m spacy download en_core_web_md
#pip3 install textblob
import requests
import json
import geocoder
import geopandas

import nltk
import spacy
import en_core_web_md
from textblob import TextBlob

nlp = en_core_web_md.load()
nltk.download('punkt_tab')
nltk.download('punkt')


DATA_PATH = Path.cwd()
if(not os.path.exists(DATA_PATH / 'csv')):
    os.mkdir(DATA_PATH / 'csv')

ipccRegions = geopandas.read_file('https://github.com/creDocker/creAssets/blob/main/cre/versions/u24.04/assets/public/ipcc/IPCC-WGI-reference-regions-v4.geojson?raw=true')

countriesInfo = pd.read_csv("https://github.com/creDocker/creAssets/blob/main/cre/versions/u24.04/assets/public/geonames/countryInfo.csv?raw=true")
countriesGeo = geopandas.read_file('https://raw.githubusercontent.com/creDocker/creAssets/refs/heads/main/cre/versions/u24.04/assets/public/geonames/shapes_countries.json')
countriesGeo['geoNameId'] = countriesGeo['geoNameId'].astype(int)
countriesInfo['geonameid'] = countriesInfo['geonameid'].astype(int)
countriesDf = pd.merge(countriesGeo, countriesInfo, left_on='geoNameId', right_on='geonameid')

def getNewsFiles():
    fileName = './cxsv/news_????_??.csv'
    files = glob.glob(fileName)
    return files  

def getNewsDFbyList(files):    
    newsDF = pd.DataFrame(None)
    for file in files:
        df = pd.read_csv(file, delimiter=',')
        if(newsDF.empty):
            newsDF = df
        else:
            newsDF = pd.concat([newsDF, df])
    newsDF = newsDF.sort_values(by=['published'], ascending=True)        
    return newsDF 

def getNewsDF():
    files = getNewsFiles()
    newsDF = getNewsDFbyList(files)
    return newsDF         

'''
keywordsDF = pd.read_csv(DATA_PATH / 'keywords.csv', delimiter=',')
keywordsDF = keywordsDF.drop(columns = ['language'])
'''

oldLocationsDf = pd.DataFrame(None)
if(os.path.exists(DATA_PATH / 'csv' / 'sentiments_locations.csv')):
  oldLocationsDf = pd.read_csv(DATA_PATH / 'csv' / 'sentiments_locations.csv', delimiter=',')

newsDf = getNewsDF()
print(newsDf)   

## keywordsNewsDF = pd.merge(keywordsDF, newsDf, how='left', left_on=['keyword'], right_on=['keyword'])
## print(keywordsNewsDF)  

newsDf['subjectivity'] = 0.0
newsDf['sentiment'] = 0.0
newsDf['count'] = 1.0
newsDf['week'] = '0000-00'
newsDf['day'] = '000-00-00'

i=0
##topicWordsAbs = {'summaryOfAllWords': emptyTopics.copy()}
for index, column in newsDf.iterrows():
    i += 1
    if(i % 50 == 0):
        print(i)
    quote = str(column['en'])
    blob = TextBlob(quote)
    newsDf.loc[newsDf['url'] == column['url'], 'subjectivity'] = blob.sentiment.subjectivity
    newsDf.loc[newsDf['url'] == column['url'], 'sentiment'] = blob.sentiment.polarity
    try:
      pubDate = parser.parse(column['published'])
      newsDf.loc[newsDf['url'] == column['url'], 'week'] = pubDate.strftime('%Y-%W')
      newsDf.loc[newsDf['url'] == column['url'], 'day'] = pubDate.strftime('%Y-%m-%d')
    except:
      print('date parse error')

##keywordsNewsDF = newsDf.groupby('keyword').mean()

def groupSentiments(df, aggColumn):
	cols = [aggColumn,'sentiment_mean','sentiment_std','subjectivity_mean','subjectivity_std','counting']
	groupDF = df.groupby([aggColumn], as_index=False).agg(
		              {'sentiment':['mean','std'],'subjectivity':['mean','std'],'count':'sum'})
	groupDF.columns = cols
	groupDF.reindex(columns=sorted(groupDF.columns))
	groupDF = groupDF.sort_values(by=['counting'], ascending=False)
	groupDF['sentiment_std'] = groupDF['sentiment_std'].fillna(1)
	groupDF['subjectivity_std'] = groupDF['subjectivity_std'].fillna(1)
	return groupDF 

domainDF = groupSentiments(newsDf, 'domain')
domainDF.loc[domainDF['counting'] < 2, 'sentiment_mean'] = 0.0
domainDF.loc[domainDF['counting'] < 2, 'subjectivity_mean'] = 0.0
print(domainDF)
cols = ['domain','sentiment_mean','sentiment_std','subjectivity_mean','subjectivity_std','counting']
domainDF.to_csv(DATA_PATH / 'csv' / 'sentiments_domains.csv', columns=cols,index=False) 

objNewsDF = pd.merge(newsDf, domainDF, how='left', left_on=['domain'], right_on=['domain'])
objNewsDF['subjectivity'] = (objNewsDF['subjectivity'] - objNewsDF['subjectivity_mean'])/objNewsDF['subjectivity_std']
objNewsDF['sentiment'] = (objNewsDF['sentiment'] - objNewsDF['sentiment_mean'])/objNewsDF['sentiment_std']
print(objNewsDF)  

weeksDF =  groupSentiments(objNewsDF, 'week')
weeksDF = weeksDF.sort_values(by=['week'], ascending=True)
weeksDF.to_csv(DATA_PATH / 'csv' / 'sentiments_weeks.csv',index=False) 

daysDF =  groupSentiments(objNewsDF, 'day')
daysDF = daysDF.sort_values(by=['day'], ascending=True)
daysDF.to_csv(DATA_PATH / 'csv' / 'sentiments_days.csv',index=False) 

'''
keywordsSentimentDF =  groupSentiments(objNewsDF, 'keyword')
keywordsSentimentDF = keywordsSentimentDF.sort_values(by=['keyword'], ascending=True)
keywordsSentimentDF.to_csv(DATA_PATH / 'csv' / 'sentiments_keywords.csv',index=False) 
'''
#todo: group by extreme

print(list(newsDf.columns))
print(list(objNewsDF.columns))
##print(list(keywordsDF.columns))
'''
topicNewsDF = pd.merge(objNewsDF, keywordsDF, how='left', left_on=['keyword'], right_on=['keyword'])
print(list(topicNewsDF.columns))
topicsDF =  groupSentiments(topicNewsDF, 'topic')
topicsDF = topicsDF.sort_values(by=['topic'], ascending=True)
topicsDF.to_csv(DATA_PATH / 'csv' / 'sentiments_topics.csv',index=False) 
'''


emptyDict = {'count':0,'sentiment':0,'subjectivity':0}
indexLocations = {}
indexOrganizations = {}
indexPersons = {}
indexMisc = {}
indexMissing = {}

foundGeonames = False
geonamesKey = 'GEONAMES_KEY'
geonamesKey = os.getenv('GEONAMES_KEY')
if(geonamesKey):
    foundGeonames = True
if(geonamesKey == '1a2b3c4d5'): 
    print('Please set geonames.org key in file: secrets.py');
    foundGeonames = False
if(geonamesKey == 'demo_demo_123'): 
    print('Please set geonames.org key in file: secrets.py');
    foundGeonames = False
print(['foundGeonames',foundGeonames])
#foundGeonames = True

def inqGeonamesByNameAndLanguage(df, index, phrase, lang):
    gn = geocoder.geonames(phrase, lang=lang, key=geonamesKey)
    print([phrase,gn,gn.geonames_id]) 
    if(gn.geonames_id):  
      df.loc[index,'geonames'] = int(gn.geonames_id)
      df.loc[index,'latitude'] = float(gn.lat)
      df.loc[index,'longitude'] = float(gn.lng)
      df.loc[index,'geotype'] = gn.feature_class
      ##df.loc[index,'country'] = gn.country  #localized!
      gne = geocoder.geonames(phrase, lang=lang, key=geonamesKey)
      if(gne.country):
        df.loc[index,'country'] = gne.country
        print(gne.country)
        print(['geo',gn.lat,gn.lng, gn.geonames_id, gn])
        return (df, float(gn.lat),float(gn.lng), int(gn.geonames_id))
    return (df, None, None, None)


def getCoordsByLatAndLng(lat,lng, phrase='somewhere'):
    coordinates = geopandas.points_from_xy([lng], [lat])
    print(['points_from_xy',coordinates])
    Coords = geopandas.GeoDataFrame({
      'geometry': coordinates,
      'name': [phrase]
    }, crs={'init': 'epsg:4326', 'no_defs': True})
    print(['GeoDataFrame',Coords]) 
    return Coords

def getIpccByCoords(df, index, Coords):
    whichIpcc = geopandas.sjoin(ipccRegions, Coords, how='inner', op='intersects')
    print(whichIpcc)
    if(not whichIpcc.empty):
      print(['Acronym & Continent',list(whichIpcc['Acronym'])[0],list(whichIpcc['Continent'])[0]])
      df.loc[index,'ipcc'] = list(whichIpcc['Acronym'])[0]
      df.loc[index,'continent'] = list(whichIpcc['Continent'])[0]
    return df        

def getCountryByCoords(df, index, Coords):
    whichCountry = geopandas.sjoin(countriesDf, Coords, how='inner', op='intersects')
    print(whichCountry)
    if(not whichCountry.empty):
      print(['country',list(whichCountry['Country'])[0]])
      df.loc[index,'country'] = list(whichCountry['Country'])[0]
    return df

geomax = 250
def enrichFromGeonames(df):
    global geomax
    print('Starting with geonames')
    if(not foundGeonames):
        print('geonames not found')
        #search by GND only
        for index, column in df.iterrows():
          phrase = str(column.phrase)
          gnd = searchGndWithCoordsAndGeonamesByName(phrase)
          print([phrase,gnd])
          if(gnd and 'gndId' in gnd):
            df.loc[index,'gnd'] = str(gnd['gndId'])
            df.loc[index,'geonames'] = int(gnd['geonames'])
            df.loc[index,'latitude'] = float(gnd['latitude'])
            df.loc[index,'longitude'] = float(gnd['longitude'])
            df.loc[index,'geotype'] = 'X'
            Coords = getCoordsByLatAndLng(float(gnd['latitude']),float(gnd['longitude']), phrase)
            df = getIpccByCoords(df, index, Coords)
            df = getCountryByCoords(df, index, Coords)
        return df
    for index, column in df.iterrows():
      if(geomax>0):
        lang = str(column.language)
        phrase = str(column.phrase)
        if(str(column.geonames) == '-1'):
          print('things to do')
          (df, lat, lng, gnId) = inqGeonamesByNameAndLanguage(df, index, phrase, lang)
          if(gnId):  
            Coords = getCoordsByLatAndLng(lat,lng, phrase)
            df = getIpccByCoords(df, index, Coords)
            df = getCountryByCoords(df, index, Coords)

            #get GND
            found = False 
            gnd = searchGndByGeonamesId(gnId)
            print(['searchGndByGeonamesId',gnd]) 
            if(gnd and 'gndId' in gnd):
              df.loc[index,'gnd'] = str(gnd['gndId'])
              found = True
            if(not found):
              gnd = searchGndByNameAndGeo(phrase, lat, lng)
              print(['searchGndByNameAndGeo',gnd]) 
              if(gnd and 'gndId' in gnd):
                df.loc[index,'gnd'] = str(gnd['gndId'])
                found = True
            if(not found):
              gnd = searchGndByName(phrase)
              print(['searchGndByName',gnd]) 
              if(gnd and 'gndId' in gnd):
                df.loc[index,'gnd'] = str(gnd['gndId'])
                found = True

          else:
            print(['geonames found nothing',phrase])
            df.loc[index,'geonames'] = 0
            ### Try GND only
            gnd = searchGndWithCoordsAndGeonamesByName(phrase)  
            print([phrase,gnd])
            if(gnd and 'gndId' in gnd):
              df.loc[index,'gnd'] = str(gnd['gndId'])
              df.loc[index,'geonames'] = int(gnd['geonames'])
              df.loc[index,'latitude'] = float(gnd['latitude'])
              df.loc[index,'longitude'] = float(gnd['longitude'])
              df.loc[index,'geotype'] = 'X' 
              Coords = getCoordsByLatAndLng(float(gnd['latitude']),float(gnd['longitude']), phrase)
              df = getIpccByCoords(df, index, Coords)
              df = getCountryByCoords(df, index, Coords)
          geomax -= 1
          time.sleep(0.1) 
    return df

def searchGndByGeonamesId(geonamesId):
    gndurl = 'https://lobid.org/gnd/search?q='+str(geonamesId)+'&filter=type%3APlaceOrGeographicName&format=json'   #hasGeometry
    page = requests.get(gndurl, timeout=60)
    if page.status_code == 200:
      content = page.content
      #print(content)
      if(content):
        #print(content)
        jsonData = json.loads(content)
        #print(jsonData)      #'variantName' !
        if('member' in jsonData):
          for member in jsonData['member']:
           if('sameAs' in member):
             for same in member['sameAs']:
               #print(25*"##")
               #print(same)
               if('id' in same):
                 if(same['id']=="https://sws.geonames.org/"+str(geonamesId)):
                   if('gndIdentifier' in member):
                     result = {'gndId':member['gndIdentifier']} 
                     #print(member['gndIdentifier']) 
                     #print(25*"=*")
                     #print(member)  
                     if('hasGeometry' in member):
                       #print(member['hasGeometry']) 
                       latitude = None
                       longitude = None
                       for geo in member['hasGeometry']:  
                         if('asWKT' in geo and 'type' in geo and geo['type']=='Point'):
                            point = geo['asWKT'][0]
                            point = point.replace('Point ','').strip().strip('()').strip()
                            #print(point)
                            coords = point.split(" ")
                            #print(coords)
                            result['longitude'] = float(coords[0])
                            result['latitude'] = float(coords[1])
                     if('variantName' in member):
                       #print(member['variantName']) 
                       result['variantNames'] = member['variantName']  
                     if('preferredName' in member):
                       #print(member['preferredName'])
                       result['preferredName'] = member['preferredName']
                     return result
    return None

def searchGndByNameAndGeo(locationName, latitude, longitude, maxDistance=10):
    gndUrl = 'https://explore.gnd.network/search?term='+locationName+'&f.satzart=Geografikum&rows=1'
    gndurl = 'https://lobid.org/gnd/search?q='+locationName+'&filter=type%3APlaceOrGeographicName&format=json'   #hasGeometry
    page = requests.get(gndurl, timeout=60)
    if page.status_code == 200:
      content = page.content
      #print(content)
      if(content):
        #print(content)
        jsonData = json.loads(content)
        #print(jsonData)      #'variantName' !
        if('member' in jsonData):
          minDistance2 = 10E9
          result = None
          for member in jsonData['member']:
           #print(25*"=*")
           #print(member)  
           if('hasGeometry' in member):
            #print(member['hasGeometry']) 
            for geo in member['hasGeometry']: 
             if('asWKT' in geo and 'type' in geo and geo['type']=='Point'):
               point = geo['asWKT'][0]
               point = point.replace('Point ','').strip().strip('()').strip()
               #print(point)
               coords = point.split(" ")
               #print(coords)
               currLongitude = float(coords[0])
               currLatitude = float(coords[1])
               distance2 = (currLongitude-longitude)**2+(currLatitude-latitude)**2
               #print(distance2)
               if(distance2<minDistance2):
                 minDistance = distance2 
                 if('gndIdentifier' in member):
                   #print(member['gndIdentifier']) 
                   result = {'longitude':currLongitude, 'latitude':currLatitude, 'distance':distance2**0.5}
                   result['gndId'] = member['gndIdentifier']
                   if('preferredName' in member):
                     #print(member['preferredName']) 
                     result['preferredName'] = member['preferredName']
          #print(result)
          if(minDistance2<maxDistance**2):
            return result
        return None                   

def searchGndByName(locationName):
    gndUrl = 'https://explore.gnd.network/search?term='+locationName+'&f.satzart=Geografikum&rows=1'
    gndurl = 'https://lobid.org/gnd/search?q='+locationName+'&filter=type%3APlaceOrGeographicName&format=json'   #hasGeometry
    page = requests.get(gndurl, timeout=60)
    if page.status_code == 200:
      content = page.content
      #print(content)
      if(content):
        #print(content)
        jsonData = json.loads(content)
        #print(jsonData)      #'variantName' !
        if('member' in jsonData):
          for member in jsonData['member']:
           #print(25*"=*")
           #print(member)  
           if('gndIdentifier' in member):
             #print(member['gndIdentifier']) 
             result = {'gndId':member['gndIdentifier']} 
             if('hasGeometry' in member):
               #print(member['hasGeometry']) 
               latitude = None
               longitude = None
               for geo in member['hasGeometry']:  
                 if('asWKT' in geo and 'type' in geo and geo['type']=='Point'):
                    point = geo['asWKT'][0]
                    point = point.replace('Point ','').strip().strip('()').strip()
                    #print(point)
                    coords = point.split(" ")
                    #print(coords)
                    result['longitude'] = float(coords[0])
                    result['latitude'] = float(coords[1])
             found = False
             if('variantName' in member):
               #print(member['variantName']) 
               result['variantNames'] = member['variantName']  
               found = locationName in member['variantName'] 
             if('preferredName' in member):
               #print(member['preferredName'])
               result['preferredName'] = member['preferredName']
               found = found or (member['preferredName'] == locationName)
             if(found): 
               return result
    return None

def searchGndWithCoordsAndGeonamesByName(locationName):
    gndUrl = 'https://explore.gnd.network/search?term='+locationName+'&f.satzart=Geografikum&rows=1'
    gndurl = 'https://lobid.org/gnd/search?q='+locationName+'&filter=type%3APlaceOrGeographicName&format=json'   #hasGeometry
    page = requests.get(gndurl, timeout=60)
    if page.status_code == 200:
      content = page.content
      #print(content)
      if(content):
        #print(content)
        jsonData = json.loads(content)
        #print(jsonData)      #'variantName' !
        if('member' in jsonData):
          for member in jsonData['member']:
           #print(25*"=*")
           #print(member)  
           if('gndIdentifier' in member):
             #print(member['gndIdentifier']) 
             result = {'gndId':member['gndIdentifier']} 

             geonameFound = False 
             if('sameAs' in member):
               for same in member['sameAs']:
                 #print(25*"##")
                 #print(same)
                 if('id' in same):
                   if('https://sws.geonames.org/' in same['id']):
                     geonameFound = True
                     geonamesId = same['id'].replace('https://sws.geonames.org/','')
                     result['geonames'] = int(geonamesId) 
             geoFound = False
             if('hasGeometry' in member):
               #print(member['hasGeometry']) 
               latitude = None
               longitude = None
               for geo in member['hasGeometry']:  
                 if('asWKT' in geo and 'type' in geo and geo['type']=='Point'):
                    point = geo['asWKT'][0]
                    point = point.replace('Point ','').strip().strip('()').strip()
                    #print(point)
                    coords = point.split(" ")
                    #print(coords)
                    result['longitude'] = float(coords[0])
                    result['latitude'] = float(coords[1])
                    geoFound = True
             found = False
             if('variantName' in member):
               #print(member['variantName']) 
               result['variantNames'] = member['variantName']  
               found = locationName in member['variantName'] 
             if('preferredName' in member):
               #print(member['preferredName'])
               result['preferredName'] = member['preferredName']
               found = found or (member['preferredName'] == locationName)
             if(found and geoFound and geonameFound): 
               return result
    return None


def strangeCharacters(testString, testCharacters):
     count = 0
     for oneCharacter in testCharacters:
          count += testString.count(oneCharacter)
     return count

i=0
##topicWordsAbs = {'summaryOfAllWords': emptyTopics.copy()}
for index, column in objNewsDF.iterrows():
    i += 1
    if(i % 50 == 0):
        print(i)
    quote = str(column['en'])
    lang = 'en' #NOT column.language !
    #quote = str(column.title)+'. ' +str(column.description)
    blob = TextBlob(quote)
    for sentence in blob.sentences:
        #sentence.sentiment.polarity
        doc = nlp(str(sentence))
        for entity in doc.ents:

            if(entity.label_ in ['LOC','GPE']):
                if(entity.text in indexLocations):
                    #indexLocations[entity.text]['count'] += 1   
                    indexLocations[entity.text]['count'] += column.valid   
                    indexLocations[entity.text]['sentiment'] += sentence.sentiment.polarity
                    indexLocations[entity.text]['subjectivity'] += sentence.sentiment.subjectivity
                else:      
                    indexLocations[entity.text] = {'phrase':entity.text, 'label':entity.label_, 'sentiment':sentence.sentiment.polarity,
                                                   'subjectivity':sentence.sentiment.subjectivity, 'language':lang, 'count':1, 
                                                   'gnd':None, 'geonames':-1, 'geotype':None, 'latitude':None, 'longitude':None, 
                                                   'continent':None, 'country':None, 'ipcc':None}
                    if ('geonames' in oldLocationsDf.columns):
                      foundInOlDf = oldLocationsDf[oldLocationsDf['phrase']==entity.text]
                      foundInOlDf = foundInOlDf[foundInOlDf['geonames']>-0.5]
                      if(not foundInOlDf.empty):
                        indexLocations[entity.text]['geonames'] = int(foundInOlDf['geonames'].median())
                        if (foundInOlDf['geonames'].median()>0):
                          indexLocations[entity.text]['geotype'] = foundInOlDf['geotype'].min()
                          indexLocations[entity.text]['latitude'] = float(foundInOlDf['latitude'].mean())
                          indexLocations[entity.text]['longitude'] = float(foundInOlDf['longitude'].mean())
                          indexLocations[entity.text]['country'] = foundInOlDf['country'].min()
                          indexLocations[entity.text]['ipcc'] = foundInOlDf['ipcc'].min()
                          if('continent' in foundInOlDf.columns):
                            indexLocations[entity.text]['continent'] = foundInOlDf['continent'].min()
                          if('gnd' in foundInOlDf.columns):
                            indexLocations[entity.text]['gnd'] = foundInOlDf['gnd'].min()


            elif(entity.label_ in ['PER','PERSON']):
             personText = entity.text
             personText = personText.strip(" .,!?;:'…/-").strip('"')
             if(strangeCharacters(personText,".,!?;:'…<>/\n\r")==0):
               if(personText.count(' ')>0):
                if(personText in indexPersons):
                    #indexPersons[personText]['count'] += 1
                    indexPersons[personText]['count'] += column.valid
                    indexPersons[personText]['sentiment'] += sentence.sentiment.polarity
                    indexPersons[personText]['subjectivity'] += sentence.sentiment.subjectivity
                else:    
                    indexPersons[personText] = {'phrase':personText, 'label':entity.label_, 'sentiment':sentence.sentiment.polarity,
                                                 'subjectivity':sentence.sentiment.subjectivity, 'language':lang, 'count':1}   
            elif('ORG' == entity.label_):
                if(entity.text in indexOrganizations):
                    #indexOrganizations[entity.text]['count'] += 1
                    indexOrganizations[entity.text]['count'] += column.valid
                    indexOrganizations[entity.text]['sentiment'] += sentence.sentiment.polarity
                    indexOrganizations[entity.text]['subjectivity'] += sentence.sentiment.subjectivity
                else:    
                    indexOrganizations[entity.text] = {'phrase':entity.text, 'label':entity.label_, 'sentiment':sentence.sentiment.polarity,
                                                       'subjectivity':0, 'language':lang, 'count':1} 
            elif('MISC' == entity.label_):
                if(entity.text in indexMisc):
                    #indexMisc[entity.text]['count'] += 1
                    indexMisc[entity.text]['count'] += column.valid
                    indexMisc[entity.text]['sentiment'] += sentence.sentiment.polarity
                    indexMisc[entity.text]['subjectivity'] += sentence.sentiment.subjectivity
                else:         
                    indexMisc[entity.text] = {'phrase':entity.text, 'label':entity.label_, 'sentiment':sentence.sentiment.polarity,
                                              'subjectivity':sentence.sentiment.subjectivity, 'language':lang, 'count':1} 
            else:
                if(entity.text in indexMissing):
                    #indexMissing[entity.text]['count'] += 1
                    indexMissing[entity.text]['count'] += column.valid
                    indexMissing[entity.text]['sentiment'] += sentence.sentiment.polarity
                    indexMissing[entity.text]['subjectivity'] += sentence.sentiment.subjectivity
                else:
                    indexMissing[entity.text] = {'phrase':entity.text, 'label':entity.label_, 'sentiment':sentence.sentiment.polarity,
                                                 'subjectivity':sentence.sentiment.subjectivity, 'language':lang, 'count':1}  

colGeo = ['phrase', 'label', 'sentiment', 'subjectivity', 'language', 'count', 
           'gnd', 'geonames', 'geotype', 'latitude', 'longitude', 'continent', 'country', 'ipcc']
indexLocationsDF = pd.DataFrame.from_dict(indexLocations, orient='index', columns=colGeo)
indexLocationsDF['sentiment'] = indexLocationsDF['sentiment']/indexLocationsDF['count']
indexLocationsDF['subjectivity'] = indexLocationsDF['subjectivity']/indexLocationsDF['count']
indexLocationsDF = indexLocationsDF.sort_values(by=['count'], ascending=False)
indexLocationsDF = enrichFromGeonames(indexLocationsDF)
indexLocationsDF.to_csv(DATA_PATH / 'csv' / "sentiments_locations.csv", index=True, float_format='%.8f')   
 
colSent = ['phrase', 'label', 'sentiment', 'subjectivity', 'language', 'count']
indexPersonsDF = pd.DataFrame.from_dict(indexPersons, orient='index', columns=colSent)
indexPersonsDF['sentiment'] = indexPersonsDF['sentiment']/indexPersonsDF['count']
indexPersonsDF['subjectivity'] = indexPersonsDF['subjectivity']/indexPersonsDF['count']
indexPersonsDF = indexPersonsDF.sort_values(by=['count'], ascending=False)
indexPersonsDF.to_csv(DATA_PATH / 'csv' / "sentiments_persons.csv", index=True)

indexOrganizationsDF = pd.DataFrame.from_dict(indexOrganizations, orient='index', columns=colSent)
indexOrganizationsDF['sentiment'] = indexOrganizationsDF['sentiment']/indexOrganizationsDF['count']
indexOrganizationsDF['subjectivity'] = indexOrganizationsDF['subjectivity']/indexOrganizationsDF['count']
indexOrganizationsDF = indexOrganizationsDF.sort_values(by=['count'], ascending=False)
indexOrganizationsDF.to_csv(DATA_PATH / 'csv' / "sentiments_organizations.csv", index=True)

indexMiscDF = pd.DataFrame.from_dict(indexMisc, orient='index', columns=colSent)
indexMiscDF['sentiment'] = indexMiscDF['sentiment']/indexLocationsDF['count']
indexMiscDF['subjectivity'] = indexMiscDF['subjectivity']/indexLocationsDF['count']
indexMiscDF = indexMiscDF.sort_values(by=['count'], ascending=False)
indexMiscDF.to_csv(DATA_PATH / 'csv' / "sentiments_misc.csv", index=True)

indexMissingDF = pd.DataFrame.from_dict(indexMissing, orient='index', columns=colSent)
indexMissingDF['sentiment'] = indexMissingDF['sentiment']/indexLocationsDF['count']
indexMissingDF['subjectivity'] = indexMissingDF['subjectivity']/indexLocationsDF['count']
indexMissingDF = indexMissingDF.sort_values(by=['count'], ascending=False)
indexMissingDF.to_csv(DATA_PATH / 'csv' / "sentiments_missing.csv", index=True)



