import pandas as pd
import numpy as np
import numbers
import math
import random

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.img_tiles as cimgt
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

import io
from urllib.request import urlopen, Request
from PIL import Image

import folium
import webbrowser
from folium.plugins import HeatMap

from pathlib import Path
import os.path

DATA_PATH = Path.cwd()

extremeColors = {'unknown':'#ffffff', 'Thunderstorm':'#53785a', 'Storm':'#222222', 'Storm Surge':'#834fa1', 'Flash Flood':'#0245d8', 'Precipitation':'#608D3A', 'Wet Spell':'#22e91f',
               'Tsunami':'#690191',  'Landslide':'#1C4840', 'Cold Wave':'#a7e9fa', 'Heat Wave':'#d85212', 'Iceberg':'#02b5b8',
                 'Snow Avalanche':'#deddd5', 'Wildfire':'#fa0007', 'Fog':'#535271', 'Snow&Ice':'#dedde5', 'Flood':'#030ba1', 'Drought':'#572c03', 'Tropical Cyclone':'#4f7fbf', 'Volcano':'#b83202', 'Earthquake':'#870047'  
               }


#adapt limits and zoom-level (scale) according to data
limits = {'latMin':-80.0, 'latMax':80.0, 'lonMin':-180.0, 'lonMax':180.0}
scale = 3

#rivers_10m = cfeature.NaturalEarthFeature('physical', 'rivers_lake_centerlines', '10m')
#rivers_europe_10m = cfeature.NaturalEarthFeature('physical', 'rivers_europe', '10m')

locationsDF = pd.read_csv(DATA_PATH / 'csv' / 'sentiments_locations.csv', delimiter=',')
locationsDF = locationsDF.sort_values(by=['count'], ascending=False)
locationsDF = locationsDF.dropna()
if(not locationsDF.empty):
  locationsDF = locationsDF[(locationsDF['count'] > 2)]

print(locationsDF['latitude'].min())
print(locationsDF['latitude'].max())
print(locationsDF['longitude'].min())
print(locationsDF['longitude'].max())

def image_spoof(self, tile): 
    url = self._image_url(tile) 
    req = Request(url) 
    req.add_header('User-agent','Anaconda 3') 
    fh = urlopen(req) 
    im_data = io.BytesIO(fh.read()) 
    fh.close() 
    img = Image.open(im_data) 
    img = img.convert(self.desired_tile_form) 
    return img, self.tileextent(tile), 'lower' 

#cartopy OSM
cimgt.OSM.get_image = image_spoof 
osm_img = cimgt.OSM() 

fig = plt.figure(figsize=(12,9))
# project using coordinate reference system (CRS) of street map 
ax1 = plt.axes(projection=osm_img.crs) 
ax1.set_title('Hazard Density Map',fontsize=18)
extent = [limits['lonMin'], limits['lonMax'], limits['latMin'], limits['latMax']] 
ax1.set_extent(extent)
ax1.set_xticks(np.linspace(limits['lonMin'],limits['lonMax'],9),crs=ccrs.PlateCarree()) 
ax1.set_yticks(np.linspace(limits['latMin'],limits['latMax'],7)[1:],crs=ccrs.PlateCarree()) 
lon_formatter = LongitudeFormatter(number_format='0.2f',degree_symbol='',dateline_direction_label=True)
lat_formatter = LatitudeFormatter(number_format='0.2f',degree_symbol='') 
ax1.xaxis.set_major_formatter(lon_formatter) 
ax1.yaxis.set_major_formatter(lat_formatter) 
ax1.xaxis.set_tick_params(labelsize=14)
ax1.yaxis.set_tick_params(labelsize=14)
# add OSM with zoom specification
ax1.add_image(osm_img, scale) 

ax1.coastlines(resolution='50m', color='black', linewidth=1)

sumCounts = np.sum(locationsDF['count'])
maxCount = np.max(locationsDF['count'])
print(['sum',sumCounts,'max',maxCount])   #211,28
lat1,long1,size1 = [],[],[]
for index, column in locationsDF.iterrows():
  if(isinstance(column['longitude'], numbers.Number) and isinstance(column['latitude'], numbers.Number)):
    if((limits['latMin']<column['latitude']<limits['latMax']) and (limits['lonMin']<column['longitude']<limits['lonMax'])):
        delta = 1.0
        colorExtr = '#2244bb'
        ## if(column['extreme'] in extremeColors):   ## extremes not in location
        ##   colorExtr = extremeColors[column['extreme']]
        counter = int(column['count']/maxCount*250+column['count']/sumCounts*180)
        if(column['geotype']=='L'):   #large
            #counter = 1
            delta = 30.0
        if(column['geotype']=='A'):   #country
            delta = 7.0
        if(column['geotype']=='P'):  #city  T
            delta = 2.0 
        ax1.plot(column['longitude'], column['latitude'], 
                 markersize=5,marker='o',linestyle='', markeredgecolor=None,
                 color=colorExtr, alpha=0.5,transform=ccrs.PlateCarree())  
        for i in range(counter):
            x=random.gauss(column['longitude'],delta)
            y=random.gauss(column['latitude'],delta)
            lat1.append(x)
            long1.append(y)
            ax1.plot(x, y, 
                    markersize=30,marker='o',linestyle='', markeredgecolor=None,
                    color=colorExtr, alpha=0.002,transform=ccrs.PlateCarree())                     
            ax1.plot(x, y, 
                    markersize=17,marker='o',linestyle='', markeredgecolor=None,
                    color=colorExtr, alpha=0.004,transform=ccrs.PlateCarree())  
#contour-plot
sns.kdeplot(x=lat1, y=long1, fill=False,  levels=10, thresh=.0005, color='grey', transform=ccrs.PlateCarree()  )  
## ax1.add_feature(rivers_10m, facecolor='None', edgecolor='cyan', linewidth=1.5, zorder=2)
## ax1.add_feature(rivers_europe_10m, facecolor='None', edgecolor='red', linewidth=1.5, zorder=2)

#North-Arrow
#ax1.text(-75.0, 175.0,u'\u25B2 \nN ', ha='center', fontsize=30, family='Arial', rotation = 0)

#for label in labels:
#    ax1.text(label['lon'],label['lat'],label['name'], color='#200000', fontsize=14, ha='center', va='center',transform=ccrs.PlateCarree())


# no legend for now
'''
colorLeg = list(extremeColors.values())
colorLeg.reverse()
labelLeg = list(extremeColors.keys())
labelLeg.reverse()
custom_lines = [plt.Line2D([],[], ls="", marker='.', 
                mec='k', mfc=c, mew=.1, ms=30) for c in colorLeg]
leg = ax1.legend(custom_lines, labelLeg, 
          loc='center left', fontsize=12, bbox_to_anchor=(0.99, .5))
leg.set_title("Topics", prop = {'size':16}) 
fig.add_artist(leg)
plt.tight_layout()
'''

if(not os.path.exists(DATA_PATH / 'img')):
    os.mkdir(DATA_PATH / 'img')
plt.savefig(DATA_PATH / 'img' / 'heatmap.png', dpi=300)


'''
heatdata = []
for index, column in locationsDF.iterrows():
    if(isinstance(column['longitude'], numbers.Number) and isinstance(column['latitude'], numbers.Number)):
    #if((50<column['latitude']<52) and (6<column['longitude']<8) and not ('Nordrhein-Westfalen' == column['phrase'])):
        #print([column['phrase'],column['count']])
        print([column['latitude'],column['longitude']])
        heatdata.append([column['latitude'],column['longitude'],1])

map_osm = folium.Map(location=[51,7],zoom_start=6,tiles='StamenTerrain',control_scale=True)
HeatMap(heatdata).add_to(map_osm)
map_osm.save(str(DATA_PATH / "img" / "heatmap.html"))
'''
