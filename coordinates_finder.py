# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 20:26:42 2020

@author: Xkfal
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from geopy.geocoders import Nominatim
import math
import cartopy.feature as cfeature
from geopy.extra.rate_limiter import RateLimiter

from tqdm import tqdm
tqdm.pandas()
df = pd.read_csv('db_data.csv')
geolocator = Nominatim(user_agent="nyc_re_proj")
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
batch = []
df.fillna({'zip_code':-1}, inplace=True)
addresses = df.address.str.split(",", expand = True)[0] + "," + df['zip_code'].astype(int).astype(str)

def eval_results(x):
     try:
         return (x.latitude, x.longitude)
     except:
         return (-1, -1)

#loc = addresses.progress_apply(lambda x: geocode(x))
loc = addresses.progress_apply(geolocator.geocode, timeout=1000000).apply(lambda x: eval_results(x))
coordinates = pd.DataFrame.from_records(loc, columns=['longitude', 'latitude'])
coordinates.to_csv("C:/Users/Xkfal/Documents/nyc_realestate_proj/coordinates.csv", index = False)
df_address = df.join(coordinates)
df_address.to_csv("C:/Users/Xkfal/Documents/nyc_realestate_proj/addresses.csv", index = False)
df.sale_price.median()


# import the geocoding services you'd like to try
from geopy.geocoders import ArcGIS, Bing, Nominatim, OpenCage, GeocoderDotUS, GoogleV3, OpenMapQuest
import csv, sys

print('creating geocoding objects!')

arcgis = ArcGIS(timeout=100)
nominatim = Nominatim(timeout=100)
geocoderDotUS = GeocoderDotUS(timeout=100)
googlev3 = GoogleV3(timeout=100)
openmapquest = OpenMapQuest(timeout=100)

# choose and order your preference for geocoders here
geocoders = [nominatim]


def geocode(address):
    i = 0
    try:
            # try to geocode using a service
        location = geocoders[i].geocode(address)

            # if it returns a location
        if location != None:
                
            batch[i] = [location.latitude, location.longitude]  
            return location.latitude, location.longitude
        else:
            return -1,-1
    except:
        # catch whatever errors, likely timeout, and return null values
        print(sys.exc_info()[0])
        return ['null','null']
    i = i+1

    # if all services have failed to geocode, return null values
    return ['null','null']
    
# =============================================================================
# 
# # list to hold all rows
# with open('data.csv', mode='rb') as fin:
# 
#     reader = csv.reader(fin)
#     j = 0
#     for row in reader:
#         print('processing #',j)
#         j+=1
#         try:
#             # configure this based upon your input CSV file
#             street = row[4]
#             city = row[6]
#             state = row[7]
#             postalcode = row[5]
#             country = row[8]
#             address = street + ", " + city + ", " + state + " " + postalcode + " " + country
#             
#             result = geocode(address)
#             # add the lat/lon values to the row
#             row.extend(result)
#             # add the new row to master list
#             dout.append(row)
#         except:
#             print('you are a beautiful unicorn')
# 
# 
# # print results to file
# with open('geocoded.csv', 'wb') as fout:
#     writer = csv.writer(fout)
#     writer.writerows(dout)
# 
# print('all done!')
# 
# =============================================================================
