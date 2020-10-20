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


df = pd.read_csv('db_data.csv')
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)


from geopy.extra.rate_limiter import RateLimiter
from tqdm import tqdm
tqdm.pandas()

def eval_results(x):
     try:
         return (x.latitude, x.longitude)
     except:
         return (-1, -1)

loc = df['Location'].progress_apply(geolocator.geocode, timeout=1000000).apply(lambda x: eval_results(x))

coordinates = pd.DataFrame.from_records(loc, columns=['longitude', 'latitude'])
coordinates.to_csv("coordinates.csv", index = False)