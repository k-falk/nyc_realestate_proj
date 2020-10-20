# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 18:00:08 2020

@author: Xkfal
"""


import pandas as pd
import mysql.connector
from mysql.connector import Error
import datetime
from sqlalchemy import create_engine
import pymysql

## SQL Alchemy engine
engine = create_engine('mysql+pymysql://root:bandit365@localhost/nyc_re')



path = 'C:/Users/Xkfal/Documents/nyc_realestate_proj/'
df_manhattan = pd.read_csv(path + 'rollingsales_manhattan.csv', header = 4)
df_bronx = pd.read_csv(path + 'rollingsales_bronx.csv', header = 4)
df_brooklyn = pd.read_csv(path + 'rollingsales_brooklyn.csv',header = 4)
df_queens = pd.read_csv(path + 'rollingsales_queens.csv',header = 4)
df_statenisland = pd.read_csv(path + 'rollingsales_statenisland.csv',header = 4)

df = pd.concat([df_manhattan, df_bronx, df_brooklyn, df_queens, df_statenisland])

df.head()       
#query = 'INSERT INTO address VALUES(' + i +', {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {})'
#for row in df:
#    print(row +'')
    #query.format()


## We are going to significantly cut down on our data by removing rows we do not need
## We are going to drop unneccesary or redudant columns, NA values, duplicates and values with a sale price of 0
## Sale price of 0 means that the property was transfered such as from one family member to the next
## so it is unneccesary for our purposes
df = df.drop(columns= ['EASE-MENT'])
df = df.drop(columns= ['APARTMENT NUMBER'])
df = df.drop(columns = ['TAX CLASS AT PRESENT'])
df = df.drop(columns = ['BUILDING CLASS AT PRESENT'])
df = df.drop(columns = ['GROSS SQUARE FEET'])

## Rename columns for easier use and for database down the road
df = df.rename(columns={'ADDRESS' : 'address', 'NEIGHBORHOOD':'neighborhood', 'BOROUGH': 'borough',
                        'BUILDING CLASS AT TIME OF SALE': 'building_class_id', 'ZIP CODE': 'zip_code', 
                        'TAX CLASS AT TIME OF SALE' : 'tax_class',
                        'BLOCK' :'block_id', 'LOT': 'lot_id', 'BUILDING CLASS CATEGORY' : 'building_category',
                        'RESIDENTIAL UNITS' : 'res_units', 'LAND SQUARE FEET' : 'square_ft', 
                        'COMMERCIAL UNITS' : 'comm_units', 'TOTAL UNITS' : 'total_units',
                        'YEAR BUILT': 'year_built', ' SALE PRICE ' : 'sale_price', 'SALE DATE': 'sale_date',})


df = df[df['sale_price'].notna()]



df.head()
df.tail()
df.columns
df = df.drop_duplicates()

# We had a lot of sales that were less than 100. These are transfers
df['sale_price'] = df['sale_price'].replace(',','', regex=True)
df["sale_price"] = df["sale_price"].astype(int)
df = df[(df['sale_price']) > 100]

## Now we are going to put the dataframe in a database
df.columns
df.sale_date = df.sale_date.apply(lambda x: date_format(x))
df.to_sql(name='address', con=engine, if_exists = 'replace', index=True)
df.to_csv('db_data.csv', index = False)
df = pd.read_csv('db_data.csv')

def date_format(x):
    if(str(x) != 'nan'):
        return datetime.datetime.strptime(str(x), "%m/%d/%Y").strftime(('%Y-%m-%d'))
    else:
        return 'NULL'

        
        
        
     ## SAMPLE QUERY FOR REFERENCE  ## 
#CREATE TABLE address (
#  address_id INT PRIMARY KEY,
#  address VARCHAR(80),
#  neighborhood VARCHAR(40),
#  building_class_category varchar(40),
#  tax_class INT,
#  block_id INT, 
#  lot_id INT, 
#  building_class_num varchar(2), 
#  apartment_number INT, 
#  zipcode INT, 
#  square_feet INT, 
#  residential_units_count INT, 
#  commercial_units_count INT, 
#  year_built INT, 
#  sale_price INT, 
#  sale_date DATE
#);

############################## JUNK CODE IGNORE ########################
 # IN CASE IT IS NEEDED. I spent about 6 hours trying to figure it out without SQL alchemy so I am not deleting this
 # if I end up somehow needing it

#df.to_sql('address', con = connection)
#df.columns
#
#for index, row in df.iterrows():
#    #for j in df.columns:
#    address = row['ADDRESS']
#    borough = str(row['BOROUGH'])
#    neighborhood = str(row['NEIGHBORHOOD'])
#    building_class_category = row['BUILDING CLASS CATEGORY']
#    tax_class = str(row['TAX CLASS AT TIME OF SALE'])
#    block_id = str(row['BLOCK'])
#    lot_id = str(row['LOT'])
#    building_class_num = row['BUILDING CLASS AT TIME OF SALE']
#    zipcode = str(row['ZIP CODE'])
#    square_feet = str(row['LAND SQUARE FEET'])
#    residential_units_count = str(row['RESIDENTIAL UNITS'])
#    commercial_units_count = str(row['COMMERCIAL UNITS'])
#    year_built = str(row['YEAR BUILT'])
#    sale_price = str(row[' SALE PRICE '])
#    sale_date = datetime.datetime.strptime(row['SALE DATE'], "%m/%d/%Y").strftime(('%Y-%m-%d'))
#    data = "'{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}'"
#    
#    query = 'INSERT INTO address VALUES(' + str(index)+ ', ' + data +')'
#    query = query.format(address,neighborhood, borough ,building_class_category, 
#                 tax_class,block_id,lot_id, building_class_num,
#                 zipcode, square_feet, residential_units_count,
#                 commercial_units_count, year_built, sale_price, sale_date)
#    
#    cursor.execute(query)
#    
#pd.DataFrame(val) 
#df.to_sql("addresses", con = connector)   
#query = """INSERT IGNORE INTO address (address,neighborhood, borough ,building_class_category, 
#                 tax_class,block_id,lot_id, building_class_num,
#                 zipcode, square_feet, residential_units_count,
#                 commercial_units_count, year_built, sale_price, sale_date) 
#                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
#cursor.executemany(query, val)
#address = df['ADDRESS']
#neighborhood = (df['NEIGHBORHOOD'])
#borough = df["BOROUGH"]
#building_class_category = df['BUILDING CLASS CATEGORY']
#tax_class = (df['TAX CLASS AT TIME OF SALE'])
#block_id = (df['BLOCK'])
#lot_id = (df['LOT'])
#building_class_num = df['BUILDING CLASS AT TIME OF SALE']
#zipcode = (df['ZIP CODE'])
#square_feet = (df['LAND SQUARE FEET'])
#residential_units_count = (df['RESIDENTIAL UNITS'])
#commercial_units_count = (df['COMMERCIAL UNITS'])
#year_built = (df['YEAR BUILT'])
#sale_price = (df[' SALE PRICE '])
#sale_date = df['SALE DATE'].apply(lambda x : date_format(x))
#val = [address,neighborhood, borough ,building_class_category, 
#             tax_class,block_id,lot_id, building_class_num,
#             zipcode, square_feet, residential_units_count,
#             commercial_units_count, year_built, sale_price, sale_date]
#   
#cursor.executemany(query,val)
#
#db_close(connection,cursor)
