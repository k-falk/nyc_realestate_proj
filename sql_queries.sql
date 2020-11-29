USE nyc_re;
SELECT * FROM address;

CREATE TABLE boroughs (
	borough varchar(255),
    borough_id int,
    PRIMARY KEY(borough_id)
);
ALTER TABLE address
RENAME COLUMN borough TO borough_id;

ALTER TABLE address
RENAME COLUMN `index` TO id;

ALTER TABLE address
ADD PRIMARY KEY (id);


ALTER TABLE address MODIFY borough_id INTEGER;

INSERT INTO boroughs (borough, borough_id)
VALUES 	('Manhattan', 1),
		('Bronx', 2),
	    ('Brooklyn', 3),
        ('Queens', 4), 
        ('Staten Island', 5);

ALTER TABLE address
ADD FOREIGN KEY (borough_id) REFERENCES boroughs(borough_id);


#return highest sale price in table 
SELECT MAX(sale_price) FROM address;

#return property info with highest sale price in table 
SELECT * FROM address
WHERE sale_price = (SELECT MAX(sale_price) FROM address);
#return address with highest sale price in table 
SELECT address, sale_price FROM address
WHERE sale_price = (SELECT MAX(sale_price) FROM address);

#select [roperty info with nth(3rd) highest sale price in table
#here our we put n-1,1 after our LIMIT to get n. So for 3rd highest we put 2,1
SELECT DISTINCT * FROM address ORDER BY sale_price DESC LIMIT 2,1;

#select range of property based on id
SELECT * FROM address WHERE id BETWEEN 1 AND 10;

#return address, highest sale price and burough
SELECT address, MAX(sale_price) as "Max Sale Price", borough
FROM address INNER JOIN boroughs
ON address.borough_id = boroughs.borough_id;

#return highest sale price, address, borough for each borough
SELECT address, MAX(sale_price) as "Max Sale Price", borough
FROM address INNER JOIN boroughs
ON address.borough_id = boroughs.borough_id
GROUP BY boroughs.borough;

#Return average sale price, address, borough for each borough
SELECT address, avg(sale_price) as "Average Sale Price", borough
FROM address INNER JOIN boroughs
ON address.borough_id = boroughs.borough_id
GROUP BY boroughs.borough;