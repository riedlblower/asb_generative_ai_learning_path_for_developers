import csv
from datetime import datetime

total_revenue = 0
max_revenue_product = None
max_revenue = 0
max_revenue_date = None

with open('sales.csv') as f:
  reader = csv.DictReader(f)
  for row in reader:
    price = int(row['price'])
    units = int(row['units_sold'])
    revenue = price * units

    date = datetime.strptime(row['date'], '%Y-%m-%d').date()
    product_id = row['product_id']

    total_revenue += revenue

    if revenue > max_revenue:
      max_revenue = revenue
      max_revenue_product = product_id
      max_revenue_date = date

print(total_revenue)
print(max_revenue_product)
print(max_revenue_date)