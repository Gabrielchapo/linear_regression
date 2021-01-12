import numpy as np
from LinearRegression import LinearRegression

model = LinearRegression()
model.load()

try:
    mileage = int(input("please enter a mileage: "))
except:
    exit("mileage must be an integer")

if mileage < 0 or mileage > 1000000:
    exit("incorrect mileage")

price = model.predict(mileage)
if price > 0:
    print("Estimated Price:", model.predict(mileage))
else:
    print("Your car probably worths nothing anymore.")
