from data_helper import calculateRGBdiff, calculate_intersection

p1 = (160, 125, 550, 427)
p2 = (448, 220, 539, 379)

area = calculate_intersection(p1, p2)

print(area)