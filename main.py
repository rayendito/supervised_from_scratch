import csv
# from src.KNN import printhello

#open file
csvfile = open("heart.csv", encoding="UTF-8")
csvread = csv.reader(csvfile, delimiter=',')

# print(csvread.line_num)
for row in csvread:
    print(len(row))
    break
