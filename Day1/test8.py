# 写文件
with open("example.txt", "w") as file:
    file.write("Hello, Python!\n")

# 读文件
with open("example.txt", "r") as file:
    print(file.read())

# 处理CSV
import csv
with open("data.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Name", "Age"])
    writer.writerow(["Alice", 20])
