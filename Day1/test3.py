# 条件语句
grade = 85
if grade >= 90:
    print("A")
elif grade >= 60:
    print("Pass")
else:
    print("Fail")

# 循环语句
for idx in range(5):
    if idx == 3:
        continue
    print(idx)

# 异常处理
try:
    user_input = int(input("请输入一个数字: "))
    print(100 / user_input)
except ZeroDivisionError:
    print("不能除以零!")
except ValueError:
    print("输入无效!")
finally:
    print("执行完毕。")
