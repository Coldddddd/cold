# 变量类型
person_name = "Alice"  # 字符串
person_age = 20        # 整数
scores = [90, 85, 88]  # 列表
profile = {"name": "Alice", "age": 20}  # 字典

# 类型转换
age_in_str = str(person_age)
numeric_value = int("123")

# 作用域
global_var = 10  # 全局变量
def test_function():
    local_var = 5  # 局部变量
    global global_var
    global_var += 1
    print(f"函数内: global_var={global_var}, local_var={local_var}")

test_function()
print(f"函数外: global_var={global_var}")
