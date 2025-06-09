# 创建模块 custommodule.py
# custommodule.py
def greet_user():
    return "Hello from custom module!"

# 主程序
import custommodule
print(custommodule.greet_user())

# 导入第三方模块
import requests
response = requests.get("https://api.github.com")
print(response.status_code)  # 200

# 包使用示例
from mypackage import custommodule
