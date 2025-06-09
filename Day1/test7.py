# 简单装饰器
def decorator(func):
    def wrapper():
        print("Before function")
        func()
        print("After function")
    return wrapper

@decorator
def greet():
    print("Hello!")

greet()

# 带参数的装饰器
def repeat(n):
    def wrap(func):
        def wrapper(*args, **kwargs):
            for _ in range(n):
                func(*args, **kwargs)
        return wrapper
    return wrap

@repeat(3)
def greet_name(name):
    print(f"Hi, {name}!")

greet_name("Alice")
