# 定义类
class Person:
    def __init__(self, full_name, age):
        self.full_name = full_name
        self.age = age

    def introduce(self):
        return f"I am {self.full_name}, {self.age} years old."

# 继承
class PostGradPerson(Person):
    def __init__(self, full_name, age, field_of_study):
        super().__init__(full_name, age)
        self.field_of_study = field_of_study

    def introduce(self):
        return f"I am {self.full_name}, a {self.field_of_study} student."

# 使用
undergrad = Person("Alice", 20)
postgrad = PostGradPerson("Bob", 22, "CS")
print(undergrad.introduce())  # I am Alice, 20 years old.
print(postgrad.introduce())   # I am Bob, a CS student.
