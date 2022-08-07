# 定义一个类 - 学生类
class Student:
    name = 'zhangsan'

    def __init__(self, name, age):
        self.name = name  # self.name 称为实体属性，赋给实体属性
        self.age = age
        '''具体流程为：
        比如需要调用__init__这个方法，传入一个name的值，此时name通过
        self.name = name ，将值传给self.name，即复制给self.name这个实例属性
        '''

    # 定义吃饭这个动作
    # 称为实例方法
    def eat(self):  # 这个self一定要加上，固定的写法， 不然会报错
        print('lisi能吃三碗饭')

    # 定义计算
    def calc(self):
        print(1 + 2)

    # 静态方法
    @staticmethod
    def method():
        print('我是张三')

    # 类方法
    @classmethod
    def classmethod(cls):
        print("我是一个类方法")


stu = Student('likui', 24)
print(Student.name)
print(stu.name)
# print(Student.__dict__)
