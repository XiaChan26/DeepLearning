#  class 类
'''
    概念介绍：
    有没有一种方法可以像“函数”那样只是调用就可以，又能实现
    “函数”不能实现的功能（封装，继承等）---------->诞生了类

    如
    class ClassName:            用 class 声明一个类（跟函数声明类似）
                                类名首字母要大写（与函数做区别）
                                内部语法结构（与函数最大的不同是，可以部分被外部调用）
    ==========================================================================
    在类内部的语法结构声明变量或者声明函数，都可以看成是类的本身属性
    如：

'''


# 1、最简单的一个函数定义
def Say(words):
    print(words)


# 调用函数Say
Say('你好')


# 2、实例化
class Human:
    name = 'Human'

    def greet(self):
        print('hello')

    def age(self):
        print('age')


# 类的实例化
temp = Human()
temp.greet()


# 此处，没有实例化，直接用类Human访问内部age()会报错
# Human.age()
# ======================================================================
# 默认self参数的优势
# 1、函数思想
class Human1:
    name = 'Human1'

    def greet(self, words):
        print(words)


# 实例化
temp = Human1()
temp.greet('hello1')


# 2、类思想
class Human2:
    name = 'Human'
    words = ''

    def greet(self):
        # self指的是类本身
        print(self.words)


temp = Human2()
# 修改类属性的值
temp.words = 'hello2'
temp.greet()


# =====================================================================
# 结合函数思想和类思想的优点，引入了__init__() 方法
# 重点重点重点重点重点重点重点！！！！！！！！！！！！！！！！！！！！！！
class Human3:
    def __init__(self, words):
        self.words = words

    def greet(self):
        print(self.words)


# 实例化
temp = Human3('hello3')
temp.greet()
