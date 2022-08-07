class Human:
    def __init__(self, words):
        self.words = words

    def speak(self):
        print(self.words)


# 实例化
temp = Human('hello')
temp.speak()
'''
    整个思想:
    class 中的函数，最少要有一个形参
    且第一个参数self默认指向的是class本身
    所以在__init__()中self.name的过程中，使得该部分在该类函数中都可以使用
    而且在实例化过程中传入的参数，可以直接传到__init__()中，然后
    赋值给self.name 
'''