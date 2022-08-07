# 类的创建
class Girlfriend:
    inc_ratio = 1.1

    def __init__(self, first_name, family_name, university, expense):
        self.first_name = first_name
        self.family_name = family_name
        self.email = first_name + "-" + "@" + university + ".edu"
        self.expense = expense

    def gfnames(self):
        return f'{self.first_name} {self.family_name}'

    def expense_inc(self):
        self.expense = int(self.expense * self.inc_ratio)


gf1 = Girlfriend("Email", "li", "tsing", 2000)
gf2 = Girlfriend("Carry", "Wang", "peking", 1500)

print(gf1.expense)
gf1.expense_inc()
print(gf1.expense)

print(Girlfriend.inc_ratio)
print(gf1.inc_ratio)
print(gf2.inc_ratio)

# 打印命名空间
print(Girlfriend.__dict__)
print(gf1.__dict__)

