def test_package():
    print('test package')

class Test():
    def __get__(self, instance, owner):
        print(1)

    def __set__(self, instance, value):
        print(2)

class Cat():
    t  = Test()
    def __init__(self):

        pass
    def Eat(self):
        print("eat")

    def __get__(self, instance, owner):
        print(1)
cat = Cat()
print(cat.t)
cat.t = 2