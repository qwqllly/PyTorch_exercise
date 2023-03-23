class Foobar:
    def __init__(self):
        pass

    def __call__(self,*args,**kwargs):# *args:位置参数，**kwargs：关键字参数
        print("hello " + str(args)) #必须转成字符串，can only concatenate str (not "tuple") to str
        print("hi " + str(kwargs))
    
f = Foobar()
f(1,2,3,c=10,d=100)

def func(*args,**kwargs):
    print("hello ",args) # 变成元组存起来
    print("hi ",kwargs) # 变成字典存起来

func(2,4,7,8,0,x=1,y=90)