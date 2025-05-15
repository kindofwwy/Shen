import random,math

e=2.718281828459

class Vec(list):
    '''
    矢量，一个元素只有数字的列表，可以进行按位加减乘等操作
    '''
    def __add__(self, other):
        if len(self)!=len(other):
            raise Exception("运算的两者长度不同")
        return Vec([self[i] + other[i] for i in range(len(self))])

    def __sub__(self, other):
        if len(self)!=len(other):
            raise Exception("运算的两者长度不同")
        return Vec([self[i] - other[i] for i in range(len(self))])

    def __mul__(self, other):
        if len(self)!=len(other):
            raise Exception("运算的两者长度不同")
        return Vec([self[i] * other[i] for i in range(len(self))])

    def __truediv__(self, other):
        if len(self)!=len(other):
            raise Exception("运算的两者长度不同")
        return Vec([self[i] / other[i] for i in range(len(self))])

    def __iadd__(self, other):
        return Vec(self+other)

    def __pow__(self, power:float, modulo=None):
        return Vec([i ** power for i in self])

    def __neg__(self):
        return Vec([-i for i in self])

    def batchprocess(self,func,*args):
        return Vec([func(self[i],*args) for i in range(len(self))])

class Ten():
    '''
    张量，一个带梯度的Vec，梯度是另一个同维Vec
    '''
    def __init__(self,lis,op=None):
        '''
        创建Ten
        :param lis: list or Vec
        :param op: 创建该张量的运算符，反向传播时用。无需输入
        '''
        self.data=Vec(lis)
        self.grad=Vec([0 for i in range(len(lis))])
        self.op=op

    def __repr__(self):
        return f"<data:{[i for i in self.data]},grad:{[i for i in self.grad]}>"

    def __add__(self, other):
        o=Add()
        return o.compute(self,other)

    def __sub__(self, other):
        o=Sub()
        return o.compute(self,other)

    def __mul__(self, other):
        o=Mul()
        return o.compute(self,other)

    def __truediv__(self, other):
        o=Div()
        return o.compute(self,other)

    def __pow__(self, power, modulo=None):
        o=Pownum()
        return o.compute(self,power)

    def __len__(self):
        return len(self.data)

    def sum(self):
        o=Sum()
        return o.compute(self)

    def relu(self):
        o=Relu()
        return o.compute(self)

    def exp(self):
        o=Exp()
        return o.compute(self)

    def log(self):
        o=Log()
        return o.compute(self)

    def softmax(self):
        a=self.exp()
        s=a.sum()
        s=s**-1 if s.data[0]!=0 else Ten([0])
        c=Ten.connect([s for i in range(len(a))])
        return c*a

    def sigmoid(self):
        o=Sigmoid()
        return o.compute(self)

    def tanh(self):
        o=Tanh()
        return o.compute(self)

    def gelu(self):
        return self*(self*Ten([1.702]).expand(len(self))).sigmoid()

    @classmethod
    def mse(cls,a,b):
        '''
        均方误差
        :param a: Ten
        :param b: Ten
        :return: Ten
        '''
        return ((a-b)**2).sum()/Ten([len(a.data)])

    @classmethod
    def sse(cls,a,b):
        return ((a-b)**2).sum()

    @classmethod
    def nll(cls, out, target):
        '''
        交叉熵误差
        :param out: Ten
        :param target: Ten
        :return: Ten
        '''
        return Ten([0])-(target*out.log()).sum()

    def zerograd(self):
        '''
        将自身梯度设为0
        :return: None
        '''
        self.grad = Vec([0 for i in self.grad])

    def onegrad(self):
        '''
        将自身梯度设为1
        :return: None
        '''
        self.grad = Vec([1 for i in self.grad])

    @classmethod
    def zero(cls,size):
        '''
        创建一个长度为size的Ten，填充0
        :param size: int
        :return: Ten
        '''
        return Ten([0 for i in range(size)])

    def graddescent(self,k):
        '''
        梯度下降，在反向传播积累梯度后使用
        :param k: float 步长
        :return: None
        '''
        self.data-=self.grad.batchprocess(lambda x:x*k)

    def back(self,clean=True):
        '''
        反向传播
        速度较慢，快速版详见 Operator.back()
        :param clean: bool 是否清零计算图
        :return: None
        '''
        self.onegrad()
        oplist=[self.op]
        for i in oplist:
            if i is None:
                continue
            if type(i.inp) is list: # 若运算符输入为list
                for ea in i.inp:
                    if ea.op not in oplist and ea.op is not None:
                        oplist.append(ea.op)    # 把每个输入的运算符(去掉重复的)加入表中
            elif i.inp.op not in oplist and i.inp.op is not None:
                oplist.append(i.inp.op)
        oplist.sort(key=lambda x:Operator.computelist.index(x),reverse=True)
        for i in oplist:
            i.diriv()
        if clean:
            Operator.clean()

    def cut(self,start,end):
        '''
        切片，取自身的一部分
        :param start: int 开始索引
        :param end: int 结束索引（结果不包括结束索引对应元素）
        :return: Ten
        '''
        o=Cut()
        return o.compute(self,start,end)

    def expand(self,times):
        '''
        膨胀，把自身复制到多维度上，用于一维数字与多维向量计算时使用
        :param times: int 复制多少次
        :return: Ten
        '''
        out=Ten.connect([self for i in range(times)])
        out.zerograd()
        return out

    @classmethod
    def connect(cls,x):
        '''
        将一列Ten和为一个Ten
        :param x: list[Ten,Ten...]
        :return: Ten
        '''
        o=Connect()
        return o.compute(x)


class Operator():
    '''
    运算符类，所有对Ten操作的运算符都需要继承此类，并重写compute和diriv方法
    '''
    computelist=[]

    def __init__(self):
        Operator.computelist.append(self)
        self.inp=[]
        self.out=[]

    def compute(self,*args):
        '''
        进行运算。每一个运算符均需重写运算过程，填写self.inp和self.out，创建新的Ten并返回
        self.inp: Ten or list
        self.out: Ten
        :return: Ten
        '''
        pass

    def diriv(self):
        '''
        进行微分。对self.inp.grad进行处理，通常使用+=用于积累梯度
        :return: None
        '''
        pass

    @classmethod
    def back(cls,last1grad=True):
        '''
        全局梯度计算。从后向前对每一个运算符使用diriv
        *使用后，会自动把computelist的内容删除，请注意
        :param last1grad: bool 是否把参与运算的最后一个Ten的导数设为1
        :return:None
        '''
        Operator.computelist.reverse()
        if last1grad:
            Operator.computelist[0].out.onegrad()
        for o in Operator.computelist:
            o.diriv()
            o.inp=None
            o.out=None
        Operator.clean()

    @classmethod
    def clean(cls):
        '''
        清理所有Operator，请务必在运行后调用，避免内存占用过大
        :return: None
        '''
        for o in Operator.computelist:
            o.inp=None
            o.out=None
        Operator.computelist=[]

class Add(Operator):
    def __init__(self):
        super().__init__()

    def compute(self, a:Ten, b:Ten):
        self.inp=[a,b]
        c = Ten(a.data + b.data,self)
        self.out=c
        return c

    def diriv(self):
        self.inp[0].grad += self.out.grad
        self.inp[1].grad += self.out.grad

class Sub(Operator):
    def __init__(self):
        super().__init__()

    def compute(self, a:Ten, b:Ten):
        self.inp=[a,b]
        c = Ten(a.data - b.data,self)
        self.out=c
        return c

    def diriv(self):
        self.inp[0].grad += self.out.grad
        self.inp[1].grad -= self.out.grad

class Mul(Operator):
    def __init__(self):
        super().__init__()

    def compute(self, a: Ten, b: Ten):
        self.inp = [a, b]
        c = Ten(a.data * b.data,self)
        self.out = c
        return c

    def diriv(self):
        self.inp[0].grad += self.inp[1].data * self.out.grad
        self.inp[1].grad += self.inp[0].data * self.out.grad

class Div(Operator):
    def __init__(self):
        super().__init__()

    def compute(self, a:Ten, b:Ten):
        self.inp=[a,b]
        c=Ten(a.data / b.data,self)
        self.out=c
        return c

    def diriv(self):
        self.inp[0].grad+=(self.inp[1].data**-1)*self.out.grad
        self.inp[1].grad+=(-self.inp[0].data/self.inp[1].data**2)*self.out.grad

class Sum(Operator):
    def __init__(self):
        super().__init__()

    def compute(self,a):
        self.inp=a
        c=Ten([sum(a.data)],self)
        self.out=c
        return c

    def diriv(self):
        for i in range(len(self.inp.grad)):
            self.inp.grad[i]+=self.out.grad[0]

class Connect(Operator):
    def __init__(self):
        super().__init__()

    def compute(self,x:list):
        self.inp=x
        c=Ten([],self)
        for i in range(len(x)):
            c.data.extend(x[i].data)
            c.grad.extend(x[i].grad)
        self.out=c
        return c

    def diriv(self):
        seg=self.out.grad
        for i in range(len(self.inp)):
            self.inp[i].grad += Vec(seg[:len(self.inp[i].grad)])
            seg= seg[len(self.inp[i].grad):]

class Cut(Operator):
    def __init__(self):
        super().__init__()

    def compute(self,a,start,end):
        self.inp=a
        self.start=start
        self.end=end
        c=Ten(a.data[start:end],self)
        self.out=c
        return c

    def diriv(self):
        for i in range(len(self.out.data)):
            self.inp.grad[i+self.start]+=self.out.grad[i]

class Relu(Operator):
    def __init__(self):
        super().__init__()

    def compute(self,a):
        self.inp=a
        c=Ten([max(i,0) for i in a.data],self)
        self.out=c
        return c

    def diriv(self):
        for i in range(len(self.inp.data)):
            if self.inp.data[i]>=0:
                self.inp.grad[i]+=self.out.grad[i]

class Pownum(Operator):
    '''
    幂运算
    '''
    def __init__(self):
        super().__init__()

    def compute(self,a,num):
        '''
        幂运算。指数只能为数字
        :param a:Ten
        :param num: float
        :return: Ten
        '''
        self.inp=a
        self.num=num
        c=Ten([i**num for i in a.data],self)
        self.out=c
        return c

    def diriv(self):
        self.inp.grad+=Vec([self.num * self.inp.data[i] ** (self.num - 1) * self.out.grad[i] for i in range(len(self.inp.grad))])

class Exp(Operator):
    '''
    自然指数函数
    '''
    def __init__(self):
        super().__init__()

    def compute(self,a):
        self.inp=a
        c=Ten([e**i for i in a.data],self)
        self.out=c
        return c

    def diriv(self):
        self.inp.grad+=self.out.data*self.out.grad

class Log(Operator):
    '''
    自然对数函数
    '''
    def __init__(self):
        super().__init__()

    def compute(self,a):
        self.inp=a
        c=Ten([math.log(i) if i!=0 else float("inf") for i in a.data],self)
        self.out=c
        return c

    def diriv(self):
        self.inp.grad+=self.inp.data**-1*self.out.grad

class Sigmoid(Operator):
    def __init__(self):
        super().__init__()

    def compute(self, a):
        self.inp=a
        c=Ten([1/(1+e**(-i)) for i in a.data],self)
        self.out=c
        return c

    def diriv(self):
        self.inp.grad+=self.out.data*(self.out.grad*(Vec([1 for i in range(len(self.out))])-self.out.data))

class Tanh(Operator):
    def __init__(self):
        super().__init__()

    def compute(self,a):
        self.inp=a
        c=Ten([(e**i-e**(-i))/(e**i+e**(-i)) for i in a.data],self)
        self.out=c
        return c

    def diriv(self):
        self.inp.grad+=self.out.grad*Vec([1-i**2 for i in self.out.data])

class Layer:
    '''
    数据层。所有要保存数据的类都需要继承此类
    '''
    layerlist=[]    # 装着所有要保存数据的实例的表
    isload=False    # 是否处于读取状态
    issave=True     # 是否处于保存状态
    pointer=0   # 读取数据用的指针
    filelen=0   # 读取的数据中子类的数量

    def __init__(self,*args):
        '''
        当处于读取状态(isload==True)时，继承了Layer的实例会按顺序读取layerlist中的内容
        当处于存储状态(issave==True)时，继承了Layer的实例会在创建时被加入layerlist，它在saveall调用时会被保存为文件
        一般情况下，继承它的类的init中需最后调用super().init()，防止读取的数据被覆盖
        '''
        if not Layer.issave:
            return
        if Layer.isload:
            self.load(Layer.layerlist[Layer.pointer])
            Layer.layerlist[Layer.pointer]=self
            Layer.pointer+=1
            if Layer.filelen==Layer.pointer:
                Layer.isload=False
        else:
            Layer.layerlist.append(self)

    def save(self):
        '''
        将自身转为str的形式。所有继承了Layer的类需重写此方法
        :return: str
        '''
        pass

    def load(self,*args):
        '''
        从str中读取数据到self
        :param args: str
        :return: None
        '''
        pass

    @classmethod
    def saveall(cls, name):
        '''
        存储所有layerlist中的实例
        :param name: str 保存的文件名
        :return: None
        '''
        with open(name,"w") as f:
            for i in Layer.layerlist:
                f.write(i.save()+"\n")

    @classmethod
    def loadall(cls, name):
        '''
        从文件中读取保存的内容，放到layerlist
        :param name: str 保存的文件名
        :return: None
        '''
        Layer.isload=True
        with open(name,"r") as f:
            Layer.layerlist=f.readlines()
            Layer.filelen=len(Layer.layerlist)

class Linear(Layer):
    def __init__(self,inpsize,outsize,bias=True):
        '''
        全连接层
        :param inpsize: int 作为输入的Ten的维度
        :param outsize: int 作为输出的Ten的维度
        :param bias: bool 是否加上偏置
        '''
        if not Layer.isload:
            self.w=[randinit(inpsize) for i in range(outsize)]
            self.bias=bias
            if bias:
                self.b=[randinit(1) for i in range(outsize)]
        super().__init__()

    def __call__(self,a):
        '''
        进行运算
        :param a: Ten
        :return: Ten
        '''
        if self.bias:
            o=[(self.w[i]*a).sum()+self.b[i] for i in range(len((self.w)))]
        else:
            o=[(self.w[i]*a).sum() for i in range(len((self.w)))]
        o=Ten.connect(o)
        return o

    def grad_descent_zero(self,k):
        '''
        进行梯度下降，并清空梯度
        :param k: float 步长
        :return: None
        '''
        for i in range(len(self.w)):
            self.w[i].graddescent(k)
            self.b[i].graddescent(k)
            self.w[i].zerograd()
            self.b[i].zerograd()

    def dcopy(self):
        '''
        对自己进行深拷贝
        注：如果不希望拷贝出的对象被保存，请在调用前把Layer.issave设定为False
        :return: Linear 拷贝出的对象
        '''
        c=Linear(len(self.w[0]),len(self.w))
        for i in range(len(self.w)):
            c.w[i]=Ten(self.w[i].data[:])
            c.b[i]=Ten(self.b[i].data[:])
        return c

    def save(self):
        t=str([i.data for i in self.w])
        if self.bias:
            t+="/"+str([i.data for i in self.b])
        return t

    def load(self,t):
        t=t.split("/")
        w=eval(t[0])
        self.w=[Ten(i) for i in w]
        if len(t)==2:
            self.bias=True
            b=eval(t[1])
            self.b=[Ten(i) for i in b]
        else:
            self.bias=False

class Ten2(Ten,Layer):
    '''
    与Ten一样，但是可以被保存
    '''
    def __init__(self,lis):
        Ten.__init__(self,lis)
        Layer.__init__(self)

    def save(self):
        t=str(self.data)
        return t

    def load(self,t):
        self.data=Vec(eval(t))

class Conv(Layer):
    def __init__(self,width,height,stride_w=1,stride_h=1,pad=True,bias=True):
        '''
        2d卷积层
        :param width: int 卷积核的宽度（如填充，请设为奇数）
        :param height: int 卷积核的高度（如填充，请设为奇数）
        :param stride_w: int 横向的步长
        :param stride_h: int 纵向的步长
        :param pad: bool 是否进行填充（使运算的输入和输出的大小一样）
        :param bias: bool 是否加上偏置
        '''
        if not Layer.isload:
            self.width=width
            self.height=height
            self.stride_h=stride_h
            self.stride_w=stride_w
            self.pad=pad
            self.kernel=randinit(width*height)
            self.bias=bias
            if bias:
                self.b=randinit(1)
        super().__init__()

    def padding(self,x):
        '''
        填充
        :param x: list[Ten(),Ten()...]  2d的Ten，或者说列表包着的一列Ten
        :return: list[Ten(),Ten()...]
        '''
        padx=(self.stride_w*(len(x[0])-1)-len(x[0])+self.width)//2
        pady=(self.stride_h*(len(x)-1)-len(x)+self.height)//2
        x2=[]
        for i in range(pady):
            x2.append(Ten.zero(len(x[0])+padx*2))
        for i in range(len(x)):
            x2.append(Ten.connect([Ten.zero(padx),x[i],Ten.zero(padx)]))
        for i in range(pady):
            x2.append(Ten.zero(len(x[0])+padx*2))
        return x2

    def __call__(self,x):
        '''
        进行运算
        :param x: list[Ten(),Ten()...]  2d的Ten，或者说list包着的一列Ten
        :return: list[Ten(),Ten()...]
        '''
        if self.pad:
            x=self.padding(x)
        x2=[]
        for ypos in range(0,len(x)-self.height+1,self.stride_h):
            x2line = []
            for xpos in range(0,len(x[0])-self.width+1,self.stride_w):
                window=Ten.connect([x[ypos+i].cut(xpos,xpos+self.width) for i in range(self.height)])
                v=(window*self.kernel).sum()
                if self.bias:
                    v+=self.b
                x2line.append(v)
            x2.append(Ten.connect(x2line))
        return x2

    def save(self):
        t=f"{self.width}/{self.height}/{self.kernel.data}/{self.stride_w}/{self.stride_h}/{self.pad}"
        if self.bias:
            t+=f"/{self.b.data}"
        return t

    def load(self,t):
        t = t.split("/")
        self.width=int(t[0])
        self.height=int(t[1])
        self.kernel=Ten(eval(t[2]))
        self.stride_w=int(t[3])
        self.stride_h=int(t[4])
        self.pad=eval(t[5])
        if len(t)==7:
            self.bias=True
            self.b=Ten(eval(t[6]))
        else:
            self.bias=False

    def grad_descent_zero(self,k):
        self.kernel.graddescent(k)
        self.kernel.zerograd()

class MultiConv:
    def __init__(self,inchannel,outchannel,width,height,stride_w=1,stride_h=1,pad=True,bias=True):
        '''
        多通道卷积层
        :param inchannel: int 输入通道数
        :param outchannel: int 输出通道数
        :param width: int 卷积核的宽度（如填充，请设为奇数）
        :param height: int 卷积核的高度（如填充，请设为奇数）
        :param stride_w: int 横向的步长
        :param stride_h: int 纵向的步长
        :param pad: bool 是否进行填充（使运算的输入和输出的大小一样）
        :param bias: bool 是否加上偏置
        '''
        self.cores=[[Conv(width,height,stride_w,stride_h,pad=False,bias=bias) for j in range(inchannel)] for i in range(outchannel)]
        self.pad=pad

    def __call__(self,x):
        '''
        进行运算
        :param x: list[list[Ten,Ten...],list[Ten,Ten...]...] 多通道的2dTen，被list包着的2dTen
        :return: list[list[Ten,Ten...],list[Ten,Ten...]...]
        '''
        if self.pad:
            x=[self.cores[0][0].padding(i) for i in x]
        x2=[]
        for chan in self.cores:
            xchan=sumchan2d([chan[i](x[i]) for i in range(len(chan))])
            x2.append(xchan)
        return x2

    def grad_descent_zero(self,k):
        for chan in self.cores:
            for i in chan:
                i.grad_descent_zero(k)

class Attention:
    def __init__(self,embedsize,qksize=None,vsize=None):
        '''
        单头自注意力模块
        :param embedsize: int 输入词向量维度
        :param qksize: int q、k维度
        :param vsize: int 输出词向量维度，默认与输入相同
        '''
        if qksize is None:
            qksize=embedsize//2
        if vsize is None:
            vsize=embedsize
        self.q=Linear(embedsize,qksize)
        self.k=Linear(embedsize,qksize)
        self.v=Linear(embedsize,vsize)
        self.embedsize=embedsize
        self.qksize=qksize
        self.outsize=vsize

    def __call__(self,x,masklist=None,trimask=False):
        '''
        进行运算
        :param x: list[Ten,Ten...]  装着词向量的列表
        :param masklist: list[int,int...] 用于在softmax前盖住填充，输入中表中为1的位置会被替换为-inf
        :param trimask: bool 是否使用三角掩码（在计算注意力权重时只关注当前和之前的词）
        :return: list[Ten,Ten...]
        '''
        qlist=[]
        klist=[]
        vlist=[]
        for w in x:
            qlist.append(self.q(w))
            klist.append(self.k(w))
            vlist.append(self.v(w))
        atlist=[]
        for i in range(len(qlist)):
            line=[]
            for j in range(len(klist)):
                if (masklist is not None and (masklist[i]==1 or masklist[j]==1))\
                 or (trimask and j>i):
                    line.append(Ten([float("-inf")]))
                else:
                    line.append((qlist[i]*klist[j]).sum()/Ten([self.qksize**0.5]))
            atlist.append(Ten.connect(line).softmax())
        newvlist=[]
        for i in range(len(qlist)):
            line=Ten.zero(self.outsize)
            for j in range(len(qlist)):
                line+=vlist[j]*(atlist[i].cut(j,j+1).expand(self.outsize))
            newvlist.append(line)
        return newvlist

    def grad_descent_zero(self, k):
        self.q.grad_descent_zero(k)
        self.k.grad_descent_zero(k)
        self.v.grad_descent_zero(k)

class MultiAtt:
    def __init__(self,headnum,embedsize,qksize=None,vsize=None):
        '''
        多头注意力模块
        :param headnum: int 注意力头数量
        :param embedsize: int 输入词向量维度
        :param qksize: int q、k维度
        :param vsize: int 输出向量维度
        '''
        self.heads=[Attention(embedsize,qksize,vsize) for i in range(headnum)]
        self.embedsize=embedsize

    def __call__(self,x,masklist=None,trimask=False):
        '''
        进行运算
        :param x: list[Ten,Ten...]  装着(多个词的词向量)的列表
        :return: list[Ten,Ten...]
        '''
        out=[h(x,masklist,trimask) for h in self.heads]
        out=sumchan2d(out)
        return out

    def grad_descent_zero(self,k):
        for i in self.heads:
            i.grad_descent_zero(k)

class LSTM:
    def __init__(self,embedsize,outputsize):
        self.forgetgate=Linear(embedsize+outputsize,outputsize)
        self.inputgate=Linear(embedsize+outputsize,outputsize)
        self.inputgate2=Linear(embedsize + outputsize, outputsize)
        self.outputgate=Linear(embedsize+outputsize,outputsize)
        self.h=Ten.zero(outputsize)
        self.s=Ten.zero(outputsize)

    def __call__(self,x):
        out=[]
        for i in x:
            i=Ten.connect([i,self.h])
            self.s*=self.forgetgate(i).sigmoid()
            self.s+=self.inputgate(i).sigmoid()*self.inputgate2(i).tanh()
            self.h=self.outputgate(i).sigmoid()*self.s.tanh()
            out.append(self.h)
        return out

    def grad_descent_zero(self,k):
        self.forgetgate.grad_descent_zero(k)
        self.inputgate.grad_descent_zero(k)
        self.inputgate2.grad_descent_zero(k)
        self.outputgate.grad_descent_zero(k)

class Norm:
    def __init__(self):
        '''
        标准化层
        '''
        self.w=Ten2([random.gauss(0,0.04)])
        self.b=Ten2([random.gauss(0,0.04)])

    def __call__(self,x:Ten,eps=0.0001):
        n=Ten([len(x)])
        mean=(x.sum()/n).expand(len(x))
        sigma=(((x-mean)**2).sum()/n)**0.5
        std=(x-mean)/(Ten([eps])+sigma).expand(len(x))
        out=self.w.expand(len(x))*std+self.b.expand(len(x))
        return out

    def grad_descent_zero(self,k):
        self.w.graddescent(k)
        self.b.graddescent(k)

class MiniTransformer:
    def __init__(self,headnum,embsize,windowsize,lowrank=False):
        self.a=MultiAtt(headnum,embsize)
        if lowrank:
            self.f1=MiniLinear(embsize*windowsize,embsize*windowsize)
            self.f2=MiniLinear(embsize*windowsize,embsize*windowsize)
        else:
            self.f1=Linear(embsize*windowsize,embsize*windowsize)
            self.f2=Linear(embsize*windowsize,embsize*windowsize)
        self.n1=Norm()
        self.n2=Norm()
        self.embsize=embsize
        self.windowsize=windowsize

    def __call__(self,x,masklist=None,trimask=False):
        x2=x    # 2dTen
        x=Ten.connect(self.a(x,masklist,trimask))
        x=self.n1(x)
        x+=Ten.connect(x2)
        x2=x    # 1dTen
        x=self.f2(self.f1(x).gelu())
        x=self.n2(x)
        x+=x2
        x=resize2d(x,self.embsize,self.windowsize)
        return x

    def grad_descent_zero(self,k):
        self.a.grad_descent_zero(k)
        self.f1.grad_descent_zero(k)
        self.f2.grad_descent_zero(k)
        self.n1.grad_descent_zero(k)
        self.n2.grad_descent_zero(k)

class Transformer:
    def __init__(self,headnum,embsize):
        self.a=MultiAtt(headnum,embsize)
        self.f1=Linear(embsize,embsize*4)
        self.f2=Linear(embsize*4,embsize)
        self.n1=Norm()
        self.n2=Norm()
    
    def __call__(self,x,masklist=None,trimask=False):
        x2=x
        x=self.a(x,masklist,trimask)
        x=[self.n1(i) for i in x]
        x=[x[i]+x2[i] for i in range(len(x))]
        
        x2=x
        x=[self.f2(self.f1(i).relu()) for i in x]
        x=[self.n2(i) for i in x]
        x=[x[i]+x2[i] for i in range(len(x))]
        return x

    def grad_descent_zero(self,k):
        self.a.grad_descent_zero(k)
        self.f1.grad_descent_zero(k)
        self.f2.grad_descent_zero(k)
        self.n1.grad_descent_zero(k)
        self.n2.grad_descent_zero(k)

class MiniLinear:
    def __init__(self,inpsize,outsize,midsize=None,bias=True):
        if midsize is None:
            midsize=round(((inpsize+outsize)/2)**0.5)
        self.f1=Linear(inpsize,midsize,bias)
        self.f2=Linear(midsize,outsize,bias)

    def __call__(self,x):
        x=self.f1(x)
        x=self.f2(x)
        return x

    def grad_descent_zero(self,k):
        self.f1.grad_descent_zero(k)
        self.f2.grad_descent_zero(k)

def randinit(size):
    '''
    初始化权重
    :param size: int 权重的维度大小
    :return: Ten
    '''
    sigma=(2/size)**0.5/5#0.04 只有定值才能使GAN正常训练？
    return Ten([random.gauss(0,sigma) for i in range(size)])

def sumchan2d(x):
    '''
    对多通道的2dTen求和，变成单通道2dTen
    :param x: list[list[Ten,Ten...],list[Ten,Ten...]...]
    :return: list[Ten,Ten...]
    '''
    out=x[0][:]
    for pici in range(1,len(x)):
        for linei in range(len(x[0])):
            out[linei]+=x[pici][linei]
    return out

def resize2d(x,embsize,windowsize):
    '''
    把一维Ten转为2dTen
    :param x: Ten
    :param embsize: int 每个Ten的大小
    :param windowsize: int 多少个Ten
    :return: list[Ten,Ten...]
    '''
    x2=[]
    for i in range(windowsize):
        x2.append(x.cut(i*embsize,(i+1)*embsize))
    return x2

def func2d(x,func):
    '''
    对2dTen进行函数操作
    例如，把输入的2dTen放入激活函数Relu中: x=func2d(x,Ten.relu)
    :param x: list[Ten,Ten...]
    :param func: function Ten里面的函数
    :return: list[Ten,Ten...]
    '''
    return [func(i) for i in x]

def gradtest(func,xten):
    x=func(xten)
    xten2=Ten(xten.data)
    xten2.data[0]+=0.001
    x2=func(xten2)
    return (x2.data[0]-x.data[0])/0.001

def test():
    x=Ten([1])
    y=Ten([1])
    z=Ten([1])

    for i in range(1000):
        s1=((x*y+z)-Ten([40]))**2
        s2=((x*z+y)-Ten([51]))**2
        s3=((x+y+z)-Ten([19]))**2
        if s1.data[0]<0.01 and s2.data[0]<0.01 and s3.data[0]<0.01:
            break
        s1.back(clean=False)
        s2.back(clean=False)
        s3.back()
        # s1.grad=Vec([1])
        # s2.grad = Vec([1])
        # s3.grad = Vec([1])
        # Operator.back()
        x.data -= x.grad * Vec([0.002])
        y.data -= y.grad * Vec([0.002])
        z.data -= z.grad * Vec([0.002])
        print(f"x{x},y{y},z{z}")
        x.zerograd()
        y.zerograd()
        z.zerograd()


