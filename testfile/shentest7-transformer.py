from Shen import *
import math,os,time,random

'''
text : str
token : [str,str...]
index : [int,int...]
embed : [Ten,Ten...]
'''

def gettokens(text,tnum,maxlenth=200000,mincount=2):
    biao=[]
    text=text[:maxlenth]
    for i in text:
        if i not in biao:
            biao.append(i)
    texttoken=[i for i in text]
    paichu=(" ", "\n", ",", ":", ";",
            "<", ">", "/", "?", "!",
            "=", "-", "(", ")", "，",
            "：", "；", "《", "》", "、",
            "？", "！", "，", "。", "（", "）", "“", "”",
            "1", "2", "3", "4", "5", "6", "7", "8", "9", "0")
    fqdict=dict()
    for i in range(tnum):
        previous_score=0
        maxtoken=""
        tokenp1=""
        tokenp2=""
        for j in range(0,len(texttoken)-1):
            if texttoken[j] in paichu or texttoken[j+1] in paichu:
                continue
            t=texttoken[j]+texttoken[j+1]
            if t in fqdict:
                c=fqdict[t]
            else:
                c=counttext(text,t)
                fqdict[t]=c
            if c<mincount:
                continue
            if texttoken[j] in fqdict:
                cl=fqdict[texttoken[j]]
            else:
                cl=counttext(text,texttoken[j])
                fqdict[texttoken[j]]=cl
            if texttoken[j+1] in fqdict:
                cr=fqdict[texttoken[j+1]]
            else:
                cr=counttext(text,texttoken[j+1])
                fqdict[texttoken[j+1]]=cr

            score=c**2.2/(cl*cr)
            if score>previous_score:
                tokenp1=texttoken[j]
                tokenp2=texttoken[j+1]
                maxtoken=t
                previous_score=score
        if maxtoken!="":
            biao.append(maxtoken)
        k=0
        while k<len(texttoken)-1:
            if texttoken[k]==tokenp1 and texttoken[k+1]==tokenp2:
                texttoken[k]=maxtoken
                texttoken.pop(k+1)
            else:
                k+=1
        #print(i,maxtoken)
    i=0
    while i <len(biao):
        if biao[i] not in texttoken:
            biao.remove(biao[i])
        else:
            i+=1
    biao.extend(["/sta","/unk","/end","/pad"])
    return biao
def counttext(text,target):
    beg=0
    v=0
    while True:
        ind=text.find(target,beg)
        if ind==-1:
            break
        v+=1
        beg=ind+len(target)
    return v

def make_embedding(num,dim):
    return [Ten2([random.gauss(0,0.04) for i in range(dim)]) for i in range(num)]

def text2token(text:str,biao):
    y = []#"/sta"
    maxlen = max(len(i) for i in biao)
    lentext = len(text)
    while lentext > 0:
        cmaxlen = maxlen
        while cmaxlen > 0:
            token = text[:cmaxlen]
            if token in biao:
                y.append(token)
                break
            elif cmaxlen == 1:
                y.append("/unk")
                break
            else:
                cmaxlen -= 1
        text = text[cmaxlen:]
        lentext -= cmaxlen
    #y.append("/end")
    return y

def text2index(text,biao):
    y=text2token(text,biao)
    for i in range(len(y)):
        y[i]=biao.index(y[i])
    return y

def text2embed(text:str,biao,embed,pbiao):
    y=[]
    index=text2index(text,biao)
    for i in range(len(text)):
        y.append(embed[index[i]]+pbiao[i])
    return y

def token2embed(token,biao,embed,pbiao):
    y=[]
    for i in range(len(token)):
        y.append(embed[biao.index(token[i])]+pbiao[i])
    return y

def position_embed(length,dimension):
    y=[]
    for i in range(length):
        w=[]
        if i%2==0:
            for d in range(dimension):
                w.append(math.sin(i/10000**(d/dimension)))
        else:
            for d in range(1,dimension+1):
                w.append(math.cos(i/10000**((d-1)/dimension)))
        y.append(Ten(w))
    return y

def one_hot(biao_size,one_index):
    x=Ten.zero(biao_size)
    x.data[one_index]=1
    return x

class model:
    def __init__(self,embed_dim,layer_n,head_n,biao_size,window_size=None):
        self.embed=[Ten2([random.gauss(0,0.04) for j in range(embed_dim)]) for i in range(biao_size)]
        self.pos_embed=position_embed(1000,embed_dim)
        self.layer=[Transformer(head_n,embed_dim) for i in range(layer_n)]
        self.last=Linear(embed_dim,biao_size)
        
        self.embed_dim=embed_dim
        self.biao_size=biao_size
        self.window_size=window_size
    
    def forward(self,x,t=1):
        # x=[Ten,Ten...]
        if self.window_size is not None:
            x=x[-self.window_size:]
        for l in self.layer:
            x=l(x,None,True)
        x=(self.last(x[-1])/Ten([t]).expand(self.biao_size)).softmax()
        return x
    
    def forward_token(self,x,biao,t=1):
        # x=[str,str...]
        if len(x)>len(self.pos_embed):
            self.pos_embed=position_embed(len(x),self.embed_dim)
        x=token2embed(x,biao,self.embed,self.pos_embed)
        x=self.forward(x,t)
        return x
    
    def forward_text(self,x,biao,t=1):
        x=text2token(x,biao)
        x.insert(0,"/sta")
        x=self.forward_token(x,biao,t)
        return x
    
    def optimize(self,k):
        for i in self.embed:
            i.graddescent(k)
            i.zerograd()
        for i in self.layer:
            i.grad_descent_zero(k)
        self.last.grad_descent_zero(k)

def train(m:model,texts,biao,batchsize=64,k=0.0001,shuffle=True):
    back_count=0
    aloss=0
    t0=time.perf_counter()
    if shuffle:
        random.shuffle(texts)
    for t in texts:
        token=text2token(t,biao)
        in_token=["/sta"]
        for i in range(len(token)):
            in_token.append(token[i])
            if i==len(token)-1:
                out_token="/end"
            else:
                out_token=token[i+1]
            target=one_hot(len(biao),biao.index(out_token))
            
            out=m.forward_token(in_token,biao)
            loss=Ten.nll(out,target)
            aloss+=loss.data[0]
            Operator.back()
            back_count+=1
            
            if back_count%batchsize==0:
                m.optimize(k/batchsize)
                t1=time.perf_counter()
                t=t1-t0
                t0=t1
                Layer.saveall(save_name)
                print(f"batch {back_count/batchsize},loss {aloss/batchsize},time {t}")
                aloss=0

def run(m,text,biao,t=1):
    print(text,end="")
    while True:
        out=m.forward_text(text,biao,t)
        Operator.clean()
        sample=biao[out.data.index(random.choices(out.data,weights=out.data)[0])]
        if sample=="/end":
            return
        print(sample,end="")
        text+=sample

def gushi_split(gushi_text:str):
    texts=[]
    t=""
    for i in gushi_text:
        if i.isdigit():
            if t!="":
                texts.append(t)
                t=""
        else:
            t+=i
    return texts

text_name="古诗300首"
file_name=text_name+".txt"
with open(file_name,"r",encoding="utf-8") as f:
    text_file=f.read()
print(f"text:{text_name}")

biao_name=text_name+"biao"+".txt"
if biao_name in os.listdir():
    with open(biao_name,"r",encoding="utf-8") as f:
        biao=eval(f.read())
        print("load",biao_name)
else:
    biao=gettokens(text_file,1000)
    with open(biao_name,"w",encoding="utf-8") as f:
        f.write(str(biao))
        print("make",biao_name)


save_name="slm"+text_name+"-test"#"-window30" "-test"
if save_name in os.listdir():
    Layer.loadall(save_name)
    print("load",save_name)

m=model(embed_dim=10,layer_n=2,head_n=2,biao_size=len(biao),window_size=30)#test
# m=model(embed_dim=30,layer_n=6,head_n=8,biao_size=len(biao),window_size=30)
texts=gushi_split(text_file)
while True:
    train(m,texts,biao,batchsize=15,k=0.003)
    Layer.saveall(save_name)
# run(m,"",biao,t=0.9)
