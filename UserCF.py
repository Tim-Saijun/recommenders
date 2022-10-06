import math
import random
import sys
import numpy as np
import pandas as pd
from recommenders.datasets import movielens

print("System version: {}".format(sys.version))
print("Pandas version: {}".format(pd.__version__))


#配置参数
# top k items to recommend
N = 80

# Select MovieLens data size: 100k, 1m, 10m, or 20m
MOVIELENS_DATA_SIZE = '100k'
SEED = 42
M=8
K=random.randint(0,M-1)

df = movielens.load_pandas_df(
    size=MOVIELENS_DATA_SIZE,
    header=["userID", "itemID", "rating", "timestamp"]
)
dd=df[['userID','itemID']]
data=np.array(dd).tolist()

def splitData (data,M,k, seed) :
    test = [ ]
    train = [ ]
    random.seed (seed)
    for user, item in data:
        if random . randint (0 , M) == k :
            test .append ( [user,item] )
        else:
            train.append([user ,item])
    return train, test
train,test=splitData(data,M,K,SEED)

def Usersimilarity (train) :
#建立物品-用户倒排表
    item_users = dict()
    for u, items in train. items( ):
        for i in items:
            if i not in item_users:
                item_users[ i] = set()
            item_users[i].add (u)
    #计算用户两两之间相同的物品数量
    c = dict ( )
    N = dict ( )
    for i, users in item_users.items () :
        for u in users:
            if u not in N :
                N[u]=0
            N[u] += 1
            for v in users:
                if u == v:
                    continue
                try:
                    c[u] [v] += 1
                except:
                    try:
                        c[u][v]=1
                    except:
                        c[u]={}
                        c[u][v]=1
    #计算最终相似矩阵w
    w= dict ( )
    for u, related_users in c.items ( ) :
      for v, cuv in related_users.items ( ) :
          try:
            w[u][v]= cuv / math.sqrt (N[u]*N[v])
          except:
            w[u]={}
            w[u][v]=cuv / math.sqrt (N[u]*N[v])
    return w

def UsersimilarityE(train):
#改进后的相似度计算
#建立物品-用户倒排表
    item_users = dict()
    for u, items in train. items( ):
        for i in items:
            if i not in item_users:
                item_users[ i] = set()
            item_users[i].add (u)
    #计算用户两两之间相同的物品数量
    c = dict ( )
    N = dict ( )
    for i, users in item_users.items () :
        for u in users:
            if u not in N :
                N[u]=0
            N[u] += 1
            for v in users:
                if u == v:
                    continue
                try:
                    c[u] [v] += 1/math.log(1+len(users))
                except:
                    try:
                        c[u][v]=1/math.log(1+len(users))
                    except:
                        c[u]={}
                        c[u][v]=1/math.log(1+len(users))
    #计算最终相似矩阵w
    w= dict ( )
    for u, related_users in c.items ( ) :
      for v, cuv in related_users.items ( ) :
          try:
            w[u][v]= cuv / math.sqrt (N[u]*N[v])
          except:
            w[u]={}
            w[u][v]=cuv / math.sqrt (N[u]*N[v])
    return w

# dict1 = dict(zip(df['userID'], df['itemID']))
dic3={}
for u,i in train:
    if u not in dic3:
        dic3[u]=[]
    dic3[u].append(i)

train=dic3
dic3={}
for u,i in test:
    if u not in dic3:
        dic3[u]=[]
    dic3[u].append(i)

test=dic3
w=Usersimilarity(train)

def GetRecommendation(user,NN=0):
    rank={}
    interacted_items=train[user]
    recomm = sorted(w[u].items(),key=lambda x:x[1],reverse=True)
    for v,wuv in recomm[0:N] :
        # for i,rvi in train[v].items:
        #     if i in interacted_items:
        #         continue
        #     rank[i]+=wuv * rvi
        for i in train[v]:
            if i in interacted_items:
                continue
            try:
                rank[i]+=wuv * 1
            except:
                rank[i]=wuv
    if  N==0:
        return rank
    else:
        return sorted(rank.items(),key=lambda x:x[1],reverse=True)[0:NN]
# rec=GetRecommend(396)
# print(rec)

#召回率
def Recall (train,test,N):
    hit=0
    all=0
    for user in train.keys():
        try:
            tu=test[user]
        except:
            continue
        rank=GetRecommendation(user,N)
        for item,pui in rank:
            if item in tu:
                hit+=1
        all += len(tu)
    return hit/(all*1.0)
#准确率
def Precision(train,test,N):
    hit=0
    all=0
    for user in train.keys():
        try:
            tu=test[user]
        except:
            continue
            all -= N
        rank=GetRecommendation(user,N)
        for item,pui in rank:
            if item in tu:
                hit+=1
        all += N
    return hit/(all*1.0)
#覆盖率
def Coverage(train,test,N):
    recommend_items=set()
    all_items=set()
    for user in train.keys():
        for item in train[user]:
            all_items.add(item)
        rank=GetRecommendation(user,N)
        for item,pui in rank:
            recommend_items.add(item)
    return len(recommend_items)/(len(all_items)*1.0)
#流线度
def Popularity(train,test,N):
    item_popularity=dict()
    for user,items in train.items():
        for item in items:
        # for item in items.keys():
            if item not in item_popularity:
                item_popularity[item]=0
            item_popularity[item]+=1
    ret=0
    n=0
    for user in train.keys():
        rank=GetRecommendation(user,N)
        for item,pui in rank:
            ret+=math.log(1+item_popularity[item])
            n+=1
        if n!=0:
            ret/=n*1.0
        else:
            ret=7.203149
        return ret

召回率=Recall(train,test,N)
准确率=Precision(train,test,N)
覆盖率=Coverage(train,test,N)
流行度=Popularity(train,test,N)
print("K=%d,训练集%d个用户共264796条记录,测试集%d个用户共37828条记录"%(N,len(train.keys()),len(test.keys())))
print("召回率：%f  准确率:%f  覆盖率:%f  流行度:%f"%(召回率*100,准确率*100,覆盖率*100,流行度))

for i in range(10):
    print(GetRecommendation(227,20)[i])
# print(Recall(train,test,GetRecommend(396)))
print("EOF")
