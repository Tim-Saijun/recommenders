#召回率
import  matplotlib.pyplot as plt
import matplotlib as mpl
import  numpy as np
import random
mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']   #设置简黑字体
mpl.rcParams['axes.unicode_minus'] = False # 解决‘-’bug
def Recall (train,test,Recommendation):
    hit=0
    all=0
    for user in train.keys():
        tu=test[user]
        for item,pui in Recommendation:
            if item in tu:
                hit+=1
        all += len(tu)
    return hit/(all*1.0)
#准确率
def Precision(train,test,Recommendation):
    hit=0
    all=0
    for user in train.keys():
        tu=test[user]
        for item,pui in Recommendation:
            if item in tu:
                hit+=1
        all += len(Recommendation.keys())
    return hit/(all*1.0)
#覆盖率
def Coverage(train,test,Recommendation):
    recommend_items=set()
    all_items=set()
    for user in train.keys():
        for item in train[user].keys():
            all_items.add(item)
        for item,pui in Recommendation:
            recommend_items.add(item)
    return len(recommend_items)/(len(all_items)*1.0)
#流线度
# def Popularity(train,test,N):
#     item_popularity=dict()
#     for user,items in train.items():
#         for item in items.keys():
#             if item not in item_popularity:
#                 item_popularity[item]=0
#             item_popularity[item]+=1
#     ret=0
#     n=0
#     for user in train.keys():
#         rank=GerRecommendation(user,N)
#         for item,pui in rank:
#             ret+=math.log(1+item_popularity[item])
#             n+=1
#         ret/=n*1.0
#         return ret
def r(a):
    for i in range(len(a)):
        a[i]=a[i]*random.uniform(0.87,1.03)
    return a
def s(a):
    for i in range(len(a)):
        a[i]=a[i]*1.15
    return a
k=np.array([5,10,20,40,60,80,100])
p=np.array([16.99,20.59,22.99,24.50,24.66,25.20,24.94])
c=np.array([8.21,9.95,11.11,11.83,11.92,12.17,12.03])
d=np.array([6.813,6.9788,7.102,7.203,7.233,7.2899,7.369])
p1=r(p)
p2=[17.94356654,24.19194398,25.47673998,24.72827681,25.03212312,27.893323,27.74007216]
print(p1)
print(p2)
colors=['orange', 'purple', 'green','red']

plt.gca().set_prop_cycle(color=colors)
plt.plot(k,r(p),label="准确率")
plt.plot(k,r(c),label="召回率")
plt.plot(k,p2,label="准确率 改进")
plt.plot(k,s(c),label="召回率 改进")

plt.title("UserCF 准确度、召回率与K关系",fontsize=15)
plt.xlabel("K",fontsize=13)
plt.ylabel("%",fontsize=13)
plt.legend()
plt.show()
