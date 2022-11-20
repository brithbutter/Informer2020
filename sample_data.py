#%%
import random
import pandas as pd
import numpy as np
from markov_chain import Markov_Chain_Env
CYCLE = 20
# %%
df = pd.read_csv("data/ETT/ETTh1.csv")
workloads = [10,10,10,14,12,10]
# %%
df

# %%
df = df.drop(columns=["HUFL","HULL","MUFL","MULL","LUFL"])

# %%
#======================
sigma=0.04
values1 = [1,3,9,12]
values2 = [2,6,12,18]
values3 = [4,7,10,16]
values4 = [3,8,14,20]
values5 = [5,10,11,19]
scales1 = [0.01,0.05,0.1,0.5]
scales2 = [0.01,0.05,0.1,0.5]
scales3 = [0.01,0.05,0.1,0.5]
scales4 = [0.01,0.05,0.1,0.5]
scales5 = [0.01,0.05,0.1,0.5]

data1 = np.array([])
data2 = np.array([])
data3 = np.array([])
data4 = np.array([])
data5 = np.array([])

data6 = np.array([])
data7 = np.array([])
data8 = np.array([])
data9 = np.array([])
data10 = np.array([])

env1 = Markov_Chain_Env(values1)
env2 = Markov_Chain_Env(values2)
env3 = Markov_Chain_Env(values3)
env4 = Markov_Chain_Env(values4)
env5 = Markov_Chain_Env(values5)

env6 = Markov_Chain_Env(scales1)
env7 = Markov_Chain_Env(scales2)
env8 = Markov_Chain_Env(scales3)
env9 = Markov_Chain_Env(scales4)
env10 = Markov_Chain_Env(scales5)
# for i in range (len(df)):
    
#     data1 = data1.append(np.random.poisson(lam=4,size=(60)))
#     data2 = data2.append(np.random.poisson(lam=8,size=(60)))
# data1 = (np.clip(data1,a_min=1,a_max=7)-4)
# data1=np.cumsum(data1)
# data1=sigma*data1
# data2 = (np.clip(data2,a_min=1,a_max=15)-8)/1.2
# data2=np.cumsum(data2)
# data2=sigma*data2
#======================

poisson1 = np.random.poisson(lam = env1.current_state.value,size=CYCLE)
poisson2 = np.random.poisson(lam = env2.current_state.value,size=CYCLE)
poisson3 = np.random.poisson(lam = env3.current_state.value,size=CYCLE)
poisson4 = np.random.poisson(lam = env5.current_state.value,size=CYCLE)
poisson5 = np.random.poisson(lam = env5.current_state.value,size=CYCLE)
expo1 = np.random.exponential(scale=env6.current_state.value,size=CYCLE)
expo2 = np.random.exponential(scale=env7.current_state.value,size=CYCLE)
expo3 = np.random.exponential(scale=env8.current_state.value,size=CYCLE)
expo4 = np.random.exponential(scale=env9.current_state.value,size=CYCLE)
expo5 = np.random.exponential(scale=env10.current_state.value,size=CYCLE)

for i in range (len(df)):
    data1 = np.append(data1,sum(np.random.choice(workloads,size= poisson1[i%CYCLE]))-20)
    data2 = np.append(data2,sum(np.random.choice(workloads,size= poisson2[i%CYCLE]))-20)
    data3 = np.append(data3,sum(np.random.choice(workloads,size= poisson3[i%CYCLE]))-20)
    data4 = np.append(data4,sum(np.random.choice(workloads,size= poisson4[i%CYCLE]))-20)
    data5 = np.append(data5,sum(np.random.choice(workloads,size= poisson5[i%CYCLE]))-20)
    data6 = np.append(data6,expo1[[i%CYCLE]])
    data7 = np.append(data7,expo2[[i%CYCLE]])
    data8 = np.append(data8,expo3[[i%CYCLE]])
    data9 = np.append(data9,expo4[[i%CYCLE]])
    data10 = np.append(data10,expo5[[i%CYCLE]])
    if i% CYCLE==0:
        env1.step()
        env2.step()
        env3.step()
        env4.step()
        env5.step()
        env6.step()
        env7.step()
        env8.step()
        env9.step()
        env10.step()
        poisson1 = np.random.poisson(lam = env1.current_state.value,size=CYCLE)
        poisson2 = np.random.poisson(lam = env2.current_state.value,size=CYCLE)
        poisson3 = np.random.poisson(lam = env3.current_state.value,size=CYCLE)
        poisson4 = np.random.poisson(lam = env5.current_state.value,size=CYCLE)
        poisson5 = np.random.poisson(lam = env5.current_state.value,size=CYCLE)
        expo1 = np.random.exponential(scale=env6.current_state.value,size=CYCLE)
        expo2 = np.random.exponential(scale=env7.current_state.value,size=CYCLE)
        expo3 = np.random.exponential(scale=env8.current_state.value,size=CYCLE)
        expo4 = np.random.exponential(scale=env9.current_state.value,size=CYCLE)
        expo5 = np.random.exponential(scale=env10.current_state.value,size=CYCLE)
data1[data1 < 0]=0
data2[data2 < 0]=0
data3[data3 < 0]=0
data4[data4 < 0]=0
data5[data5< 0]=0
datas = [data3,data4,data5,data6,
        data7,data8,data9,data10]

# data1 = np.random.exponential(scale=3,size= len(df))
# data1 = data1/max(data1)
# data2 = np.random.exponential(scale=6,size= len(df))
# data2 = data2/max(data2)
# print(len(data1[0]))

#%%
# df = pd.DataFrame(d,index=range(len(df)))
for i in range(8):
    df["data{}".format(i)] = datas[i]
for i in range(len(df)):
    # print(len(data1[i]))
    # pr = random.uniform(0,1)
    # count = 1
    # while pr < random.uniform(0,1):
    #     count += 1
    # df.iloc[i,1] = sum(data1[0][i*60:(i+1)*60])
    # df.iloc[i,2] = sum(data2[0][i*60:(i+1)*60])
    df.iloc[i,1] = (data1[i])
    df.iloc[i,2] = (data2[i])
    
# %%
df
# %%
df.to_csv("Poisson.csv",index=False)
# %%
test = np.load("./results/informer_TEST_ftM_sl96_ll48_pl24_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_test_1/pred.npy")
# %%
test
# %%
true = np.load("./results/informer_TEST_ftM_sl96_ll48_pl24_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_test_1/true.npy")
# %%
true
# %%
