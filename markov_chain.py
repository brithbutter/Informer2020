
#%%
import random
#%%

def check_range(checker,value):
    index = 0
    for i in range(len(checker)):
        if checker[i]>=value:
            index = i
            break            
    return index

class Makkov_State:
    def __init__(self,value) -> None:
        self.value = value
        self.state_transition = []
        pass
    def add_transition(self,state,probability):
        self.state_transition.append({"state":state,"probability":probability})
    def transit(self):
        total_probability = sum([x["probability"]for x in self.state_transition])
        checker = []
        for x in self.state_transition:
            checker.append(x["probability"]/total_probability)
        index = check_range(checker=checker,value=random.uniform(0,1))
        return self.state_transition[index]["state"]
class Markov_Chain:
    def __init__(self,states) -> None:
        states = states
        pass
class Markov_Chain_Env:
    def __init__(self,values) -> None:
        self.state1 = Makkov_State(values[0])
        self.state2 = Makkov_State(values[1])
        self.state3 = Makkov_State(values[2])
        self.state4 = Makkov_State(values[3])
        # self.state1.add_transition(state=self.state1,probability=0.2)
        self.state1.add_transition(state=self.state2,probability=1)
        # self.state1.add_transition(state=self.state3,probability=0.7)
        self.state2.add_transition(state=self.state1,probability=0.4)
        # self.state2.add_transition(state=self.state2,probability=0.1)
        self.state2.add_transition(state=self.state3,probability=0.6)
        self.state2.add_transition(state=self.state4,probability=0.6)
        
        self.state3.add_transition(state=self.state2,probability=0.3)
        # self.state3.add_transition(state=self.state3,probability=0.1)
        # self.state3.add_transition(state=self.state4,probability=0.7)
        # self.state3.add_transition(state=self.state4,probability=0.3)
        self.state4.add_transition(state=self.state2,probability=0.7)
        # self.state4.add_transition(state=self.state4,probability=0.3)
        # self.state4.add_transition(state=self.state1,probability=0.5)
        self.current_state = self.state1
        pass
    def step(self):
        self.current_state = self.current_state.transit()
# %%
import numpy as np
print(max(np.random.exponential(scale=1,size=5000)))
print(min(np.random.exponential(scale=1,size=5000)))
# %%
import matplotlib.pyplot as plt

plt.plot(np.arange(start=1,stop=5001), np.random.exponential(scale=1.5,size=5000), 'o', color='black')

# %%
