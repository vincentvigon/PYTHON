
import popup_lib.popup2 as pop
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict


class Agent_ultra_toy(pop.Abstract_Agent):

    def __init__(self):
        self.wei0=np.array([0.])
        self.wei1=np.array([0.])
        self.famparams={"add0":0,"add1":0}

    #Abstract_Agent: obligatoire
    def get_famparams(self):
        return self.famparams

    #Abstract_Agent: obligatoire
    def set_and_perturb_famparams(self,famparams,period_count):
        self.famparams=famparams
        self.famparams["add0"]+=np.random.uniform(-1,1)
        self.famparams["add1"]+=np.random.uniform(-1,1)

    #Abstract_Agent: obligatoire
    def optimize_and_return_score(self) -> float:
        #ce n'est pas vraiment une optimization ici
        self.wei0 +=self.famparams["add0"]
        self.wei1 +=self.famparams["add1"]
        return self.wei0[0] - self.wei1[0]

    #Abstract_Agent: obligatoire
    def set_weights(self, weights):
        self.wei0,self.wei1=weights

    #Abstract_Agent: obligatoire
    def get_copy_of_weights(self):
        return [self.wei0,self.wei1]

    #Abstract_Agent: facultatif: pour observer les quantités (ici les poids du modèle)
    def to_register_at_period_end(self) ->Dict[str,float]:
        return {"wei0":self.wei0[0],"wei1":self.wei1[0]}



def test():
    agents = [Agent_ultra_toy(), Agent_ultra_toy()]
    family_trainer = pop.Family_trainer(agents, period_duration="10 steps", nb_strong=1)

    for _ in range(20):
        family_trainer.period()

    family_trainer.plot()

    family_trainer.plot("time")

    family_trainer.plot("score",["wei0","wei1"])

    plt.show()


