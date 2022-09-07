from popup_lib.popup2 import transform_period_for_each, is_number, Abstract_Agent, Family_trainer
import numpy as np
from typing import List,Dict

def test_transform_period_for_each():
    assert transform_period_for_each("10seconds")==(10,"second")
    assert transform_period_for_each("1.2seconds")==(1.2,"second")
    assert transform_period_for_each("10 seconds")==(10,"second")
    assert transform_period_for_each("10 secondes")==(10,"second")
    assert transform_period_for_each("10 seconde")==(10,"second")
    assert transform_period_for_each("10 minutes")==(10*60,"second")
    assert transform_period_for_each("10.5 steps")==(10.5,"step")

def test_is_number():
    assert is_number(3)
    assert is_number(3.2)
    assert is_number(np.array([3],dtype=np.float32)[0])
    assert not is_number(np.array([3]))
    assert not is_number([3,4])
    assert is_number(np.nan)


class Agent_test(Abstract_Agent):

    def optimize_and_return_score(self) -> float:
        self.weights[0]+=1
        return np.random.uniform(0,2)+self.famparams["normal"]+self.famparams["decadence"]

    def get_famparams(self) -> Dict[str, float]:
        return self.famparams

    def set_and_perturb_famparams(self, famparam, period_count: int) -> None:
        self.famparams["normal"]+=1
    def perturb_famparams_on_decadence(self,period_count:int) ->None:
        self.famparams["decadence"] += 2

    def set_weights(self, weights: List):
        self.weights=weights

    def get_copy_of_weights(self) -> List:
        return self.weights.copy()

    def __init__(self):
        self.famparams={"normal":0,"decadence":0}
        self.weights=[np.ones([2])]


def test_decadence():

        agents=[Agent_test() for _ in range(3)]
        fm=Family_trainer(agents,"5 step",nb_bestweights_averaged=2,nb_strong=2)
        for _ in range(30):
            fm.period()
        fm.get_best_agent()


if __name__=="__main__":
    test_decadence()
    test_is_number()
