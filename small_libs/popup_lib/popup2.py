import numpy as np
import time
from typing import List,Dict
from abc import ABC, abstractmethod
from collections import deque
pp=print
import copy
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
print("Popup2 loaded!")


class History:

    def __init__(self):

        self.localTime_previous_cumulated_values = 0
        # self.score_times =  []
        # self.metrics_times:Dict[str,List[float]] = {}
        # self.metrics_values:Dict[str,List[float]]={}
        self.localTime_isActive=False

    """Important: il faut démarrer le local time dès que l'agent est actif. Puis l'arrêter dès qu'il est inactif """
    def start_local_time(self):
        self.localTime_restart=time.time()
        assert not self.localTime_isActive , "on a oublié d'arrêter le localTime"
        self.localTime_isActive =True

    def stop_local_time(self):
        if not self.localTime_isActive:
            return
        self.localTime_previous_cumulated_values=self.get_local_time()
        self.localTime_isActive =False

    def get_local_time(self):
        if not self.localTime_isActive: #c'est logique
            return self.localTime_previous_cumulated_values
        else:
            return time.time()-self.localTime_restart+self.localTime_previous_cumulated_values

    #
    # def record_metric(self, name,value):
    #     assert is_number(value) ,f"Error when recording the value of '{name}': must be a number, but is: {value} of type {type(value)}"
    #     metric_times=self.metrics_times.get(name,[])
    #     metric_values=self.metrics_values.get(name,[])
    #     metric_times.append(self.get_local_time())
    #     metric_values.append(value)
    #     self.metrics_times[name]=metric_times
    #     self.metrics_values[name]=metric_values


def is_number(x):
    return isinstance(x, (int, float, np.float32, np.float64, np.int32, np.int64))

class Abstract_Agent(ABC):

    @abstractmethod
    def optimize_and_return_score(self)->float:
        pass

    @abstractmethod
    def get_famparams(self)->Dict[str,float]:
        pass

    @abstractmethod
    def set_and_perturb_famparams(self, famparam, period_count:int)->None:
        pass

    """ faut-il plutot demandé un dico à l'utilisateur ? """
    @abstractmethod
    def set_weights(self,weights:List):
        pass

    @abstractmethod
    def get_copy_of_weights(self)->List:
        pass
    #
    # #facultatif
    # def perturb_famparams_on_decadence(self,period_count:int)->None:
    #     """
    #     Quand le score courrant de l'agent est dans les weak, mais que
    #     son record est dans les strong, les poids de l'agents sont ramener en
    #     arrières, et cette méthode est appelée
    #     :return:
    #     """
    #     pass

    def to_register_on_mutation(self)->Dict[str, float]:
        # by default an empty dictionnary
        return dict()


    def to_register_at_period_end(self)->Dict[str,float]:
        #by default an empty dictionnary
        return dict()

    def on_overfitting_restore_me(self)->bool:
        #exemple d'utilisation:
        # l'agent compare son score_train et son score_val. Quand le score_val est trop bas:
        # l'agent augmente son dropout ou sa pénalisation, puis la méthode "on_overfitting_restore_me"
        # renvoie "True", cela demande
        # au familyTrainer de restorer l'agent, en lui mettant la moyenne des poids des anciens records
        pass
    #
    # def return_score(self):
    #     #appeler  pour les tests ou la validation: pas d'optimisation
    #     pass



class Agent_wraper:

    def __init__(self, name,color, agent:Abstract_Agent, max_nb_best,nb_current_scores,current_score_fn):

        self.name=name
        self.color=color
        self.agent=agent
        self.max_nb_best=max_nb_best
        self.current_score_fn=current_score_fn

        self.best_weights = deque(maxlen=max_nb_best)


        self.nb_consecutive_decadence=0

        self.decadent_score_and_weight=None

        self.best_score = None

        self.nb_current_scores=nb_current_scores
        self.current_scores = deque(maxlen=self.nb_current_scores)
        self.best_famparams: Dict[str, float]

        self.long_name=self.name

        self.memory={}
        self.memory_mutations={}

        for k in self.agent.get_famparams().keys():
            self.memory[k]=[]
            self.memory_mutations[k]=[]



        self.memory["score"]=[]
        self.memory["time"]=[]

        self.memory_mutations["score"]=[]
        self.memory_mutations["time"]=[]


    init_memory_was_done=False
    def register_famparams(self,time):



        #un autre enregistrement a lieu au moment des mutations dans "load_from
        for k,v in self.agent.get_famparams().items():
            self.memory[k].append(v)
        self.memory["score"].append(self.compute_current_score())
        self.memory["time"].append(time)


        if not self.init_memory_was_done and hasattr(self.agent,"to_register_at_period_end"):
            for k, v in self.agent.to_register_at_period_end().items():
                self.memory[k]=[]
                self.memory_mutations[k] = []
            self.init_memory_was_done=True


        if hasattr(self.agent,"to_register_at_period_end"):
            for k, v in self.agent.to_register_at_period_end().items():
                self.memory[k].append(v)


    def register_on_mutation(self,time,score,fam_params_before_mutation):
        """ Pour les plots"""
        for k, v in fam_params_before_mutation.items():
            new_v = self.agent.get_famparams()[k]
            self.memory_mutations[k].append((v, new_v))
        self.memory_mutations["score"].append(score)
        self.memory_mutations["time"].append(time)

        if hasattr(self.agent, "to_register_at_period_end"):
            for k, v in self.agent.to_register_at_period_end().items():
                self.memory_mutations[k].append(v)




    #
    # def get_name_suffixed(self,decadence):
    #     res=self.name
    #     for mut in self.mutation_names:
    #         #la double flèche indique le transfert des poids et des famparams
    #         res+="⇇"+mut
    #     return res
    #
    # def perhapds_restablish_decadent(self):
    #     if self.decadent_score_and_weight is not None and self.decadent_score_and_weight[0]>self.best_score:
    #         print(f"Attention, le meilleurs score de l'agent {self.name} est décadent")
    #         self.best_score=self.decadent_score_and_weight[0]
    #         self.best_weights=self.decadent_score_and_weight[1]

    def save_at_best(self,new_best_score:float):
        self.best_weights.append(self.agent.get_copy_of_weights())
        self.best_score=new_best_score
        self.best_famparams=copy.deepcopy(self.agent.get_famparams())


    def mean_of_best_weights(self):
        nb_wei = len(self.best_weights[0])
        nb_best = len(self.best_weights)
        new_wei_list = []
        for j in range(nb_wei):
            ponde = 1 / (nb_best  + 1)
            res = self.best_weights[0][j]*ponde
            sum_ponde=ponde

            for i in range(1,nb_best):
                ponde=1/(nb_best-i+1)
                sum_ponde+=ponde
                res += self.best_weights[i][j]*ponde
            res /= sum_ponde
            new_wei_list.append(res)
        return new_wei_list

    def restore(self):
        new_wei_list = self.mean_of_best_weights()
        self.agent.set_weights(new_wei_list)
        #on ne change pas les famparams


    #exploitation
    def load_from_another(self, better_agent_w: 'Agent_wraper', period_count,time):

        #décadence
        # if decadence:
        #     """ L'agent a jadis optenu le meilleurs score, mais maintenant il est nul.
        #       On enregistre quand même les poids du temps de sa splendeur """
        #     if self.decadent_score_and_weight is None or self.decadent_score_and_weight[0]<self.best_score:
        #         self.decadent_score_and_weight=(self.best_score,self.best_weights)

        """les weights (ou learning variable) d'un réseau sont toujours une liste ou un tuple de tenseur"""
        new_weights = better_agent_w.mean_of_best_weights()

        self.long_name += "⇇" + better_agent_w.name
        print(f"{self.long_name}|", end="")


        self.agent.set_and_perturb_famparams(copy.deepcopy(better_agent_w.best_famparams), period_count)


        self.agent.set_weights(new_weights)

        """
         Ici, c'est pas l'idéal: on attribut comme current_score le best_score du l'other_agent.
         Or le mutant devrait faire ses preuves avant de passer dans la liste des premier.
         Mais on veut éviter qu'il ne soit remuter immédiatement après (il faudrait mettre en place un mécanisme de bonus pour les jeunes mutants)
         """
        self.current_scores=deque(maxlen=self.nb_current_scores)
        self.current_scores.append(better_agent_w.best_score)
        """ne pas oublier de vider la liste des best_weights. Les anciens peuvent être assez différents : le changement de famparams peu induire des changement brutal de poids (ex: coef de pénalisation)"""
        self.best_weights=deque(maxlen=self.max_nb_best)
        self.best_weights.append(new_weights)

        self.register_on_mutation(time,better_agent_w.best_score,better_agent_w.best_famparams)


    def load_from_myself(self, period_count,time):

        new_weights = self.mean_of_best_weights()
        self.long_name += "↫" + self.name
        print(f"{self.long_name}|", end="")


        famparams_before_mustation=copy.deepcopy(self.best_famparams)
        self.agent.set_and_perturb_famparams(copy.deepcopy(self.best_famparams), period_count)

        self.current_scores = deque(maxlen=self.nb_current_scores)
        self.current_scores.append(self.best_score)
        """ne pas oublier de vider la liste des best_weights. Les anciens peuvent être assez différents : le changement de famparams peu induire des changement brutal de poids (ex: coef de pénalisation)"""
        self.best_weights = deque(maxlen=self.max_nb_best)
        self.best_weights.append(new_weights)

        self.register_on_mutation(time,self.best_score,famparams_before_mustation)



    def compute_current_score(self):
        # en par défaut self.current_score_fn c'est la moyenne des 3 dernier scores courants
        return self.current_score_fn(self.current_scores)


def get_a_name(i:int):
    #les premiers noms sont des lettres, ensuite c'est a1,b1,c1,etc
    letters="a b c d e f g h i j k l m n o p q r s t u v w_ A y z A B C D E F G H I J K L M N O P Q R S T U V w_ X Y Z"
    letters=letters.split(" ")
    nb=len(letters)
    if i<nb:
        return letters[i]
    return letters[i%nb]+str(i//nb)

def transform_period_for_each(period_for_each):
    if period_for_each.endswith("s"):
        period_for_each = period_for_each[:-1]
    if period_for_each.endswith("e"):
        period_for_each = period_for_each[:-1]

    if period_for_each.endswith("second"):
        period_unity="second"
        period_duration=float(period_for_each[:-6].rstrip())

    elif period_for_each.endswith("minut"):
        period_unity = "second"
        period_duration = float(period_for_each[:-5].rstrip())*60

    elif period_for_each.endswith("step"):
        period_unity = "step"
        period_duration = float(period_for_each[:-5].rstrip())

    else: raise Exception("perdio must finish by second(e.s) or minut(e.s) or step(s)")

    return period_duration,period_unity



class Family_trainer:
    instance_count=0
    def __init__(self,
                 agents:List[Abstract_Agent],
                 # à la fin de chaque période on fait les mutations
                 period_duration:str, #ex: "10 steps", "20 seconds", "3 minute"
                 #nombre d'agents faible qui sont mutés à la fin de chaque période
                 nb_weak=1,
                 #nombre d'agents fort transmettant leur poids et le famparams
                 nb_strong=1,
                 nb_bestweights_averaged=1,
                 color="k",
                 nb_current_scores=3,
                 current_score_fn=np.mean,#np.max pour garder un agent qui renvoie un nan
                 max_nb_consecutive_decadence=2,
                 name=None,
                 ):

        Family_trainer.instance_count+=1

        self.period_duration,self.period_duration_unity=transform_period_for_each(period_duration)
        self.nb_current_scores=nb_current_scores
        self.current_score_fn=current_score_fn
        self.max_nb_consecutive_decadence=max_nb_consecutive_decadence
        self.name=name if name is not None else "fam_"+str(Family_trainer.instance_count)

        self.nb_weak=nb_weak
        if self.nb_weak>len(agents):
            print(f"Warning: attention, le nombre de strong:{self.nb_weak} est plus grand que le nombre d'agent:{len(agents)}, par conséquent il sera diminué à:{max(len(agents)//2,1)}.")
            self.nb_weak=max(len(agents)//2,1)

        self.nb_strong=nb_strong
        if self.nb_strong>len(agents):
            print(f"Warning: attention, le nombre de strong:{self.nb_strong} est plus grand que le nombre d'agent:{len(agents)}, par conséquent il sera diminué à:{len(agents)}.")
            self.nb_strong=len(agents)

        self.nb_bestweights_averaged=nb_bestweights_averaged

        self.color=color
        self._period_count=-1
        self._agent_pass_count=0

        #un dico, car on voudrait pouvoir supprimer des agents
        self.agents:Dict[str,Agent_wraper] = {}
        self.history=History()

        self.color_list = [color for color in mcolors.TABLEAU_COLORS]#mcolors.CSS4_COLORS

        for i,agent in enumerate(agents):
            agent_name=get_a_name(i)
            agent_color=self.color_list[i%len(self.color_list)]
            agent_w = Agent_wraper(agent_name, agent_color,agent, self.nb_bestweights_averaged,self.nb_current_scores,self.current_score_fn)
            self.agents[agent_name] = agent_w
            # for k, v in agent.get_famparams().items():
            #     self.history.record_metric(k, v)

        self.best_decadent_score=None
        self.best_decadent_weights=None
        self.best_decadent_famparams=None

    def interupt_period(self):
        self.history.stop_local_time()

    def period(self):

        self._period_count += 1
        agents_list=list(self.agents.values())
        if self._period_count==0:
            #tous les agents doivent passer au moins une fois avant de muter.
            print(f"\n{self.name},échauffement ", end="")
            for agent_w in agents_list:
                self.pass_one_agent(agent_w)
        else:
            self.history.start_local_time()
            print(f"\n{self.name},period:{self._period_count} ",end="")
            step_count = 0
            ok = True
            while ok:
                step_count += 1
                if self.period_duration_unity == "second":
                    #on cumule le temps. Avantage: si une pédiode dépasse (à cause d'un step un peu long), la suivante devra être plus courte.
                    #inconvénient: il ne faut pas changer la period_duration au milieu de l'apprentissage
                    #todo changer en utilisant un delta-time plutôt qu'un time absolu (plus robuste pour les modifications futures)
                    ok = self.history.get_local_time() < self.period_duration*self._period_count
                else:
                    ok = step_count < self.period_duration

                #si des agents ne passent pas lors d'une période, ils passeront dans la suivante.
                self.pass_one_agent(agents_list[self._agent_pass_count % len(agents_list)])
                self._agent_pass_count+=1


            #on arrête le localTime, car cela peut être le tour d'un autre familyTrainer
            self.history.stop_local_time()

            # À la fin de chaque periode: mutation
            self.mutation()


    def pass_one_agent(self, agent_w:Agent_wraper):


        score=agent_w.agent.optimize_and_return_score()

        if np.isnan(score): score = -float("inf")

        #self.history.record_metric("score", score)

        agent_w.current_scores.append(score)

        agent_w.register_famparams(self.history.get_local_time())


        if  agent_w.best_score is None or score>agent_w.best_score:
            agent_w.save_at_best(score)
            print(agent_w.name+"↗"+str(np.round(score,4)),end="")
        else:
            print("-",end="")


    def mutation(self):
        """"""

        """
        * les weaks sont les pires agents (selon de leur score courants)
        * les decadents sont les meilleurs agents (selon leur score record), mais qui font également partis des weaks
        * les strong sont les meilleurs agents (selon leur score record) qui ne sont pas décadent.
        """
        agents_best_score_sorted = sorted(self.agents.values(), key=lambda a_w: a_w.best_score)
        agents_current_score_sorted = sorted(self.agents.values(), key=lambda a_w: a_w.compute_current_score())

        weaks=agents_current_score_sorted[:self.nb_weak]
        weak_names={weak.name for weak in weaks}

        decadent_names=[]
        decadents = []
        for i in range(self.nb_strong):
            strong=agents_best_score_sorted[-i]
            if strong.name in weak_names:
                sco=strong.compute_current_score()
                #ex:  sco=-10.  strong.best_score > -10+5
                #ex:  sco=+10.  strong.best_score > +10+5
                #todo quand on a des scores -infty cela crée un warning ici... réfléchir à que faire
                if strong.best_score>sco+np.abs(sco)*0.5:
                    decadent_names.append(strong.name)
                    decadents.append(strong)

        strong_non_decadent=[]
        for strong in agents_best_score_sorted:
            if strong.name not in decadent_names:
                strong_non_decadent.append(strong)
        strongs=strong_non_decadent[-self.nb_strong:]
        if len(strongs)==0:
            print("ATTENTION: il n'y a aucun strong, seulement des décadents. Un agent aléatoire sera désigné comme strong")
            strongs.append(agents_best_score_sorted[np.random.randint(len(agents_best_score_sorted))])
        elif len(strongs)<self.nb_strong:
            print(f"ATTENTION: il y a seulement {len(strongs)} strong non-décadent, ce qui est inférieur au nombre de strong réclamé: {self.nb_strong}")

        print(", mutations:", end="")

        """Les weak prennent l'état d'un strong aléatoire"""
        for weak in weaks:
            strong  = strongs[np.random.randint(len(strongs))]
            weak.load_from_another(strong,self._period_count,self.history.get_local_time())

        """Les décadents sont séparés en 2 parties:
         * les décadents-récurrents  sont ceux qui on été trop de fois décadent de manière consécutives. La limite étant l'attribut  `self.max_nb_consecutive_decadence` 
         * les décadents-non-récurrents sont les autres.
         
         Voilà ce qui leur arrivent:
         * Les décadents-récurrents repartent de l'état d'un strong aléatoire.
           Leur état record est sauvegardé pour la fin, si l'utilisateur demande de regarder le meilleurs état, 
           y compris parmis les états des décadents-récurrent
        * Les décadents-non-récurrent repartent de leur propre état record.
          C'est une bonne chose pour lui, car il aura la possibilité d'évoluer ailleurs grâce aux gradients stochastiques. 
         """
        for decadent in decadents:
            if decadent.nb_consecutive_decadence<self.max_nb_consecutive_decadence:
                decadent.nb_consecutive_decadence += 1
                print(f"\n/!\ L'agent:{decadent.name} est décadent pour la {decadent.nb_consecutive_decadence}-ième fois consécutive; record:{decadent.best_score}, scores courants:{decadent.current_scores}, best_famparams: {decadent.best_famparams} ")

                decadent.load_from_myself( self._period_count,self.history.get_local_time())

            else:
                print(f"\n/!\ L'agent:{decadent.name} est décadent-recurent car il a été `max_nb_consecutive_decadence`={self.max_nb_consecutive_decadence} fois décadents de manière consécutive. Il est évacué de la compétition. Son record:{decadent.best_score}, ses scores courants:{decadent.current_scores}, best_famparams: {decadent.best_famparams} ")
                strong = strongs[np.random.randint(len(strongs))]
                if self.best_decadent_score is None or strong.best_score>self.best_decadent_score:
                    self.best_decadent_score=strong.best_score
                    self.best_decadent_weights=decadent.best_weights
                    self.best_decadent_famparams=decadent.best_famparams

                decadent.nb_consecutive_decadence=0
                decadent.load_from_another(strong,self._period_count,self.history.get_local_time())


    def plot(self,x_key="score",y_keys=None,all_famparams=True):

        print("family_plot")

        agent_list=list(self.agents.values())

        if y_keys is None:
            y_keys=[]

        if all_famparams:
            y_keys+=agent_list[0].agent.get_famparams().keys()

        nb=len(y_keys)
        fig,axs=plt.subplots(nb,1,figsize=(10,2*nb),sharex="all")
        if nb==1:
            axs=[axs]
        axs[-1].set_xlabel(x_key)

        for i,fam_name in enumerate(y_keys):
            axs[i].set_ylabel(fam_name)
            for agent in agent_list:
                axs[i].plot(agent.memory[x_key],agent.memory[fam_name],".",color=agent.color)

            for agent in agent_list:
                for y_val,x_val in zip(agent.memory_mutations[fam_name],agent.memory_mutations[x_key]):
                    if isinstance(x_val,tuple):
                        x_val=x_val[1]
                    if isinstance(y_val,  tuple):
                        axs[i].plot([x_val,x_val],y_val,color=agent.color)
                        axs[i].plot(x_val,y_val[1],"+",color=agent.color)
                    else:
                        axs[i].plot(x_val, y_val, "-.",color=agent.color)
        fig.tight_layout()


    def stats_of_best(self,nb_best=None):
        if nb_best is None:
            nb_best = int(len(self.agents) * 0.5)
        assert nb_best<=len(self.agents)

        agent_w_sorted=sorted(self.agents.values(),key=lambda ag:ag.best_score)
        best:List[Agent_wraper]=agent_w_sorted[-nb_best-1:]

        res=dict()
        sum_score=0
        for agent in best:
            sum_score+=agent.best_score
            for k,v in agent.best_famparams.items():
                res[k]=res.get(k,0)+v*agent.best_score

        for k,v in res.items():
            res[k]/=sum_score

        return res


    def get_best_agent(self, mean_its_weights=False, including_decadent=True):

        liste=sorted(self.agents.values(),key=lambda agent:agent.best_score)
        best=liste[-1]

        decadent_loaded=False
        if including_decadent:
            if self.best_decadent_score is not None and best.best_score<self.best_decadent_score:
                print("ATTENTION: le meilleurs agent est un agent décadent. Si voulez un non-décadent, utiliser including_decadent=False")
                #on récupère le pire agent pour lui transferer les poids du décadent
                an_agent=liste[0]
                an_agent.agent.set_weights(self.best_decadent_weights)
                an_agent.best_famparams=self.best_decadent_famparams
                best=an_agent
                decadent_loaded=True

        if mean_its_weights and not decadent_loaded:
            weights=best.mean_of_best_weights()
        else:
            weights=best.best_weights[-1]
        best.agent.set_weights(weights)

        return best.agent



############################################## TEST ##############################################


class Agent_ultra_toy(Abstract_Agent):

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
    family_trainer = Family_trainer(agents, period_duration="10 steps", nb_strong=1)

    for _ in range(20):
        family_trainer.period()

    #par défaut: x=score, y=les famparams
    family_trainer.plot()
    # x=time, y=le score uniquement
    family_trainer.plot("time",["score"],all_famparams=False)
    # x=score, y= wei0,wei1 et les famparams
    family_trainer.plot("score",["wei0","wei1"])
    # etc
    family_trainer.plot("wei0",["wei0","wei1"],all_famparams=False)
    family_trainer.plot("add0",["score"],all_famparams=False)

    plt.show()

if __name__=="__main__":
    test()



