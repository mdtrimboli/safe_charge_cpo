from functional import seq
import numpy as np
import torch
import copy


from safe_explorer.core.config import Config
from safe_explorer.env.ballnd import BallND
from safe_explorer.env.spaceship import Spaceship
from safe_explorer.env.battery import Battery
from safe_explorer.ddpg.actor import Actor
from safe_explorer.ddpg.critic import Critic
from safe_explorer.ddpg.ddpg import DDPG
from safe_explorer.safety_layer.safety_layer import SafetyLayer


class Trainer:
    def __init__(self):
        self._config = Config.get().main.trainer
        self._set_seeds()

    def _set_seeds(self):
        # Desactivar en caso de total aleatoriedad
        a = 0
        #torch.manual_seed(self._config.seed)        # Para clase torch
        #np.random.seed(self._config.seed)           # Para clase ndarray

    def _print_ascii_art(self):
        print(
        """
          _________       _____        ___________              .__                              
         /   _____/____ _/ ____\____   \_   _____/__  _________ |  |   ___________   ___________ 
         \_____  \\__  \\   __\/ __ \   |    __)_\  \/  /\____ \|  |  /  _ \_  __ \_/ __ \_  __ \\
         /        \/ __ \|  | \  ___/   |        \>    < |  |_> >  |_(  <_> )  | \/\  ___/|  | \/
        /_______  (____  /__|  \___  > /_______  /__/\_ \|   __/|____/\____/|__|    \___  >__|   
                \/     \/          \/          \/      \/|__|                           \/    
        """)                                                                                                                  

    def train(self):
        self._print_ascii_art()
        print("============================================================")
        print("Initialized SafeExplorer with config:")
        print("------------------------------------------------------------")
        Config.get().pprint()
        print("============================================================")

        env = BallND() if self._config.task == "ballnd" else Battery()

        switch = True


        SAVE = switch               # Almacenamiento de los pesos
        LOAD = not(switch)          # Carga de los pesos vs Entrenamiento

        safety_layer = None #Prueba

        if self._config.use_safety_layer:
            safety_layer = SafetyLayer(env)
            # load or train
            if LOAD:
                #safety_layer._models[0].load_state_dict = torch.load('model/safety_1_weights.pth')
                #safety_layer._models[1].load_state_dict = torch.load('model/safety_2_weights.pth')

                safety_layer._models[0].load_state_dict(torch.load('model/safety_1_weights.pth'))
                safety_layer._models[1].load_state_dict(torch.load('model/safety_2_weights.pth'))

            else:
                safety_layer.train()

            if SAVE:
                #safety_layer._models
                #best_sl_model0 = copy.deepcopy(safety_layer._models[0].state_dict())
                #best_sl_model1 = copy.deepcopy(safety_layer._models[1].state_dict())

                torch.save(safety_layer._models[0].state_dict(), 'model/safety_1_weights.pth')
                torch.save(safety_layer._models[1].state_dict(), 'model/safety_2_weights.pth')

        
        observation_dim = (seq(env.observation_space.spaces.values())
                            .map(lambda x: x.shape[0])
                            .sum())

        actor = Actor(observation_dim, env.action_space.shape[0])
        critic = Critic(observation_dim, env.action_space.shape[0])


        safe_action_func = safety_layer.get_safe_action if safety_layer else None
        ddpg = DDPG(env, actor, critic, safe_action_func)

        if LOAD:
            ddpg._actor.load_state_dict(torch.load('model/actor_weights.pth'))
            ddpg._critic.load_state_dict(torch.load('model/critic_weights.pth'))
        else:
            ddpg.train()

        # PATH = 'model/model_weights.pth'
        if SAVE:
            torch.save(ddpg._actor.state_dict(), 'model/actor_weights.pth')
            torch.save(ddpg._critic.state_dict(), 'model/critic_weights.pth')

        ddpg.evaluate()

        """
        np.savetxt("curves/Rew_DDPG_RS.csv", ddpg.episodic_reward_buffer, delimiter=", ", fmt='% s')
        np.savetxt("curves/Len_DDPG_RS.csv", ddpg.episodic_length_buffer, delimiter=", ", fmt='% s')
        np.savetxt("curves/AVConst_DDPG_RS.csv", ddpg.accum_constraint_violations, delimiter=", ", fmt='% s')
        """


        if LOAD:
            np.savetxt("curves/T_DDPG_RS.csv", ddpg.temp, delimiter=", ", fmt='% s')
            np.savetxt("curves/V_DDPG_RS.csv", ddpg.volt, delimiter=", ", fmt='% s')
            np.savetxt("curves/I_DDPG_RS.csv", ddpg.curr, delimiter=", ", fmt='% s')
            np.savetxt("curves/SOC_DDPG_RS.csv", ddpg.soc, delimiter=", ", fmt='% s')
            np.savetxt("curves/SOH_DDPG_RS.csv", ddpg.soh, delimiter=", ", fmt='% s')





if __name__ == '__main__':
    Trainer().train()
