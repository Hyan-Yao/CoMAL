import math
import os
import numpy as np
from openai import OpenAI
import textwrap
import time
import re
from pprint import pprint
from flow.controllers.base_controller import BaseController
from flow.controllers.PID import IncrementalPID


os.environ["OPENAI_API_KEY"] = ""
Flow_API = None

MapDescription = {
    "ring" : """The ring road network consists of a circular lane with a specified length.""",
    "merge" : textwrap.dedent(
        """The merge map consists of two straight single lanes. One is the main highway where all vehicles drive,
        and another is the merge branch to intersect with main highy way. The perturbations caused by the vehicles entering
        through the ramp leads to the formation of congested waves downstream in the main highway."""),
    "figureeight" : textwrap.dedent("""
        The shape of the map is like a figure eight:
        The figure eight network is an extension of the ring road network: Two rings, placed at opposite ends of the network, are connected by an intersection with road segments of length equal to the diameter of the rings.
        If two vehicles attempt to cross the intersection from opposing directions, the dynamics of these vehicles are constrained by right-of-way rules provided by SUMO.
        """),
}

# How to elimite waves
# if there is Relative Traffic Congestion:
#    set target speed v0 to a little higher(1 m/s) than the lead.
#    update target speed after the speed approched.
# else:
#    speed up to tailgate the lead vehicle.
Memory = {
    "figureeight" : ["The best stragety is to help all vehicles form a queque to pass the intersection quickly, which avoids collisions there. Thus, driver agents can divide into a leader and some followers to help with a queue.",
        "Below are best IDM planners for different roles: Leader: SlowDown {'a' : 0.2, 'v' : 6.0, 's' : 2.0}, FollowUp {'a' : 1.0, 'v' : 30.0, 's' : 0.6}",
        "While reasoning your role, You can say not decided yet and wait for next round. There is no rush to give a final decision answer unless you are confident about it after fully communicating with other vehicles. "],
    "ring": ["If the difference between self headway and lead headway is relatively large (self headway > 1.5 * lead headway), you're in traffic jam. Otherwise you're not in traffic jam.",
        "If you're in a traffic jam, your role should be Approacher. If not, your role should be Follower",
        "If you're in traffic jam, the best IDM planner for you is Approch {'a' : 1.0, 'v' : lead velocity + 1.0, 's' : 2.0}",
        "If you're not in traffic jam, the best IDM planner for you is FollowUp {'a' : 5.0, 'v' : 150.0, 's' : 0.0}",
        "While reasoning your role, it had better to reason your role directly in the round."],
    "merge": ["If the difference between self headway and lead headway is relatively large (self headway > 1.5 * lead headway), you're in traffic jam. Otherwise you're not in traffic jam.",
        "If you're in a traffic jam, your role should be Approacher. If not, your role should be Follower",
        "If you're in traffic jam, the best IDM planner for you is Approch {'a' : 1.0, 'v' : lead velocity + 1.0, 's' : 2.0}",
        "If you're not in traffic jam, the best IDM planner for you is FollowUp {'a' : 5.0, 'v' : 150.0, 's' : 0.0}",
        "While reasoning your role, it had better to reason your role directly in the round."],
}

class DriverAgent():
    def __init__(self, veh_id):
        self.client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY")
        )
        self.llm_model = "gpt-4o-mini"
        self.veh_id = veh_id

        self.role = None

    def call(self):
        human_message = textwrap.dedent(f"""
        Role
        You are the brain of an autonomous vehicle in the road. Your vehicle ID is {self.veh_id}. You can connect all the autonomous vehicles in the scenario. Please make the decision to optimize the average velocity of all the vehicles. Try your best to avoid collisions with other objects.

        Context
        - Coordinates: X-axis is oriented along the road between 0 and 1.

        Inputs
        1. Map: {self.map_description}
        2. Perception: the accurate description of the velocity and position of each vehicle.
        {self.perception}
        2. Shared Messages: The message to share in the public channel from the other agents. You should consider these messages in your decision.
        {self.shared_message}

        Memory
        {self.memory_retrieval}
        
        Task & Output
        {self.task}
        
        Make the decision as best you can. Begin!
        """)

        chat_completion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": human_message,
                }
            ],
            model=self.llm_model,
        )
        response = chat_completion.choices[0].message.content
        return response

    def collaborate(self, map_name, perception, shared_message):
        self.perception = perception
        self.shared_message = shared_message
        self.map_description = MapDescription[map_name]
        self.memory_retrieval = Memory[map_name]
        self.task = textwrap.dedent("""
        You should discuss with other vehicles in the road and figure out your role to follow the human instruction. You can recieve the messages from other vehicles in the public channel and send your message to it. Each vehicle will speak one by one until its role is decided.
        - Your one message to the public channel that other vehicles can read.
        - Your role decision: a word or brief phrase to describe your role and task.
        Ouput is only a Dictonary format like: {"message" : "your message here to public", "role" : "your role decision"}
        """)
                
        response = self.call()
        return response

    def execute(self, map_name, perception):
        self.perception = perception
        self.shared_message = ""
        self.map_description = MapDescription[map_name]
        self.memory_retrieval = Memory[map_name]
        self.task = textwrap.dedent(f'''
        You have confirmed your role in collaboration: {self.role},
        The memory of this scenario is: {self.memory_retrieval},
        ''')
        self.task += """You should choose an IDM planner from memory based on your specific role in collaboration.
        Output format is only a dictionary, like: {'a' : 0.2, 'v': 6.0, 's' : 2.0}
        YOUR OUTPUT SHOULD BE ONLY INCLUE A DICTIONARY.
        """

        response = self.call()
        return response


class LLMController(BaseController):
    def __init__(self, veh_id, map='figureeight', v0=30, T=1, a=1, b=1.5, delta=4, s0=2, time_delay=0.0, noise=0, fail_safe=None, display_warnings=True, car_following_params=None):
        BaseController.__init__(self, veh_id, car_following_params, delay=time_delay, fail_safe=fail_safe, noise=noise, display_warnings=display_warnings)
        self.v0 = v0
        self.T = T
        self.a = a
        self.b = b
        self.delta = delta
        self.s0 = s0

        self.DA = DriverAgent(veh_id)
        self.map_name = map

        self.cool_down_time = 4 # ring 10, merge 4
        if self.map_name == 'ring':
            self.cool_down_time = 10
        self.cd_counter = 0
        self.vel_track = []

    def get_accel(self, env):
        '''Perception'''
        p = env.k.vehicle.get_position(self.veh_id)
        v = env.k.vehicle.get_speed(self.veh_id)
        lead_id = env.k.vehicle.get_leader(self.veh_id)
        h = env.k.vehicle.get_headway(self.veh_id)
        current_time = env.time_counter
        
        lead_v = [0 for i in range(5)]
        lead_h = [0 for i in range(5)]
        for i in range(5):
            lead_v[i] = env.k.vehicle.get_speed(lead_id)
            lead_h[i] = env.k.vehicle.get_headway(lead_id)
            lead_id = env.k.vehicle.get_leader(lead_id)
        self.lead_v = abs(np.mean(lead_v))
        self.lead_h = abs(np.mean(lead_h))

        self.vel_track.append(v)
        if current_time == 1500:
            print("vel std: ", np.std(self.vel_track))
            print("vel avg: ", np.mean(self.vel_track))

        '''LLM Reasoning'''
        # LLM Invoke: ring - first time, ring/merge: is_traffic_jam (>1.0s)
        if self.map_name == "figureeight":
            t = env.time_counter
            collaboration_round = 2
            if t in range(collaboration_round + 1): # collaboration round
                self.llm_collaborate(env)
            if t == collaboration_round:
                self.llm_reason(env) # execution mmodule

        elif self.map_name == "ring" or self.map_name == "merge":
            # is_traffic_jam = (h > self.lead_h * 1.5)  # for any car
            # if is_traffic_jam:
            if current_time - self.cd_counter > self.cool_down_time: # llm call per 1s/2s
                self.cd_counter = current_time

                self.llm_collaborate(env)
                self.llm_reason(env) # execution module
            

        '''IDM planner'''
        if abs(h) < 1e-3:
            h = 1e-3

        if lead_id is None or lead_id == '':  # no car ahead
            s_star = 0
        else:
            lead_vel = env.k.vehicle.get_speed(lead_id)
            s_star = self.s0 + max(0, v * self.T + v * (v - lead_vel) / (2 * np.sqrt(self.a * self.b)))

        acc = self.a * (1 - (v / self.v0)**self.delta - (s_star / h)**2)
        return acc


    def llm_collaborate(self, env):
        scenario_description = self.get_perception(env)
        shared_message = env.message_pool.get_all_msg()
        while True:
            try:
                response = self.DA.collaborate(self.map_name, scenario_description, shared_message)
                self.DA.role = eval(response)['role']
                break
            except SyntaxError:
                print("----SyntaxError: generate again")
        env.message_pool.join(self.veh_id, response)
        print(response)
    
    def llm_reason(self, env):
        scenario_description = self.get_perception(env)
        while True:
            try:
                response = self.DA.execute(self.map_name, scenario_description)
                paras = eval(response)
                self.a, self.v0, self.s0 = paras['a'], paras['v'], paras['s']
                break
            except SyntaxError:
                print("----SyntaxError: generate again")
        print(response)

    def get_perception(self, env):
        if self.map_name == "figueeight":
            state = env.get_state()
            speed, pos = state['speed'], state['pos']
            scenario_description = textwrap.dedent(f"""\
            Your speed is {round(env.k.vehicle.get_speed(self.veh_id), 2)} m/s, and lane position is {round(env.k.vehicle.get_position(self.veh_id), 2)} m. 
            There are other vehicles driving around you, and below is their basic information:
            """)
            for i in range(len(speed)):
                scenario_description += f" - Vehicle {i} is driving on the same lane as you. The speed of it is {round(speed[i], 2)} m/s, and lane position is {round(pos[i], 2)} m.\n"
        
        elif self.map_name == "ring" or self.map_name == "merge":
            scenario_description = textwrap.dedent(f"""\
            Your speed is {round(env.k.vehicle.get_speed(self.veh_id), 2)} m/s, and self headway is {round(env.k.vehicle.get_headway(self.veh_id), 2)} m. 
            The lead vehicles drive at speed of it is {round(self.lead_v, 2)} m/s, and lead headway is {round(self.lead_h, 2)} m.\n
            """)

        
        return scenario_description
        
