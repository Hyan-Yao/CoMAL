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

os.environ["http_proxy"] = "http://localhost:7890"
os.environ["https_proxy"] = "http://localhost:7890"
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
    def __init__(self, veh_id, v0=30, T=1, a=1, b=1.5, delta=4, s0=2, time_delay=0.0, noise=0, fail_safe=None, display_warnings=True, car_following_params=None):
        BaseController.__init__(self, veh_id, car_following_params, delay=time_delay, fail_safe=fail_safe, noise=noise, display_warnings=display_warnings)
        self.v0 = v0
        self.T = T
        self.a = a
        self.b = b
        self.delta = delta
        self.s0 = s0

        self.DA = DriverAgent(veh_id)
        self.map_name = 'merge'

        self.cool_down_time = 4 # ring 10, merge 4
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


class MergeController(BaseController):
    """
    Attributes
    ----------
    v0 : float
        desirable velocity, in m/s (default: 30)
    T : float
        safe time headway, in s (default: 1)
    a : float
        max acceleration, in m/s2 (default: 1)
    b : float
        comfortable deceleration, in m/s2 (default: 1.5)
    delta : float
        acceleration exponent (default: 4)
    s0 : float
        linear jam distance, in m (default: 2)
    """

    def __init__(self,
                 veh_id,
                 v0=30,
                 T=1,
                 a=1,
                 b=1.5,
                 delta=4,
                 s0=2,
                 time_delay=0.0,
                 noise=0,
                 fail_safe=None,
                 display_warnings=True,
                 car_following_params=None):
        """Instantiate an IDM controller."""
        BaseController.__init__(
            self,
            veh_id,
            car_following_params,
            delay=time_delay,
            fail_safe=fail_safe,
            noise=noise,
            display_warnings=display_warnings,
        )
        self.v0 = v0
        self.T = T
        self.a = a
        self.b = b
        self.delta = delta
        self.s0 = s0
        
        self.v0_cache = self.v0
        self.a_cache = self.a
        self.s0_cache = self.s0

        self.cool_down_time = 4
        self.cd_counter = 0

    def get_accel(self, env):
        """Basic Perception"""
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
        lead_v = np.mean(lead_v)
        lead_h = np.mean(lead_h)

        '''
        if self.veh_id[:11] == "human_merge":
            env.message_pool.state['merge'] = 1
            env.message_pool.state['merge_pos'] = env.k.vehicle.get_position(self.veh_id)
            env.message_pool.state['merge_t'] = current_time
        elif current_time > env.message_pool.state['merge_t'] + 1:
            env.message_pool.state['merge'] = 0

            self.a = 1.0 * 5.0
            self.v0 = 30 * 5.0
        
        if env.message_pool.state['merge'] == 1:
            if p > 400:
                self.a = 1.0 * 0.2
                self.v0 = 30 * 0.2
            # if p in interval(merge_pos), slow down
            # print(self.veh_id, p, env.message_pool.state['merge_pos'],  v)
        '''
        
        # Strategy For Merge
        if current_time > 50:

            if h > lead_h * 1.5:
                if current_time - self.cd_counter > self.cool_down_time: # llm per 1s 
                    self.cd_counter = current_time

                    if abs(v - lead_v) < 0.5:
                        self.s0 = self.s0_cache * 1.0
                        self.a = self.a_cache * 1.0
                        self.v0_cache = lead_v + 1.0
                        print("reset: ", self.veh_id)

                self.v0 = self.v0_cache
            
            else:
                self.s0 = self.s0_cache * 1.0
                self.a = self.a_cache * 5.0
                self.v0 = self.v0_cache * 5.0

        


        """Compute Accelaration"""
        # in order to deal with ZeroDivisionError
        if abs(h) < 1e-3:
            h = 1e-3

        if lead_id is None or lead_id == '':  # no car ahead
            s_star = 0
        else:
            lead_vel = env.k.vehicle.get_speed(lead_id)
            s_star = self.s0 + max(
                0, v * self.T + v * (v - lead_vel) /
                (2 * np.sqrt(self.a * self.b)))

        acc = self.a * (1 - (v / self.v0)**self.delta - (s_star / h)**2)
        # print(self.veh_id, p, v, acc)
        return acc


class RingController(BaseController):
    """
    Attributes
    ----------
    v0 : float
        desirable velocity, in m/s (default: 30)
    T : float
        safe time headway, in s (default: 1)
    a : float
        max acceleration, in m/s2 (default: 1)
    b : float
        comfortable deceleration, in m/s2 (default: 1.5)
    delta : float
        acceleration exponent (default: 4)
    s0 : float
        linear jam distance, in m (default: 2)
    """

    def __init__(self,
                 veh_id,
                 v0=30,
                 T=1,
                 a=1,
                 b=1.5,
                 delta=4,
                 s0=2,
                 time_delay=0.0,
                 noise=0,
                 fail_safe=None,
                 display_warnings=True,
                 car_following_params=None):
        """Instantiate an IDM controller."""
        BaseController.__init__(
            self,
            veh_id,
            car_following_params,
            delay=time_delay,
            fail_safe=fail_safe,
            noise=noise,
            display_warnings=display_warnings,
        )
        self.v0 = v0
        self.T = T
        self.a = a
        self.b = b
        self.delta = delta
        self.s0 = s0
        
        self.v0_cache = 2.0
        self.a_cache = self.a
        self.s0_cache = self.s0

        self.cool_down_time = 10
        self.cd_counter = 0

        self.vel_track = []

    def get_accel(self, env):
        """Basic Perception"""
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
        lead_v = np.mean(lead_v)
        lead_h = np.mean(lead_h)

        
        # print(current_time, v, self.v0, h, lead_v)
        # print(self.veh_id, current_time, v)
        self.vel_track.append(v)

        if current_time == 1500:
            print("vel std: ", np.std(self.vel_track))
            print("vel avg: ", np.mean(self.vel_track))
            # print(self.vel_track)

        """Strategy For Ring"""
        # print(h, lead_h)
        '''
        if h > lead_h * 1.5:
            if current_time % 10 < 5:
            #if h > lead_h * 3:
                self.v0 = 150
                self.a = 5
                self.s0 = 0
            else:
                self.v0 = 2
                self.a = 0.5
        else:
            self.v0 = 150
            self.a = 5
            self.s0 = 0
        '''
        if h > lead_h * 1.5:
            if current_time - self.cd_counter > self.cool_down_time: # llm per 1s 
                self.cd_counter = current_time

                print("raise", current_time)

                self.s0 = self.s0_cache * 1.0
                self.a = self.a_cache * 1.0
                self.v0 = lead_v + 1.0

        
        else:
            self.s0 = self.s0_cache * 1.0
            self.a = self.a_cache * 5.0
            self.v0 = self.v0_cache * 5.0

        """Compute Accelaration"""
        # in order to deal with ZeroDivisionError
        if abs(h) < 1e-3:
            h = 1e-3

        if lead_id is None or lead_id == '':  # no car ahead
            s_star = 0
        else:
            lead_vel = env.k.vehicle.get_speed(lead_id)
            s_star = self.s0 + max(
                0, v * self.T + v * (v - lead_vel) /
                (2 * np.sqrt(self.a * self.b)))

        acc = self.a * (1 - (v / self.v0)**self.delta - (s_star / h)**2)
        return acc



class FigureeightController(BaseController):
    def __init__(self, veh_id, api_key=None, v0=30, T=1, a=1, b=1.5, delta=4, s0=2, time_delay=0.0, noise=0, fail_safe=None, display_warnings=True, car_following_params=None):
        BaseController.__init__(self, veh_id, car_following_params, delay=time_delay, fail_safe=fail_safe, noise=noise, display_warnings=display_warnings)
        self.v0 = v0
        self.T = T
        self.a = a
        self.b = b
        self.delta = delta
        self.s0 = s0
        # self.DA = DriverAgent(veh_id, api_key=api_key)

        # store the potential variables to cache
        self.v0_cache = self.v0
        self.a_cache = self.a
        self.s0_cache = self.s0

        self.start_pid1 = False
        self.start_pid2 = False
        self.start_inter = False
        self.pid1 = IncrementalPID(0.01, 0.04, 0.01, 0.50, 0.20)
        self.pid2 = IncrementalPID(0.01, 0.00, 0.00, 0.50, 0.20)

        self.track_v = []
        self.track_p = []
        self.track_f = []


    def custom_adapt(self, env):
        # 1. Basic Knowledge: LLM as the direct controller makes no sense.
        #    It can't understand the number at all. We should design some policy for it in advance.
        #    Experiment result: LLM outputs acc directly? (Guess will be mad)
        # 2. LLM makes the high-level decision
        # (1) It is too complex to design the function a=f(p,v)
        #     It's not good to change acc directly (that is, reflect the result of IDM)
        #     Experiment result: 5.7-5.8
        # (2) IDM can help with a(v, s), given the expected [v_expected, a_max, s_min]
        #     We should ask LLM to control the paras of IDM
        #     IDM can give a guarantee a lower limit ~ 6.3
        # Compared Exp:
        # a. IDM[v0, a, s0] + manual rules <=> LLM
        # - IDM: a <- IDM(v, s)
        # - maunal rules: IDM <- p
        current_time = env.time_counter

        state = env.get_state()
        speed, pos = state['speed'], state['pos']
        self_pos, self_vel = pos[13], speed[13]
        tail_pos, tail_vel = pos[0], speed[0]
        headway = (1.0 + tail_pos - self_pos) % 1.0

        # Manual Rules
        # 0. s0 non-relevant, give a, v0 the same factor
        # 1. simple const factor: 6.30
        
        # 0.20 is the best for 0, 0.18 is for 2
        '''
        self.v0 = self.v0_cache * 0.20 # v0+a => queque
        self.a = self.a_cache * 0.20
        '''
        # self.s0 = self.s0_cache # non-sensitive

        # 2. spatial-dependent factor: 6.40
        # 0.22/0.20 is the best for 0, - is for 2
        '''
        if 0 < self_pos < 0.15 or 0.5 < self_pos < 0.65:
            self.v0 = self.v0_cache * 0.22
            self.a = self.a_cache * 0.22
            # self.s0 = self.s0_cache
        else:
            self.v0 = self.v0_cache * 0.20
            self.a = self.a_cache * 0.20
            # self.s0 = self.s0_cache
        '''

        # 3. time-dependent linear factor: 6.65
        if current_time < 500:
            self.v0 = self.v0_cache * 0.20 # v0+a => queque
            self.a = self.a_cache * 0.20 # 0.20 is the best as the const factor
            self.s0 = self.s0_cache # non-sensitive
        else:
            factor = 0.20 + 0.010 * ((current_time - 500) / 200)
            self.v0 = self.v0_cache * factor # v0+a => queque
            self.a = self.a_cache * factor
            self.s0 = self.s0_cache # non-sensitive

        # 4. track the tail veh: 6.92
        '''
        if current_time < 500:
            factor = 0.20
        else:
            # factor = (self_dis/v0 - tail_remain/v0_cache) * k + b
            factor = 0.20
            eps = 0.0001
            if 0.3 < self_pos < 0.6:
                self.start_pid2 = False
                self.start_inter = False
                if not self.start_pid1:
                    self.start_pid1 = True

                    #self.track_v.append(env.k.vehicle.get_speed(self.veh_id))
                    #self.track_p.append(1.2)

                self_dis = 0.6 - self_pos
                tail_remain = (0.1 + 0.5) - (tail_pos + 0.5) % 1.0
                distant_factor = self_dis / (self_vel + eps) - tail_remain / (tail_vel + eps)
                factor += distant_factor * 0.10

            elif 0 < self_pos < 0.1 or self_pos > 0.8:
                self.start_pid1 = False
                self.start_inter = False
                if not self.start_pid2:
                    self.start_pid2 = True

                    #self.track_v.append(env.k.vehicle.get_speed(self.veh_id))
                    #self.track_p.append(1.2)

                self_dis = (0.1 + 0.5) - (self_pos + 0.5) % 1.0
                tail_remain = 0.5 - tail_pos
                distant_factor = self_dis / (self_vel + eps) - tail_remain / (tail_vel + eps)
                factor += distant_factor * 0.10

            else:
                self.start_pid1 = False
                self.start_pid2 = False
                if not self.start_inter:
                    self.start_inter = True

                    #self.track_v.append(env.k.vehicle.get_speed(self.veh_id))
                    #self.track_p.append(1.2)

                factor += 0.02 * ((current_time - 500) / 200)
                
            if factor < 0:
                factor = 0.01

        self.v0 = self.v0_cache * factor # v0+a => queque
        self.a = self.a_cache * factor
        '''

        # 5. PID for factor -> DT
        '''
        if current_time < 500:
            factor = 0.20
        else:
            # factor = (self_dis/v0 - tail_remain/v0_cache) * k + b
            factor = 0.20
            eps = 0.0001
            if 0.3 <= self_pos < 0.6:
                self.start_pid2 = False
                self.start_inter = False
                if not self.start_pid1:
                    self.pid1.reset(0.50, 0.20)
                    self.start_pid1 = True

                    self.track_v.append(env.k.vehicle.get_speed(self.veh_id))
                    self.track_p.append(1.2)

                self_dis = 0.6 - self_pos
                tail_remain = (0.1 + 0.5) - (tail_pos + 0.5) % 1.0
                DT = self_dis / (self_vel + eps) - tail_remain / (tail_vel + eps)
                factor = self.pid1.SetStepSignal(DT)
                # factor += DT * 0.10

            elif 0.8 <= self_pos or 0 <= self_pos < 0.1:
                self.start_pid1 = False
                self.start_inter = False
                if not self.start_pid2:
                    self.pid2.reset(0.50, 0.20)
                    self.start_pid2 = True

                    self.track_v.append(env.k.vehicle.get_speed(self.veh_id))
                    self.track_p.append(1.2)

                self_dis = (0.1 + 0.5) - (self_pos + 0.5) % 1.0
                tail_remain = 0.5 - tail_pos
                DT = self_dis / (self_vel + eps) - tail_remain / (tail_vel + eps)
                factor = self.pid2.SetStepSignal(DT)
                # factor += DT * 0.10

            else:
                self.start_pid1 = False
                self.start_pid2 = False
                if not self.start_inter:
                    self.start_inter = True

                    self.track_v.append(env.k.vehicle.get_speed(self.veh_id))
                    self.track_p.append(1.2)

                # factor += 0.10
                factor += 0.010 * ((current_time - 500) / 200)
                

            if factor < 0:
                factor = 0.01

        self.v0 = self.v0_cache * factor # v0+a => queque
        self.a = self.a_cache * factor
        '''

        # self.track_f.append(factor)


    # @measure_time
    def get_accel(self, env):
        self.custom_adapt(env)

        v = env.k.vehicle.get_speed(self.veh_id)
        lead_id = env.k.vehicle.get_leader(self.veh_id)
        h = env.k.vehicle.get_headway(self.veh_id)

        if abs(h) < 1e-3:
            h = 1e-3

        if lead_id is None or lead_id == '':  # no car ahead
            s_star = 0
        else:
            lead_vel = env.k.vehicle.get_speed(lead_id)
            s_star = self.s0 + max(0, v * self.T + v * (v - lead_vel) / (2 * np.sqrt(self.a * self.b)))

        IDM_acc = self.a * (1 - (v / self.v0)**self.delta - (s_star / h)**2)
        
        p = env.k.vehicle.get_x_by_id(self.veh_id) / env.k.network.length()
        
        # self.track_v.append(v)
        # self.track_p.append(p)
        # track_data = np.array([self.track_v, self.track_p, self.track_f])
        #np.save("/home/yao/flow/log/track_data_figureeight0.npy", track_data)

        return IDM_acc
        
