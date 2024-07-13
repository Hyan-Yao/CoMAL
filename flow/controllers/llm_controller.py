import math
import os
import numpy as np
from openai import OpenAI
import textwrap
import time
from pprint import pprint
from flow.controllers.base_controller import BaseController

os.environ["http_proxy"] = "http://localhost:7890"
os.environ["https_proxy"] = "http://localhost:7890"
os.environ["OPENAI_API_KEY"] = ""

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
delimiter = "###########"

expname = "figure_eight_0" # the name of each exp
logdir = f"log/{expname}"
os.makedirs(logdir, exist_ok=True)

# Logging Functions
def log_scenario_description(scenario_description):
    with open(f"{logdir}/scenario_log.txt", "a") as log_file:
        log_file.write(scenario_description + "\n")

def log_agent_reasoning(veh_id, reasoning, action_id, message):
    with open(f"{logdir}/agent_{veh_id}_log.txt", "a") as log_file:
        log_file.write(f"Reasoning: {reasoning}\n")
        log_file.write(f"Action ID: {action_id}, Message: {message}\n")

def log_communication(veh_id, message):
    with open(f"{logdir}/communication_{veh_id}_log.txt", "a") as log_file:
        log_file.write(f"Message: {message}\n")

def log_decision_theory(veh_id, decision_logic):
    with open(f"{logdir}/decision_theory_{veh_id}_log.txt", "a") as log_file:
        log_file.write(f"Decision Logic: {decision_logic}\n")

def measure_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        # print(f"Execution time: {execution_time} seconds")
        return result
    return wrapper

def calculate_reward(env):
    reward = 0
    # Reward for maintaining safe distance and smooth traffic
    for veh_id in env.k.vehicle.get_ids():
        speed = env.k.vehicle.get_speed(veh_id)
        headway = env.k.vehicle.get_headway(veh_id)
        reward += speed - abs(headway) * 0.1

    # Penalty for collisions
    collisions = env.k.simulation.get_collision_number()
    reward -= collisions * 10
    return reward


class DriverAgent():
    def __init__(self, veh_id):
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.veh_id = veh_id

    def make_decision(self, scenario_description, env):
        system_prompt = textwrap.dedent("""
        You are now act as a mature driving assistant, who can give accurate and correct advice for human driver in complex urban driving scenarios.
        Please give a suitable advice based on the scenario description and shared message below.
        ######
        Your available actions:
        Maintain - remain in the current lane with current acceleration - Action_id: 0
        Speed up - Increase acceleration of the vehicle - Action_id: 1
        Slow down - Reduce acceleration of the vehicle - Action_id: 2
        ######
        Please output an action id and a message you want to tell other vehicles. Let's think step by step. You can give your reasoning progress.
        Output Format: only give a dictionary as below.
        {'action': 0, 'message': 'I suggest veh 0,1 slow down. Others please maintain the current speed.'}
        """)

        shared_message = env.message_pool.get_msg()

        human_message = textwrap.dedent(f"""
        {system_prompt}\n
        {delimiter} scenario_description:\n{scenario_description}\n
        {delimiter} shared message:\n{shared_message}\n
        """)

        chat_completion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": human_message,
                }
            ],
            model="gpt-3.5-turbo",
        )

        response = chat_completion.choices[0].message.content
        try:
            response = eval(response)
        except:
            # print("Output Format Error: ", response)
            response = {'action': 0, 'message': ' '}
        
        action_id = response['action']
        message = response['message']
        env.message_pool.join(message, self.veh_id)

        # Loging
        #log_scenario_description(scenario_description)
        #log_agent_reasoning(self.veh_id, human_message, action_id, message)
        # print(f"\n\n------------{self.veh_id}--------------\n\n")
        # print(human_message)
        print(f"\n{delimiter}\naction_id: {action_id}\nmessage:{message}")

        return action_id

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

    @measure_time
    def get_accel(self, env):
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
        
        llm_acc = self.llm_decision(IDM_acc, env)
        return llm_acc

    def llm_decision(self, IDM_acc, env):
        state = env.get_state()
        speed, pos = state['speed'], state['pos']

        scenario_description = textwrap.dedent(f"""\
        You are driving on a road like figure eight. There is only a single lane in one direction with an intersection.
        Your target is to give an advice for human driver to help all vehicles pass the intersection quickly and smoothly.
        Your speed is {round(env.k.vehicle.get_speed(self.veh_id), 2)} m/s, IDM acceleration is {round(IDM_acc, 2)} m/s^2, and lane position is {round(env.k.vehicle.get_position(self.veh_id), 2)} m. 
        There are other vehicles driving around you, and below is their basic information:
        """)
        for i in range(len(speed)):
            scenario_description += f" - Vehicle {i} is driving on the same lane as you. The speed of it is {round(speed[i], 2)} m/s, and lane position is {round(pos[i], 2)} m.\n"
        
        strategy_instruction = textwrap.dedent(f"""
        {delimiter} Here I'll tell you a strategy to make decision. PLEASE TRY TO MAKE USE OF IT.
        In figure eight scenario, most of the vehicles gather on the intersections to cause traffic jams. But there are few cars in the other area.
        We should try to balance the distribution of vehicles.
        - If there are many cars near the intersection and you're far away from it, slow down.
        - If there are few cars near the intersection and you're far away from it, speed up.
        Specifically, you can monitor the density of vehicles near the intersection, whose position locates between 0 to 0.15, or whose postion locates between 0.5 and 0.65.
        So you can count the number of vehicles above
        - If there are more than 5 vehicles near the intersection, you should slow down or take action 2.
        - If there are less than 2 vehicles near the intersection, you should speed up or take action 1.
        - Otherwise, you can maintain the speed or take action 0.
        """)

        action_query = textwrap.dedent(f"""
        {delimiter} Now the ego vehicle gives acceleration {round(IDM_acc, 2)} m/s^2.
        Now please give your advice among 3 actions, you MUST use the format as above to output a python dictionary.
        {delimiter}
        The driving task usually invovles many steps. You can break this task down into subtasks and complete them one by one. 
        There is no rush to give a final answer unless you are confident that the answer is correct.
        Make the decision as best you can. Begin! """)
        scenario_description += strategy_instruction + action_query

        # action = 2
        # print(env.k.vehicle.get_x_by_id("llm_0") / env.k.network.length())
        action = self.DA.make_decision(scenario_description, env)

        # Log communication and decision process
        #log_communication(self.veh_id, action)
        log_decision_theory(self.veh_id, scenario_description)

        if action == 1:
            IDM_acc += abs(IDM_acc) * 0.5
        elif action == 2:
            IDM_acc -= abs(IDM_acc) * 0.5

        return IDM_acc

"""
# Integrate reward calculation and speed measurement
def run_simulation(env, llm_controller):
    total_reward = 0
    total_speed = 0
    num_vehicles = len(env.k.vehicle.get_ids())
    for step in range(env.sim_params.simulation_step):
        env.step()
        reward = calculate_reward(env)
        total_reward += reward
        for veh_id in env.k.vehicle.get_ids():
            total_speed += env.k.vehicle.get_speed(veh_id)
        llm_controller.get_accel(env)
    average_speed = total_speed / (num_vehicles * env.sim_params.simulation_step)
    print(f"Total Reward: {total_reward}")
    print(f"Average Speed: {average_speed} m/s")
"""