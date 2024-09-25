"""Used as an example of ring experiment.

This example consists of 22 IDM cars on a ring creating shockwaves.
"""

from flow.controllers import IDMController, ContinuousRouter, LLMController, RingController
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.core.params import VehicleParams
from flow.envs.ring.accel import AccelEnv, ADDITIONAL_ENV_PARAMS
from flow.networks.ring import RingNetwork, ADDITIONAL_NET_PARAMS


vehicles = VehicleParams()
vehicles.add(
    veh_id="idm",
    acceleration_controller=(IDMController, {}),
    routing_controller=(ContinuousRouter, {}),
    num_vehicles=21)
vehicles.add(
    veh_id="llm",
    acceleration_controller=(LLMController, {}),
    routing_controller=(ContinuousRouter, {}),
    num_vehicles=1,
    color='red')

# avg = 2.88, std = 0.79
# avg = 2.83, std = 0.28
# avg = 2.83, std = 0.29

# avg = 3.04, std = 0.88  <idm>
# avg = 2.95, std = 0.01  <1.0/0.1> track 5 mean
# avg = 2.75, std = 0.03  track lead[0]
# avg = 2.76, std = 0.01  track mean 5 leads
# avg = 2.98, std = 0.18  <0.5/0.2>
# avg = 2.97, std = 0.07  <0.2/0.1>
# avg = 2.97, std = 0.07  <0.2/0.2>
# avg = 2.97, std = 0.16  <0.2/0.3>
# avg = 2.95, std = 0.06  <0.1/0.1>

# num = [27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 10, 5, 3, 2, 1]
# sstd = [0.59, 0.56, 0.58, 0.58, 0.61, 0.62, 0.58, 0.57, 0.50, 0.40, 0.28, 0.20, 0.13, 0, 0, 0.03, 0.02, 0]
# avg = [0.99, 1.31, 1.66, 2.04, 2.44, 2.88, 3.36, 3.88, 4.44, 5.07, 5.76, 6.56, 7.46, 13.80, 22.69, 24.80, 25.3, 26.60]
# std = [0.85, 0.86, 0.88, 0.89, 0.91, 0.96, 1.02, 1.10, 1.16, 1.24, 1.28, 1.32, 1.33, 2.68, 5.71, 6.48, 6.56, 7.17]
# std(vel[500:])
# std / avg
# The waves almost disappear at # vel = 15

'''
for i in range(11):
    vehicles.add(
        veh_id=f"idm_{i}",
        acceleration_controller=(IDMController, {}),
        routing_controller=(ContinuousRouter, {}),
        num_vehicles=1)
    vehicles.add(
        veh_id=f"llm_{i}",
        acceleration_controller=(CustomController, {}),
        routing_controller=(ContinuousRouter, {}),
        num_vehicles=1,
        color='red')
'''

flow_params = dict(
    # name of the experiment
    exp_tag='ring',

    # name of the flow environment the experiment is running on
    env_name=AccelEnv,

    # name of the network class the experiment is running on
    network=RingNetwork,

    # simulator that is used by the experiment
    simulator='traci',

    # sumo-related parameters (see flow.core.params.SumoParams)
    sim=SumoParams(
        render=True,
        sim_step=0.1,
    ),

    # environment related parameters (see flow.core.params.EnvParams)
    env=EnvParams(
        horizon=1500,
        additional_params=ADDITIONAL_ENV_PARAMS,
    ),

    # network-related parameters (see flow.core.params.NetParams and the
    # network's documentation or ADDITIONAL_NET_PARAMS component)
    net=NetParams(
        additional_params=ADDITIONAL_NET_PARAMS.copy(),
    ),

    # vehicles to be placed in the network at the start of a rollout (see
    # flow.core.params.VehicleParams)
    veh=vehicles,

    # parameters specifying the positioning of vehicles upon initialization/
    # reset (see flow.core.params.InitialConfig)
    initial=InitialConfig(
        bunching=20,
    ),
)
