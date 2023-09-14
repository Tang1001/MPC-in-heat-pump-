import pandas as pd
import numpy as np
from scipy.io import loadmat
from matplotlib.dates import HourLocator, DateFormatter
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pyomo.environ import *
from dataprocess import load_and_preprocess_data
from heatpump import initial_T_setting
from heatpump import parameters_tank
from heatpump import system_equation
import datetime
# from datetime import datetime, timedelta
from numpy import linalg as LA
from plot_temp import plot_show_tanktemp
from plot_temp import plot_show_pipetemp


#Sample time
sample_time=5
dk = sample_time / 60

#%% Read data 7:00~19:00
# interpolated data are 1-min
# 1min
Temp_out_water_he, Temp_out_water_he_interp=load_and_preprocess_data('./data/Mon 1 May 2023 outlet water T in heat exchanger.csv',sample_time)
# print(Temp_out_water_he)
FlowRate_water_hp, FlowRate_water_hp_interp=load_and_preprocess_data('./data/Mon 1 May 2023 water flow rate in heat pump.csv',sample_time)
Temp_bottom_water_tank2, Temp_bottom_water_tank2_interp=load_and_preprocess_data('./Data/Mon 1 May 2023 water T in bottom 2 tank.csv',sample_time)
Temp_upper_water_tank1, Temp_upper_water_tank1_interp=load_and_preprocess_data('./data/Mon 1 May 2023 water T in upper 1 tank.csv',sample_time)
Temp_ambient, Temp_ambient_interp=load_and_preprocess_data('./data/Mon 1 May 2023 ambient air T.csv',sample_time)
Temp_out_water_hp, Temp_out_water_hp_interp=load_and_preprocess_data('./data/Mon 1 May 2023 outlet water T in heat pump.csv',sample_time)
Temp_in_water_hp, Temp_in_water_hp_interp=load_and_preprocess_data('./data/Mon 1 May 2023 inlet water T in heat pump.csv',sample_time)

# 8min
Temp_upper_water_tank2,Temp_upper_water_tank2_interp=load_and_preprocess_data('./data/Mon 1 May 2023 water T in upper 2 tank.csv',sample_time)

# 1h
Consp_water,Consp_water_interp=load_and_preprocess_data('./data/Mon 1 May 2023 water V consumption.csv',sample_time)
#
FlowRate_water_hp_interp['value'] = FlowRate_water_hp_interp['value'] * 1000
Consp_water_interp['value'] = Consp_water_interp['value'] * 1000

Consp_water_interp_value=Consp_water_interp['value']
# print(Consp_water_interp_value)
# m_s_dot = np.zeros(3)
# m_s_dot[0:2]=Consp_water_interp['value'][0:2]
# print(m_s_dot)
# Consp_water_interp_value.append(2)
# print(Consp_water_interp_value)

# print("Temp_out_water_he_interp")
# print(Temp_out_water_he_interp['value'][0:10])
# print(Temp_out_water_he['value'].shape)
# print(Temp_out_water_he_interp['value'].shape)
#
# print("FlowRate_water_hp_interp")
# print(FlowRate_water_hp_interp['value'][14:20])
#
# print("Temp_bottom_water_tank2_interp")
# print(Temp_bottom_water_tank2_interp['value'][0:10])
#
# print("Temp_upper_water_tank1_interp")
# print(Temp_upper_water_tank1_interp['value'][0:10])
#
# print("Temp_ambient_interp")
# print(Temp_ambient_interp['value'][0:10])
#
# print("Temp_out_water_hp_interp")
# print(Temp_out_water_hp_interp['value'][0:10])
# #
# print("Temp_in_water_hp_interp")
# print(Temp_in_water_hp_interp['value'][0:10])
#
# print("Temp_upper_water_tank2_interp")
# print(Temp_upper_water_tank2_interp['value'][0:30])
# print(Temp_upper_water_tank2['value'][0:10])

# print("Consp_water_interp")
# print(Consp_water_interp['value'][0:10])
# print(Consp_water['value'])



df_price = pd.read_csv('./data/slow_Price.csv',header=None)
df_price=df_price.iloc[:,7:20]
# print(df_price.head())
# df_5min = df.repeat(12).reset_index(drop=True)
data_5min = np.repeat(df_price.values.flatten(), 12)
df_price_5min = pd.DataFrame(data_5min.reshape(df_price.shape[0], -1))
df_price_5min = df_price_5min.iloc[:,:-11]
# print(df_price_5min[0:10])
day_prices = pd.DataFrame(data=df_price_5min, index=[3])
# print(df_price)
# print(day_prices)
# print(day_prices.iloc[0,:])
#
# print(Consp_water_interp.index[0].date())
# print(type(Consp_water_interp.index[0]))


## Read data
# Load data from .mat file
data_mpc = loadmat("optimal_MPC_Model_params.mat")
data_model = loadmat("optimal_params.mat")
# print(data_model['P_optimal'])

# Initial Temperature Setting
m_layer_tank1 = np.array([250, 250])

# Create an array of zeros with size 4
m_layer_tank2 = np.zeros(4)

# Assign values to m_layer_tank2 from P_optimal in the .mat data
for i in range(4):
    m_layer_tank2[i] = data_model['P_optimal'][0][i + 1]


T_initial_layers=initial_T_setting(2,4,m_layer_tank1,m_layer_tank2,Temp_upper_water_tank1_interp['value'][0],Temp_upper_water_tank2_interp['value'][0],Temp_bottom_water_tank2_interp['value'][0])
X_initial = np.concatenate([[Temp_out_water_hp_interp['value'][0]],[Temp_in_water_hp_interp['value'][0]],[Temp_out_water_he_interp['value'][0]],[Temp_bottom_water_tank2['value'][0]],T_initial_layers])
# print("X_initial")
# print(X_initial)
# # print(X_initial)
# # print(T_initial_layers)
# # print(m_layer_tank1)
# # print(m_layer_tank2)
# # print(Temp_upper_water_tank1_interp['value'][0])
# # print(Temp_upper_water_tank2_interp['value'][0])
# # print(Temp_bottom_water_tank2_interp['value'][0])








# Number of states and controls
num_states = 10
# num_controls = 1


# Initial condition
x0 = X_initial


# Parameters and initializations
# Nsim = 25
Nsim = len(Temp_upper_water_tank2_interp)
# print(Nsim)
Nsim_all = len(Temp_upper_water_tank2_interp)
Xsim_all = np.zeros((num_states, Nsim_all+1))
Usim_all = np.zeros(Nsim_all)
Xsim_all[:, 0] = X_initial
# past_horizon = 12
# num_switch = 0


# print(Temp_out_water_he_interp)


## Simulation with original strategy
for i in range(Nsim_all):
    m_s_dot = Consp_water_interp['value'][i]
    T_amb = Temp_ambient['value'][i]
    if FlowRate_water_hp_interp['value'][i]>0:
        Usim_all[i]=1
    else:
        Usim_all[i]=0
    Xsim_all[:, i+1] = system_equation(Xsim_all[:, i], Usim_all[i], m_s_dot, T_amb, dk, data_mpc, data_model)

# plot_show_tanktemp(Temp_upper_water_tank1_interp, Temp_upper_water_tank2_interp, Temp_bottom_water_tank2_interp, Xsim_all,Nsim_all)
# plot_show_pipetemp(Temp_out_water_hp_interp,Temp_in_water_hp_interp,Temp_out_water_he_interp, Xsim_all,Nsim_all)


# plt.plot(Temp_upper_water_tank1_interp.index,Temp_upper_water_tank1_interp['value'])
# plt.show()



cp = 4186
R_out_hp_loss, R_in_hp_loss, R_in_tank_loss, R_thermal_hp_pipe, R_pipe_upper, R_pipe_bottom, m_out_hp_pipe, m_in_hp_pipe, m_in_tank_pipe, T_diff_he, m_p_dot, P_aver = data_mpc['P_MPC_optimal'][0]

P_optimal = data_model['P_optimal'][0]
R_tank12 = P_optimal[0]
m_layer_tank2 = P_optimal[1:5]
diff_T_c = P_optimal[5]
T_supply = P_optimal[6]
m_c_dot = P_optimal[7]

m_layer_tank1 = np.array([250, 250])
h_layer_tank1, h_layer_tank2, R_layer_tank1, R_layer_tank2, R_thermal_tank1, R_thermal_tank2 = parameters_tank(2, 4, m_layer_tank1, m_layer_tank2)

# Load cop_P3 data
data_cop = loadmat("cop_P3.mat")
# print(data['cop_3'].shape)
cop_3 = data_cop['cop_3']


m1, m2 = m_layer_tank1
m3, m4, m5, m6 = m_layer_tank2

R12 = R_thermal_tank1
R34, R45, R56 = R_thermal_tank2

# print(Usim_all[ 0:20])
# print(day_prices.iloc[0, 0:20])
# print()
# print(Usim_all[ 0:20]*P_aver*dk/1000*day_prices.iloc[0, 0:20])


## Ploting Heat Pump Original Strategy
# fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
#
# # First subplot
# ax1.step(Consp_water_interp.index[:Nsim_all], Usim_all[ :Nsim_all], where="post")
# ax1.set_title('Heat Pump On/Off States')
# ax1.set_ylabel('On/Off')
# ax1.set_yticks([0, 1])
# ax1.grid()
#
# # Second subplot
# ax2.plot(Consp_water_interp.index[:Nsim_all], Usim_all[ :Nsim_all]*P_aver*dk/1000, marker="o")
# W_consump = sum(Usim_all[ :Nsim_all]*P_aver*dk/1000)
# str = f'Heat Pump Energy Consumption: {W_consump:.2f} kWh'
# ax2.set_title(str)
# ax2.set_ylabel('kWh')
# ax2.grid()
#
# # Third subplot
# # Demand Request Plot
# ax3.plot(Consp_water_interp.index[:Nsim_all], Usim_all[ :Nsim_all]*P_aver*dk/1000*day_prices.iloc[0, :Nsim_all], marker="o")
# c_cost = sum(Usim_all[ :Nsim_all]*P_aver*dk/1000*day_prices.iloc[0, :Nsim_all])
# str = f'Heat Pump Energy Cost: {c_cost:.2f} Euro'
# ax3.set_title(str)
# ax3.set_ylabel('Euro')
# ax3.grid()
#
# # Setting x-axis date formatter for all subplots (since sharex=True)
# minute_formatter = DateFormatter('%Y-%m-%d %H:%M')
# ax3.xaxis.set_major_formatter(minute_formatter)
# hour_locator = HourLocator()
# ax3.xaxis.set_major_locator(hour_locator)
# ax3.tick_params(axis='x', rotation=45)
#
# # Adjusting layout
# # plt.xticks(rotation=45)
# plt.setp(ax3.get_xticklabels(), ha="right")
# str = 'Heat Pump Original Strategy'
# plt.suptitle(str)
# plt.tight_layout()
# plt.show()





# MPC
N = 30  # prediction horizon
control_horizon = 12  # control horizon
w1=100000    #energy consumption
w2=10000000   #relax factor
w3 = 50   #switch number
w4 = 0   #price
def solve_NLMPC(x0, m_s_dot ,T_amb,C_price,u_last):
    m = ConcreteModel()

    m.t = RangeSet(0, N)    #The set m.t has a length of N+1
    m.t_minus_one = RangeSet(0, N - 1)    #The set m.t_minus_one has a length of N
    # print('t')
    # print(m.t.value)
    m.x = Var(m.t, range(num_states), within=NonNegativeReals)
    m.u = Var(m.t_minus_one, within=Binary)
    # m.T_random = Var(within=NonNegativeReals, bounds=(0,None))
    m.T_random = Var(within=NonNegativeReals, bounds=(0,8))


    # Objective

    m.obj = Objective(expr= w1* sum( P_aver * dk * m.u[t] for t in m.t_minus_one) + w2 * m.T_random + w3*((m.u[0]-u_last)**2+sum((m.u[t+1]-m.u[t])**2 for t in RangeSet(0, N - 2))) + w4 * sum(P_aver/1000 * dk *C_price[t] * m.u[t] for t in m.t_minus_one), sense=minimize)


    def init(m,i):
        return m.x[0, i] == x0[i]

    m.cons_init = Constraint(range(num_states), rule = init)
    # Fix initial condition
    # for i in range(num_states):
    #     m.x[0, i].fix(x0[i])
    # m.X[0].fix(T0)

    def num_switch_rule(m):

        # return (m.u[0] - u_last) ** 2 + sum((m.u[t + 1] - m.u[t]) ** 2 for t in range(0, 6)) <= 6
        return (m.u[0] - u_last) ** 2 + sum((m.u[t + 1] - m.u[t]) ** 2 for t in range(0, N - 1)) <= 4
    m.num_switch = Constraint(rule=num_switch_rule)

    # Constraints
    def set_point_rule_lower(m, t):
        return m.x[t, 4] >= 62 - m.T_random
        # return m.x[t, 4] >= 60 - m.T_random[t]
        # return m.x[t, 4] >= 60 - m.T_random_bottom[t]
    def set_point_rule_upper(m, t):
        return m.x[t, 4] <= 72 + m.T_random
        # return m.x[t, 4] <= 75 + m.T_random[t]
        # return m.x[t, 4] <= 75 + m.T_random_upper[t]
    m.set_point1_lower = Constraint(m.t, rule=set_point_rule_lower)
    m.set_point1_upper = Constraint(m.t, rule=set_point_rule_upper)
    # m.set_point1 = Constraint(expr=(m.x[t,4] >=58 for t in m.t))
    # m.set_point2 = Constraint(expr=(m.x[t, 4] <= 75 for t in m.t))

    m.dynamics = ConstraintList()
    for t in m.t:
        if t < N:
            cop_f = cop_3[0][0] + cop_3[1][0] * m.x[t, 1] + cop_3[2][0] * T_amb[t] + cop_3[3][0] * m.x[t, 1] * T_amb[t]
            Q_hp = P_aver * cop_f * 3600
            T_diff_in_out_tank = m.x[t, 2] - m.x[t, 3]


            m.dynamics.add(m.x[t + 1,0] == (m.x[t,1] + Q_hp / (m_p_dot * cp)) * m.u[t] + (
                    m.x[t,0] - (R_out_hp_loss * (m.x[t,0] - T_amb[t]) + R_thermal_hp_pipe * (m.x[t,0] - m.x[t,1])) * (
                    60 * 60) * dk / (m_out_hp_pipe * cp)) * (1 - m.u[t]))

            m.dynamics.add(m.x[t + 1,1] == (T_diff_he + T_supply * m_s_dot[t] / m_p_dot + m.x[t,9] * (1 - m_s_dot[t] / m_p_dot)) * m.u[t] + (
                    m.x[t,1] - (R_in_hp_loss * (m.x[t,1] - T_amb[t]) + R_thermal_hp_pipe * (m.x[t,1] - m.x[t,0])) * (
                    60 * 60) * dk / (m_in_hp_pipe * cp)) * (1 - m.u[t]))

            m.dynamics.add(m.x[t + 1,2] == ((m.x[t,1] + Q_hp / (m_p_dot * cp)) - T_diff_he) * m.u[t] + (m.x[t,2] - (
                    R_in_tank_loss * (m.x[t,2] - T_amb[t]) + R_pipe_upper * (m.x[t,2] - m.x[t,4]) + R_pipe_bottom * (
                    m.x[t, 2] - m.x[t,9])) * (60 * 60) * dk / (m_in_tank_pipe * cp)) * (1 - m.u[t]))

            m.dynamics.add(m.x[t + 1,3] == T_supply * m_s_dot[t] / m_p_dot + m.x[t,9] * (1 - m_s_dot[t] / m_p_dot))

            # Layer tank dynamics
            m.dynamics.add(m.x[t + 1,4] == m.x[t,4] + dk * (m_p_dot * m.u[t] * cp * (m.x[t,2] - m.x[t,4]) - R12 * (m.x[t,4] - m.x[t,5]) * (60 * 60) - (
                    m_c_dot - m_s_dot[t]) * cp * diff_T_c + m_s_dot[t] * (1 - m.u[t]) * cp * (m.x[t,5] - m.x[t,4])) / (m1 * cp))
            m.dynamics.add(m.x[t + 1,5] == m.x[t,5] + dk * ((m_p_dot - m_s_dot[t]) * m.u[t] * cp * (m.x[t,4] - m.x[t,5]) + R12 * (m.x[t,4] - m.x[t,5]) * (
                    60 * 60) - R_tank12 * (m.x[t,5] - m.x[t,6]) + m_s_dot[t] * (1 - m.u[t]) * cp * (m.x[t,6] - m.x[t,5])) / (
                               m2 * cp))
            m.dynamics.add(m.x[t + 1,6] == m.x[t,6] + dk * ((m_p_dot - m_s_dot[t]) * m.u[t] * cp * (m.x[t,5] - m.x[t,6]) - R34 * (m.x[t,6] - m.x[t,7]) * (
                    60 * 60) + R_tank12 * (m.x[t,5] - m.x[t,6]) + m_s_dot[t] * (1 - m.u[t]) * cp * (m.x[t,7] - m.x[t,6])) / (
                               m3 * cp))
            m.dynamics.add(m.x[t + 1,7] == m.x[t,7] + dk * ((m_p_dot - m_s_dot[t]) * m.u[t] * cp * (m.x[t,6] - m.x[t,7]) + R34 * (m.x[t,6] - m.x[t,7]) * (
                    60 * 60) - R45 * (m.x[t,7] - m.x[t,8]) * (60 * 60) + m_s_dot[t] * (1 - m.u[t]) * cp * (
                                              m.x[t,8] - m.x[t,7])) / (m4 * cp))
            m.dynamics.add(m.x[t + 1,8] == m.x[t,8] + dk * ((m_p_dot - m_s_dot[t]) * m.u[t] * cp * (m.x[t,7] - m.x[t,8]) + R45 * (m.x[t,7] - m.x[t,8]) * (
                    60 * 60) - R56 * (m.x[t,8] - m.x[t,9]) * (60 * 60) + m_s_dot[t] * (1 - m.u[t]) * cp * (
                                              m.x[t,9] - m.x[t,8])) / (m5 * cp))
            m.dynamics.add(m.x[t + 1,9] == m.x[t,9] + dk * (
                    (m_p_dot - m_s_dot[t]) * m.u[t] * cp * (m.x[t,8] - m.x[t,9]) + R56 * (m.x[t,8] - m.x[t,9]) * (60 * 60) + m_s_dot[t] * (
                        1 - m.u[t]) * cp * (T_supply - m.x[t,9])) / (m6 * cp))

    # Solve
    # solver = SolverFactory('scip')
    solver = SolverFactory('gurobi')
    solver.options['NonConvex']=2
    solver.options["TimeLimit"] = 1800  # set max time to 600 seconds
    # solver.options["MIPGap"] = 0.01  # 1% gap
    # solver.options["feastol"] = 1e-5  # set primal feasibility tolerance
    # solver.options["dualfeastol"] = 1e-5  # set dual feasibility tolerance
    # solver.options["emphasis"] = "optimal"  # set emphasis on optimality
    # solver.options["heuristics/shifting/freq"] = 5
    # solver.options['ResultFile'] = 'infeasible.ilp'
    solver.solve(m, tee=True)
    # solver.solve(m)
    print('Relax')
    print([m.T_random.value])
    # print([m.T_random[t].value for t in m.t])
    # print([m.T_random_upper[t].value for t in m.t])
    # print([m.T_random_bottom[t].value for t in m.t])

    return [m.x[t,:].value for t in m.t], [m.u[t].value for t in m.t_minus_one]



Dr_begin_time_hour = 14
Dr_begin_time_min = 0
Dr_begin_time = datetime.time(Dr_begin_time_hour, Dr_begin_time_min)
Dr_end_time_hour = 15
Dr_end_time_min = 0
Dr_end_time=datetime.time(Dr_end_time_hour, Dr_end_time_min)

DR_bound = 2    #kwh
DR_reward=100000#30000
def solve_NLMPC_DR(x0, m_s_dot ,T_amb,C_price,u_last,N_dr,Dr_begin_index,Dr_end_index):
    m = ConcreteModel()

    m.t = RangeSet(0, N_dr)
    m.t_minus_one = RangeSet(0, N_dr - 1)
    # print('t')
    # print(m.t.value)
    m.x = Var(m.t, range(num_states), within=NonNegativeReals)
    m.u = Var(m.t_minus_one, within=Binary)
    m.T_random = Var(within=NonNegativeReals, bounds=(0,None))
    m.DR_bin = Var(within=Binary)
    # m.T_random = Var(within=NonNegativeReals, bounds=(0,10))


    # Objective

    m.obj = Objective(expr= w1* sum( P_aver *dk * m.u[t]  for t in m.t_minus_one) + w2 * m.T_random + w3*((m.u[0]-u_last)**2+sum((m.u[t+1]-m.u[t])**2 for t in RangeSet(0, N_dr - 2))) + w4 * sum(P_aver/1000*dk*C_price[t] * m.u[t] for t in m.t_minus_one) - m.DR_bin*DR_reward, sense=minimize)
    def dr_rule(m):
        return sum( P_aver *dk * m.u[t]  for t in range(Dr_begin_index, Dr_end_index+1)) <= DR_bound + (1-m.DR_bin)*10000
    m.dr_request = Constraint(rule = dr_rule)


    def init(m,i):
        return m.x[0, i] == x0[i]
    m.cons_init = Constraint(range(num_states), rule = init)
    # Fix initial condition
    # for i in range(num_states):
    #     m.x[0, i].fix(x0[i])
    # m.X[0].fix(T0)

    def num_switch_rule(m):
        # return (m.u[0] - u_last) ** 2 + sum((m.u[t + 1] - m.u[t]) ** 2 for t in range(0, 6)) <= 6
        return (m.u[0] - u_last) ** 2 + sum((m.u[t + 1] - m.u[t]) ** 2 for t in range(0, N_dr - 1)) <= 6
    m.num_switch = Constraint(rule=num_switch_rule)

    # Constraints
    def set_point_rule_lower(m, t):
        return m.x[t, 4] >= 55 - m.T_random
        # return m.x[t, 4] >= 60 - m.T_random[t]
        # return m.x[t, 4] >= 60 - m.T_random_bottom[t]
    def set_point_rule_upper(m, t):
        return m.x[t, 4] <= 75 + m.T_random
        # return m.x[t, 4] <= 75 + m.T_random[t]
        # return m.x[t, 4] <= 75 + m.T_random_upper[t]
    m.set_point1_lower = Constraint(m.t, rule=set_point_rule_lower)
    m.set_point1_upper = Constraint(m.t, rule=set_point_rule_upper)
    # m.set_point1 = Constraint(expr=(m.x[t,4] >=58 for t in m.t))
    # m.set_point2 = Constraint(expr=(m.x[t, 4] <= 75 for t in m.t))

    m.dynamics = ConstraintList()
    for t in m.t:
        if t < N_dr:
            cop_f = cop_3[0][0] + cop_3[1][0] * m.x[t, 1] + cop_3[2][0] * T_amb[t] + cop_3[3][0] * m.x[t, 1] * T_amb[t]
            Q_hp = P_aver * cop_f * 3600
            T_diff_in_out_tank = m.x[t, 2] - m.x[t, 3]


            m.dynamics.add(m.x[t + 1,0] == (m.x[t,1] + Q_hp / (m_p_dot * cp)) * m.u[t] + (
                    m.x[t,0] - (R_out_hp_loss * (m.x[t,0] - T_amb[t]) + R_thermal_hp_pipe * (m.x[t,0] - m.x[t,1])) * (
                    60 * 60) * dk / (m_out_hp_pipe * cp)) * (1 - m.u[t]))

            m.dynamics.add(m.x[t + 1,1] == (T_diff_he + T_supply * m_s_dot[t] / m_p_dot + m.x[t,9] * (1 - m_s_dot[t] / m_p_dot)) * m.u[t] + (
                    m.x[t,1] - (R_in_hp_loss * (m.x[t,1] - T_amb[t]) + R_thermal_hp_pipe * (m.x[t,1] - m.x[t,0])) * (
                    60 * 60) * dk / (m_in_hp_pipe * cp)) * (1 - m.u[t]))

            m.dynamics.add(m.x[t + 1,2] == ((m.x[t,1] + Q_hp / (m_p_dot * cp)) - T_diff_he) * m.u[t] + (m.x[t,2] - (
                    R_in_tank_loss * (m.x[t,2] - T_amb[t]) + R_pipe_upper * (m.x[t,2] - m.x[t,4]) + R_pipe_bottom * (
                    m.x[t, 2] - m.x[t,9])) * (60 * 60) * dk / (m_in_tank_pipe * cp)) * (1 - m.u[t]))

            m.dynamics.add(m.x[t + 1,3] == T_supply * m_s_dot[t] / m_p_dot + m.x[t,9] * (1 - m_s_dot[t] / m_p_dot))

            # Layer tank dynamics
            m.dynamics.add(m.x[t + 1,4] == m.x[t,4] + dk * (m_p_dot * m.u[t] * cp * (m.x[t,2] - m.x[t,4]) - R12 * (m.x[t,4] - m.x[t,5]) * (60 * 60) - (
                    m_c_dot - m_s_dot[t]) * cp * diff_T_c + m_s_dot[t] * (1 - m.u[t]) * cp * (m.x[t,5] - m.x[t,4])) / (m1 * cp))
            m.dynamics.add(m.x[t + 1,5] == m.x[t,5] + dk * ((m_p_dot - m_s_dot[t]) * m.u[t] * cp * (m.x[t,4] - m.x[t,5]) + R12 * (m.x[t,4] - m.x[t,5]) * (
                    60 * 60) - R_tank12 * (m.x[t,5] - m.x[t,6]) + m_s_dot[t] * (1 - m.u[t]) * cp * (m.x[t,6] - m.x[t,5])) / (
                               m2 * cp))
            m.dynamics.add(m.x[t + 1,6] == m.x[t,6] + dk * ((m_p_dot - m_s_dot[t]) * m.u[t] * cp * (m.x[t,5] - m.x[t,6]) - R34 * (m.x[t,6] - m.x[t,7]) * (
                    60 * 60) + R_tank12 * (m.x[t,5] - m.x[t,6]) + m_s_dot[t] * (1 - m.u[t]) * cp * (m.x[t,7] - m.x[t,6])) / (
                               m3 * cp))
            m.dynamics.add(m.x[t + 1,7] == m.x[t,7] + dk * ((m_p_dot - m_s_dot[t]) * m.u[t] * cp * (m.x[t,6] - m.x[t,7]) + R34 * (m.x[t,6] - m.x[t,7]) * (
                    60 * 60) - R45 * (m.x[t,7] - m.x[t,8]) * (60 * 60) + m_s_dot[t] * (1 - m.u[t]) * cp * (
                                              m.x[t,8] - m.x[t,7])) / (m4 * cp))
            m.dynamics.add(m.x[t + 1,8] == m.x[t,8] + dk * ((m_p_dot - m_s_dot[t]) * m.u[t] * cp * (m.x[t,7] - m.x[t,8]) + R45 * (m.x[t,7] - m.x[t,8]) * (
                    60 * 60) - R56 * (m.x[t,8] - m.x[t,9]) * (60 * 60) + m_s_dot[t] * (1 - m.u[t]) * cp * (
                                              m.x[t,9] - m.x[t,8])) / (m5 * cp))
            m.dynamics.add(m.x[t + 1,9] == m.x[t,9] + dk * (
                    (m_p_dot - m_s_dot[t]) * m.u[t] * cp * (m.x[t,8] - m.x[t,9]) + R56 * (m.x[t,8] - m.x[t,9]) * (60 * 60) + m_s_dot[t] * (
                        1 - m.u[t]) * cp * (T_supply - m.x[t,9])) / (m6 * cp))

    # Solve
    # solver = SolverFactory('scip')
    solver = SolverFactory('gurobi')
    solver.options['NonConvex']=2
    solver.options["TimeLimit"] = 1800  # set max time to 600 seconds
    solver.options["MIPGap"] = 0.01  # 1% gap
    # solver.options["feastol"] = 1e-5  # set primal feasibility tolerance
    # solver.options["dualfeastol"] = 1e-5  # set dual feasibility tolerance
    # solver.options["emphasis"] = "optimal"  # set emphasis on optimality
    # solver.options["heuristics/shifting/freq"] = 5
    # solver.options['ResultFile'] = 'infeasible.ilp'
    solver.solve(m, tee=True)
    # solver.solve(m)
    print('Relax')
    print([m.T_random.value])
    # print([m.T_random[t].value for t in m.t])
    # print([m.T_random_upper[t].value for t in m.t])
    # print([m.T_random_bottom[t].value for t in m.t])

    return [m.x[t,:].value for t in m.t], [m.u[t].value for t in m.t_minus_one]



# Simulation of NMPC in closed loop
x0 = X_initial
# N_mpc=Nsim - N +1
N_mpc=Nsim
Xsim = np.zeros((num_states,N_mpc+1))
Usim = np.zeros(N_mpc)
Xsim[:,0] = x0
u_last=0

m_s_dot = np.zeros(N)
T_amb = np.zeros(N)
C_price = np.zeros(N)




Dr_length_min = (Dr_end_time_hour-Dr_begin_time_hour)*60+(Dr_end_time_min-Dr_begin_time_min)
Dr_flag = 0

N_delta = datetime.timedelta(minutes=N*sample_time)


control_horizon_flag=0
for k in range(N_mpc):

    #Demand Respond
    # if (Consp_water_interp.index[k] + N_delta).time() == Dr_begin_time:
    #     Dr_flag = 1
    # if Consp_water_interp.index[k].time() == Dr_end_time:
    #     Dr_flag = 0
    #
    # if Dr_flag == 1:
    #     if Dr_end_time > (Consp_water_interp.index[k] + N_delta).time():
    #         dt1 = datetime.datetime.combine(datetime.datetime.today(), Dr_end_time)
    #         dt2 = datetime.datetime.combine(datetime.datetime.today(), (Consp_water_interp.index[k] + N_delta).time())
    #         time_difference = dt1 - dt2
    #         minutes_difference = time_difference.total_seconds() / 60
    #         N_dr = int(N + minutes_difference / sample_time)
    #     else:
    #         N_dr = N
    #
    #     if Consp_water_interp.index[k].time() >= Dr_begin_time:
    #         Dr_begin_index = 0
    #     else:
    #         dt_begin = datetime.datetime.combine(datetime.datetime.today(), Dr_begin_time)
    #         dt_k = datetime.datetime.combine(datetime.datetime.today(), Consp_water_interp.index[k].time())
    #         time_difference_minutes = (dt_begin - dt_k).total_seconds() / 60
    #         Dr_begin_index = int((time_difference_minutes / sample_time) - 1)
    #     dt_end = datetime.datetime.combine(datetime.datetime.today(), Dr_end_time)
    #     dt_k = datetime.datetime.combine(datetime.datetime.today(), Consp_water_interp.index[k].time())
    #     time_difference_minutes = (dt_end - dt_k).total_seconds() / 60
    #     Dr_end_index = int((time_difference_minutes / sample_time) - 1)
    #     print("N_dr")
    #     print(N_dr)
    #     # print(type(N_dr))
    #     print(Dr_begin_index, Dr_end_index)
    #     m_s_dot_dr = np.zeros(N_dr)
    #     T_amb_dr = np.zeros(N_dr)
    #     C_price_dr = np.zeros(N_dr)

    # if Dr_flag == 0:
    #     if k <= Nsim - N:
    #         m_s_dot[0:N] = Consp_water_interp['value'][k:k + N]
    #         T_amb[0:N] = Temp_ambient['value'][k:k + N]
    #         C_price[0:N] = day_prices.iloc[0, k:k + N]
    #     else:
    #         num_avail = Nsim - k
    #         m_s_dot[0:num_avail] = Consp_water_interp['value'][k:Nsim]
    #         m_s_dot[num_avail:] = 0
    #         T_amb[0:num_avail] = Temp_ambient['value'][k:Nsim]
    #         T_amb[num_avail:] = Temp_ambient['value'][Nsim - 1]
    #         C_price[0:num_avail] = day_prices.iloc[0, k:Nsim]
    #         C_price[num_avail:] = day_prices.iloc[0, Nsim - 1]
    # else:
    #     if k <= Nsim - N_dr:
    #         m_s_dot_dr[0:N_dr] = Consp_water_interp['value'][k:k + N_dr]
    #         T_amb_dr[0:N_dr] = Temp_ambient['value'][k:k + N_dr]
    #         C_price_dr[0:N_dr] = day_prices.iloc[0, k:k + N_dr]
    #     else:
    #         num_avail = Nsim - k
    #         m_s_dot_dr[0:num_avail] = Consp_water_interp['value'][k:Nsim]
    #         m_s_dot_dr[num_avail:] = 0
    #         T_amb_dr[0:num_avail] = Temp_ambient['value'][k:Nsim]
    #         T_amb_dr[num_avail:] = Temp_ambient['value'][Nsim - 1]
    #         C_price_dr[0:num_avail] = day_prices.iloc[0, k:Nsim]
    #         C_price_dr[num_avail:] = day_prices.iloc[0, Nsim - 1]

    if k <= Nsim - N:
        m_s_dot[0:N] = Consp_water_interp['value'][k:k + N]
        T_amb[0:N] = Temp_ambient['value'][k:k + N]
        C_price[0:N] = day_prices.iloc[0, k:k + N]
    else:
        num_avail = Nsim - k
        m_s_dot[0:num_avail] = Consp_water_interp['value'][k:Nsim]
        m_s_dot[num_avail:] = 0
        T_amb[0:num_avail] = Temp_ambient['value'][k:Nsim]
        T_amb[num_avail:] = Temp_ambient['value'][Nsim - 1]
        C_price[0:num_avail] = day_prices.iloc[0, k:Nsim]
        C_price[num_avail:] = day_prices.iloc[0, Nsim - 1]

    if control_horizon_flag==0:
        print(Consp_water_interp.index[k])
        # print(Consp_water_interp['value'][k])
        # x_pred, u_pred = solve_NLMPC(Xsim[:,k], m_s_dot, T_amb,u_last)
        # print(C_price)

        # if Dr_flag==0:
        #     x_pred, u_pred = solve_NLMPC(Xsim[:, k], m_s_dot, T_amb,C_price, u_last)
        # else:
        #     x_pred, u_pred = solve_NLMPC_DR(Xsim[:, k], m_s_dot_dr, T_amb_dr, C_price_dr, u_last, N_dr, Dr_begin_index, Dr_end_index)

        x_pred, u_pred = solve_NLMPC(Xsim[:, k], m_s_dot, T_amb, C_price, u_last)

        print(u_pred)
        Usim[k] = round(u_pred[0])
        # print([round(x) for x in u_pred])
        control_horizon_flag+=1
    else:
        print(Consp_water_interp.index[k])
        # print(Consp_water_interp['value'][k])
        Usim[k] = round(u_pred[control_horizon_flag])
        print([Usim[k]])
        control_horizon_flag += 1
        if control_horizon_flag==control_horizon:
            control_horizon_flag=0

    # print("m_s_dot")
    # print(m_s_dot[0])
    # print("T_amb")
    # print(T_amb[0])


    Xsim[:, k + 1] = system_equation(Xsim[:, k], Usim[k], m_s_dot[0], T_amb[0], dk, data_mpc, data_model)
    u_last=Usim[k]





# Plot results
plt.rcParams['text.usetex'] = True
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

# First subplot
ax1.plot(Consp_water_interp.index[:N_mpc], Xsim[4, :N_mpc], label=r'$T_{1}$', marker="o")
ax1.legend(fontsize='x-large')
ax1.set_title('Temperature of the Upper Tank 1')
ax1.set_ylabel('Temperature (℃)')
ax1.grid()

# Second subplot
ax2.plot(Consp_water_interp.index[:N_mpc], Consp_water_interp['value'][:N_mpc], label=r'$\dot{m}_{s}$', marker="o")
ax2.legend(fontsize='x-large')
ax2.set_title('Water Consumption/Supplied Water Flow Rate')
ax2.set_ylabel('Flow Rate (L/h)')
ax2.grid()

# Third subplot
# Demand Request Plot
# date_of_interest = Consp_water_interp.index[0].date()
# start_time = pd.Timestamp.combine(date_of_interest, Dr_begin_time)
# end_time = pd.Timestamp.combine(date_of_interest, Dr_end_time)
# ax3.axvspan(start_time, end_time, facecolor="green", alpha=0.5)

ax3.step(Consp_water_interp.index[:N_mpc], Usim, label="Heat Pump Action", where="post")
ax3.legend(fontsize='x-large')
ax3.set_title('Heat Pump On/Off States')
ax3.set_ylabel('On/Off')
ax3.set_yticks([0, 1])
ax3.grid()

# Setting x-axis date formatter for all subplots (since sharex=True)
minute_formatter = DateFormatter('%Y-%m-%d %H:%M')
ax3.xaxis.set_major_formatter(minute_formatter)
hour_locator = HourLocator()
ax3.xaxis.set_major_locator(hour_locator)
ax3.tick_params(axis='x', rotation=45)

# Adjusting layout
# plt.xticks(rotation=45)
plt.setp(ax3.get_xticklabels(), ha="right")
str = f'w1={w1}, w2={w2:.0f}, w3={w3},w4={w4}\n N={N}, NC={control_horizon}, dT={sample_time}min'
plt.suptitle(str)
plt.tight_layout()
plt.show()






fig2, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

# First subplot
ax1.step(Consp_water_interp.index[:Nsim_all], Usim[ :Nsim_all], where="post")
ax1.set_title('Heat Pump On/Off States')
ax1.set_ylabel('On/Off')
ax1.set_yticks([0, 1])
ax1.grid()

# Second subplot
ax2.plot(Consp_water_interp.index[:Nsim_all], Usim[ :Nsim_all]*P_aver*dk/1000, marker="o")
W_consump = sum(Usim[ :Nsim_all]*P_aver*dk/1000)
str = f'Heat Pump Energy Consumption: {W_consump:.2f} kWh'
ax2.set_title(str)
ax2.set_ylabel('kWh')
ax2.grid()

# Third subplot
# Demand Request Plot
ax3.plot(Consp_water_interp.index[:Nsim_all], Usim[ :Nsim_all]*P_aver*dk/1000*day_prices.iloc[0, :Nsim_all], marker="o")
c_cost = sum(Usim[ :Nsim_all]*P_aver*dk/1000*day_prices.iloc[0, :Nsim_all])
str = f'Heat Pump Energy Cost: {c_cost:.2f} Euro'
ax3.set_title(str)
ax3.set_ylabel('Euro')
ax3.grid()

# Setting x-axis date formatter for all subplots (since sharex=True)
minute_formatter = DateFormatter('%Y-%m-%d %H:%M')
ax3.xaxis.set_major_formatter(minute_formatter)
hour_locator = HourLocator()
ax3.xaxis.set_major_locator(hour_locator)
ax3.tick_params(axis='x', rotation=45)

# Adjusting layout
# plt.xticks(rotation=45)
plt.setp(ax3.get_xticklabels(), ha="right")
str = 'Heat Pump Propose Strategy'
plt.suptitle(str)
plt.tight_layout()
plt.show()

