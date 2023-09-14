import numpy
import numpy as np
from scipy.io import loadmat
from pyomo.environ import *


def parameters_tank(num_layer_tank1, num_layer_tank2, m_layer_tank1, m_layer_tank2):

    # Parameters
    k = 0.6  # the thermal conductivity of water  [W/m·K]
    r = 0.65 / 2  # radius of tank  [m]
    A = np.pi * r**2  # cross-sectional area between layers [m^2]
    H = 1.64  # height of tank [m]

    h_layer_tank1 = H * (m_layer_tank1 / 500)
    h_layer_tank2 = H * (m_layer_tank2 / 500)
    # print(h_layer_tank1)
    # print(h_layer_tank2)

    # thermal resistances
    R_layer_tank1 = h_layer_tank1 / (k * A)
    R_layer_tank2 = h_layer_tank2 / (k * A)

    # heat transfer coefficient
    R_thermal_tank1 = np.zeros(num_layer_tank1 - 1)
    R_thermal_tank2 = np.zeros(num_layer_tank2 - 1)

    # print(R_layer_tank1)
    # print(R_layer_tank2)
    for i in range(num_layer_tank1 - 1):
        R_thermal_tank1[i] = 1 / (2 * (R_layer_tank1[i] * R_layer_tank1[i + 1]) / (R_layer_tank1[i] + R_layer_tank1[i + 1]))

    for i in range(num_layer_tank2 - 1):
        R_thermal_tank2[i] = 1 / (2 * (R_layer_tank2[i] * R_layer_tank2[i + 1]) / (R_layer_tank2[i] + R_layer_tank2[i + 1]))

    return h_layer_tank1, h_layer_tank2, R_layer_tank1, R_layer_tank2, R_thermal_tank1, R_thermal_tank2





def initial_T_setting(num_layer_tank1, num_layer_tank2, m_layer_tank1, m_layer_tank2, T0_upper_layer_tank1,
                      T0_upper_layer_tank2, T0_bottom_layer_tank2):
    # Assuming the function parameters_tank() is already defined or imported
    h_layer_tank1, h_layer_tank2, R_layer_tank1, R_layer_tank2, R_thermal_tank1, R_thermal_tank2 = parameters_tank(
        num_layer_tank1, num_layer_tank2, m_layer_tank1, m_layer_tank2)
    R_tank1_tot = np.sum(R_layer_tank1)
    R_tank2_tot = np.sum(R_layer_tank2)

    T_initial_layer_tank1 = np.zeros(num_layer_tank1)
    T_initial_layer_tank2 = np.zeros(num_layer_tank2)

    # Measurement values
    T_initial_layer_tank1[0] = T0_upper_layer_tank1
    T_initial_layer_tank2[0] = T0_upper_layer_tank2
    T_initial_layer_tank2[-1] = T0_bottom_layer_tank2

    # print("T_initial_layer_tank1")
    # print(T_initial_layer_tank1)
    # print("T_initial_layer_tank2")
    # print(T_initial_layer_tank2)

    # Tank1
    for i in range(num_layer_tank1 - 1):
        d_T_tank1 = (T_initial_layer_tank1[0] - T_initial_layer_tank2[0]) * (R_layer_tank1[i] / R_tank1_tot)
        T_initial_layer_tank1[i + 1] = T_initial_layer_tank1[i] - d_T_tank1

    # Tank2
    for i in range(num_layer_tank2 - 2):  # Adjusted the range to match MATLAB's 1-indexing
        d_T_tank2 = (T_initial_layer_tank2[0] - T_initial_layer_tank2[-1]) * (R_layer_tank2[i] / R_tank2_tot)
        T_initial_layer_tank2[i + 1] = T_initial_layer_tank2[i] - d_T_tank2

    T_initial = np.concatenate((T_initial_layer_tank1, T_initial_layer_tank2))

    return T_initial



def system_dynamics_equation(m, tau, i, dk, N, data_mpc,data_model):
    if tau == N - 1:
        return Constraint.Skip

    cp = 4186
    R_out_hp_loss, R_in_hp_loss, R_in_tank_loss, R_thermal_hp_pipe, R_pipe_upper, R_pipe_bottom, m_out_hp_pipe, m_in_hp_pipe, m_in_tank_pipe, T_diff_he, m_p_dot, P_aver = data_mpc['P_MPC_optimal'][0]

    # Load optimal parameters
    P_optimal = data_model['P_optimal'][0]

    R_tank12 = P_optimal[0]
    m_layer_tank2 = P_optimal[1:5]
    diff_T_c = P_optimal[5]
    T_supply = P_optimal[6]
    m_c_dot = P_optimal[7]

    m_layer_tank1 = np.array([250, 250])
    h_layer_tank1, h_layer_tank2, R_layer_tank1, R_layer_tank2, R_thermal_tank1, R_thermal_tank2 = parameters_tank(2, 4, m_layer_tank1, m_layer_tank2)

    # Load cop_P3 data
    data = loadmat("cop_P3.mat")
    # print(data['cop_3'].shape)
    cop_3 = data['cop_3']


    m1, m2 = m_layer_tank1
    m3, m4, m5, m6 = m_layer_tank2

    R12 = R_thermal_tank1
    R34, R45, R56 = R_thermal_tank2



    u = m.u[tau, 0]
    x_k = [m.x[tau, j] for j in range(10)]

    cop_f = cop_3[0][0] + cop_3[1][0] * x_k[1] + cop_3[2][0] * m.d2[tau] + cop_3[3][0] * x_k[1] * m.d2[tau]
    Q_hp = P_aver * cop_f * 3600

    # x_k_1 equations
    if i == 0:
        T_diff_in_out_tank = x_k[2] - x_k[3]
        return m.x[tau + 1, i] == (x_k[1] + Q_hp / (m_p_dot * cp)) * u + (
                    x_k[0] - (R_out_hp_loss * (x_k[0] - m.d2[tau]) + R_thermal_hp_pipe * (x_k[0] - x_k[1])) * (
                        60 * 60) * dk / (m_out_hp_pipe * cp)) * (1 - u)

    elif i == 1:
        return m.x[tau + 1, i] == (
                    T_diff_he + T_supply * m.d[tau] / m_p_dot + x_k[9] * (1 - m.d[tau] / m_p_dot)) * u + (
                    x_k[1] - (R_in_hp_loss * (x_k[1] - m.d2[tau]) + R_thermal_hp_pipe * (x_k[1] - x_k[0])) * (
                        60 * 60) * dk / (m_in_hp_pipe * cp)) * (1 - u)

    elif i == 2:
        return m.x[tau + 1, i] == ((x_k[1] + Q_hp / (m_p_dot * cp)) - T_diff_he) * u + (x_k[2] - (
                    R_in_tank_loss * (x_k[2] - m.d2[tau]) + R_pipe_upper * (x_k[2] - x_k[4]) + R_pipe_bottom * (
                        x_k[2] - x_k[9])) * (60 * 60) * dk / (m_in_tank_pipe * cp)) * (1 - u)

    elif i == 3:
        return m.x[tau + 1, i] == T_supply * m.d[tau] / m_p_dot + x_k[9] * (1 - m.d[tau] / m_p_dot)

    # Layer tank dynamics
    else:
        eqs = [
            x_k[4] + dk * (m_p_dot * u * cp * (x_k[2] - x_k[4]) - R12 * (x_k[4] - x_k[5]) * (60 * 60) - (
                        m_c_dot - m.d[tau]) * cp * diff_T_c + m.d[tau] * (1 - u) * cp * (x_k[5] - x_k[4])) / (m1 * cp),
            x_k[5] + dk * ((m_p_dot - m.d[tau]) * u * cp * (x_k[4] - x_k[5]) + R12 * (x_k[4] - x_k[5]) * (
                        60 * 60) - R_tank12 * (x_k[5] - x_k[6]) + m.d[tau] * (1 - u) * cp * (x_k[6] - x_k[5])) / (
                        m2 * cp),
            x_k[6] + dk * ((m_p_dot - m.d[tau]) * u * cp * (x_k[5] - x_k[6]) - R34 * (x_k[6] - x_k[7]) * (
                        60 * 60) + R_tank12 * (x_k[5] - x_k[6]) + m.d[tau] * (1 - u) * cp * (x_k[7] - x_k[6])) / (
                        m3 * cp),
            x_k[7] + dk * ((m_p_dot - m.d[tau]) * u * cp * (x_k[6] - x_k[7]) + R34 * (x_k[6] - x_k[7]) * (
                        60 * 60) - R45 * (x_k[7] - x_k[8]) * (60 * 60) + m.d[tau] * (1 - u) * cp * (
                                            x_k[8] - x_k[7])) / (m4 * cp),
            x_k[8] + dk * ((m_p_dot - m.d[tau]) * u * cp * (x_k[7] - x_k[8]) + R45 * (x_k[7] - x_k[8]) * (
                        60 * 60) - R56 * (x_k[8] - x_k[9]) * (60 * 60) + m.d[tau] * (1 - u) * cp * (
                                            x_k[9] - x_k[8])) / (m5 * cp),
            x_k[9] + dk * (
                        (m_p_dot - m.d[tau]) * u * cp * (x_k[8] - x_k[9]) + R56 * (x_k[8] - x_k[9]) * (60 * 60) + m.d[
                    tau] * (1 - u) * cp * (T_supply - x_k[9])) / (m6 * cp)
        ]
        return m.x[tau + 1, i] == eqs[i - 4]


def system_equation(x_k, u, m_s_dot, T_amb, dk, data_mpc, data_model):

    x_k_1 = numpy.zeros(len(x_k))

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
    data = loadmat("cop_P3.mat")
    # print(data['cop_3'].shape)
    cop_3 = data['cop_3']
    cop_f = cop_3[0][0] + cop_3[1][0] * x_k[1] + cop_3[2][0] * T_amb + cop_3[3][0] * x_k[1] * T_amb
    Q_hp = P_aver * cop_f * 3600


    m1, m2 = m_layer_tank1
    m3, m4, m5, m6 = m_layer_tank2

    R12 = R_thermal_tank1
    R34, R45, R56 = R_thermal_tank2


    T_diff_in_out_tank = x_k[2] - x_k[3]
    x_k_1[0] = (x_k[1] + Q_hp / (m_p_dot * cp)) * u + (
                    x_k[0] - (R_out_hp_loss * (x_k[0] - T_amb) + R_thermal_hp_pipe * (x_k[0] - x_k[1])) * (
                        60 * 60) * dk / (m_out_hp_pipe * cp)) * (1 - u)

    x_k_1[1]= (T_diff_he + T_supply * m_s_dot / m_p_dot + x_k[9] * (1 - m_s_dot / m_p_dot)) * u + (
                    x_k[1] - (R_in_hp_loss * (x_k[1] - T_amb) + R_thermal_hp_pipe * (x_k[1] - x_k[0])) * (
                        60 * 60) * dk / (m_in_hp_pipe * cp)) * (1 - u)

    x_k_1[2] = ((x_k[1] + Q_hp / (m_p_dot * cp)) - T_diff_he) * u + (x_k[2] - (
                    R_in_tank_loss * (x_k[2] - T_amb) + R_pipe_upper * (x_k[2] - x_k[4]) + R_pipe_bottom * (
                        x_k[2] - x_k[9])) * (60 * 60) * dk / (m_in_tank_pipe * cp)) * (1 - u)

    x_k_1[3] = T_supply * m_s_dot / m_p_dot + x_k[9] * (1 - m_s_dot / m_p_dot)

    # Layer tank dynamics
    x_k_1[4]=x_k[4] + dk * (m_p_dot * u * cp * (x_k[2] - x_k[4]) - R12 * (x_k[4] - x_k[5]) * (60 * 60) - (
                        m_c_dot - m_s_dot) * cp * diff_T_c + m_s_dot * (1 - u) * cp * (x_k[5] - x_k[4])) / (m1 * cp)
    x_k_1[5]=x_k[5] + dk * ((m_p_dot - m_s_dot) * u * cp * (x_k[4] - x_k[5]) + R12 * (x_k[4] - x_k[5]) * (
                        60 * 60) - R_tank12 * (x_k[5] - x_k[6]) + m_s_dot * (1 - u) * cp * (x_k[6] - x_k[5])) / (
                        m2 * cp)
    x_k_1[6]=x_k[6] + dk * ((m_p_dot - m_s_dot) * u * cp * (x_k[5] - x_k[6]) - R34 * (x_k[6] - x_k[7]) * (
                        60 * 60) + R_tank12 * (x_k[5] - x_k[6]) + m_s_dot * (1 - u) * cp * (x_k[7] - x_k[6])) / (
                        m3 * cp)
    x_k_1[7]=x_k[7] + dk * ((m_p_dot - m_s_dot) * u * cp * (x_k[6] - x_k[7]) + R34 * (x_k[6] - x_k[7]) * (
                        60 * 60) - R45 * (x_k[7] - x_k[8]) * (60 * 60) + m_s_dot * (1 - u) * cp * (
                                            x_k[8] - x_k[7])) / (m4 * cp)
    x_k_1[8]=x_k[8] + dk * ((m_p_dot - m_s_dot) * u * cp * (x_k[7] - x_k[8]) + R45 * (x_k[7] - x_k[8]) * (
                        60 * 60) - R56 * (x_k[8] - x_k[9]) * (60 * 60) + m_s_dot * (1 - u) * cp * (
                                            x_k[9] - x_k[8])) / (m5 * cp)
    x_k_1[9]=x_k[9] + dk * (
                        (m_p_dot - m_s_dot) * u * cp * (x_k[8] - x_k[9]) + R56 * (x_k[8] - x_k[9]) * (60 * 60) + m_s_dot * (1 - u) * cp * (T_supply - x_k[9])) / (m6 * cp)

    return x_k_1

