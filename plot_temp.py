import matplotlib.pyplot as plt
import numpy as np
# from datetime import datetime, timedelta
from matplotlib.dates import DateFormatter
# from matplotlib import rc



def plot_show_tanktemp(Temp_upper_water_tank1_interp, Temp_upper_water_tank2_interp, Temp_bottom_water_tank2_interp, Xsim,Nsim):
    # Enable LaTeX formatting for text elements
    plt.rcParams['text.usetex'] = True

    err1 = np.sqrt(np.mean((Xsim[4, :-1] - Temp_upper_water_tank1_interp['value'][:Nsim]) ** 2))  # Root Mean Squared Error
    err2 = np.sqrt(np.mean((Xsim[6, :-1] - Temp_upper_water_tank2_interp['value'][:Nsim]) ** 2))  # Root Mean Squared Error
    err3 = np.sqrt(np.mean((Xsim[9, :-1] - Temp_bottom_water_tank2_interp['value'][:Nsim]) ** 2))  # Root Mean Squared Error

    norm_Xsim1 = np.linalg.norm(Xsim[4, :-1] - Temp_upper_water_tank1_interp['value'][:Nsim], 2)
    norm_Temp_upper_water_tank1_interp = np.linalg.norm(Temp_upper_water_tank1_interp['value'][:Nsim], 2)
    vaf1 = max(0, (1 - (norm_Xsim1 ** 2) / (norm_Temp_upper_water_tank1_interp ** 2)))  # Variance Accounted For

    norm_Xsim2 = np.linalg.norm(Xsim[6, :-1] - Temp_upper_water_tank2_interp['value'][:Nsim], 2)
    norm_Temp_upper_water_tank2_interp = np.linalg.norm(Temp_upper_water_tank2_interp['value'][:Nsim], 2)
    vaf2 = max(0, (1 - (norm_Xsim2 ** 2) / (norm_Temp_upper_water_tank2_interp ** 2)))  # Variance Accounted For

    norm_Xsim3 = np.linalg.norm(Xsim[9, :-1] - Temp_bottom_water_tank2_interp['value'][:Nsim], 2)
    norm_Temp_bottom_water_tank2_interp = np.linalg.norm(Temp_bottom_water_tank2_interp['value'][:Nsim], 2)
    vaf3 = max(0, (1 - (norm_Xsim3 ** 2) / (norm_Temp_bottom_water_tank2_interp ** 2)))  # Variance Accounted For


    # Create subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 15), sharex=True)

    # Plot the data
    ax1.plot(Temp_upper_water_tank1_interp.index[:Nsim], Xsim[4, :-1], label=r'$T_{1}$', color='blue')
    ax1.plot(Temp_upper_water_tank1_interp.index[:Nsim], Temp_upper_water_tank1_interp['value'][:Nsim], label='Ground Truth',
             color='red', linestyle='dashed')
    # ax1.set_xlabel('Time')
    str1 = f'Variance accounted for (VAF): {vaf1 * 100:.2f}\% \n Root Mean Squared Error: {err1:.2f}°C'
    ax1.set_title(str1)
    ax1.set_ylabel('Value')
    ax1.legend()
    ax1.grid()

    ax2.plot(Temp_upper_water_tank2_interp.index[:Nsim], Xsim[6, :-1], label=r'$T_{3}$', color='blue')
    ax2.plot(Temp_upper_water_tank2_interp.index[:Nsim], Temp_upper_water_tank2_interp['value'][:Nsim], label='Ground Truth',
             color='red', linestyle='dashed')
    # ax2.set_xlabel('Time')
    str2 = f'Variance accounted for (VAF): {vaf2 * 100:.2f}\% \n Root Mean Squared Error: {err2:.2f}°C'
    ax2.set_title(str2)
    ax2.set_ylabel('Value')
    ax2.legend()
    ax2.grid()

    ax3.plot(Temp_bottom_water_tank2_interp.index[:Nsim], Xsim[9, :-1], label=r'$T_{6}$', color='blue')
    ax3.plot(Temp_bottom_water_tank2_interp.index[:Nsim], Temp_bottom_water_tank2_interp['value'][:Nsim], label='Ground Truth',
             color='red', linestyle='dashed')
    str3 = f'Variance accounted for (VAF): {vaf3 * 100:.2f}\% \n Root Mean Squared Error: {err3:.2f}°C'
    ax3.set_title(str3)
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Value')
    ax3.legend()
    ax3.grid()

    minute_formatter = DateFormatter('%Y-%m-%d %H:%M')
    ax3.xaxis.set_major_formatter(minute_formatter)


    plt.xticks(rotation=45)
    # plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.suptitle('Comparison of Temperatures of Tanks', fontsize=16, fontweight='bold')
    # suptitle.set_y(1.02)
    plt.subplots_adjust(top=0.93)
    # plt.suptitle('Comparison of Tempertures of Tanks')

    # Show the plot
    plt.show()

def plot_show_pipetemp(Temp_out_water_hp_interp,Temp_in_water_hp_interp,Temp_out_water_he_interp, Xsim,Nsim):
    # Enable LaTeX formatting for text elements
    plt.rcParams['text.usetex'] = True

    err1 = np.sqrt(np.mean((Xsim[0, :-1] - Temp_out_water_hp_interp['value'][:Nsim]) ** 2))  # Root Mean Squared Error
    err2 = np.sqrt(np.mean((Xsim[1, :-1] - Temp_in_water_hp_interp['value'][:Nsim]) ** 2))  # Root Mean Squared Error
    err3 = np.sqrt(np.mean((Xsim[2, :-1] - Temp_out_water_he_interp['value'][:Nsim]) ** 2))  # Root Mean Squared Error

    norm_Xsim1 = np.linalg.norm(Xsim[0, :-1] - Temp_out_water_hp_interp['value'][:Nsim], 2)
    norm_Temp_out_water_hp = np.linalg.norm(Temp_out_water_hp_interp['value'][:Nsim], 2)
    vaf1 = max(0, (1 - (norm_Xsim1 ** 2) / (norm_Temp_out_water_hp ** 2)))  # Variance Accounted For

    norm_Xsim2 = np.linalg.norm(Xsim[1, :-1] - Temp_in_water_hp_interp['value'][:Nsim], 2)
    norm_Temp_in_water_hp = np.linalg.norm(Temp_in_water_hp_interp['value'][:Nsim], 2)
    vaf2 = max(0, (1 - (norm_Xsim2 ** 2) / (norm_Temp_in_water_hp ** 2)))  # Variance Accounted For

    norm_Xsim3 = np.linalg.norm(Xsim[2, :-1] - Temp_out_water_he_interp['value'][:Nsim], 2)
    norm_Temp_out_water_he = np.linalg.norm(Temp_out_water_he_interp['value'][:Nsim], 2)
    vaf3 = max(0, (1 - (norm_Xsim3 ** 2) / (norm_Temp_out_water_he ** 2)))  # Variance Accounted For

    # Create subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 15), sharex=True)

    # Plot the data
    ax1.plot(Temp_out_water_hp_interp.index[:Nsim], Xsim[0, :-1], label=r'$T_{out,hp}$', color='blue')
    ax1.plot(Temp_out_water_hp_interp.index[:Nsim], Temp_out_water_hp_interp['value'][:Nsim], label='Ground Truth',
             color='red', linestyle='dashed')
    # ax1.set_xlabel('Time')
    str1 = f'Variance accounted for (VAF): {vaf1 * 100:.2f}\% \n Root Mean Squared Error: {err1:.2f}°C'
    ax1.set_title(str1)
    ax1.set_ylabel('Value')
    ax1.legend()
    ax1.grid()

    ax2.plot(Temp_in_water_hp_interp.index[:Nsim], Xsim[1, :-1], label=r'$T_{in,hp}$', color='blue')
    ax2.plot(Temp_in_water_hp_interp.index[:Nsim], Temp_in_water_hp_interp['value'][:Nsim], label='Ground Truth',
             color='red', linestyle='dashed')
    # ax2.set_xlabel('Time')
    str2 = f'Variance accounted for (VAF): {vaf2 * 100:.2f}\% \n Root Mean Squared Error: {err2:.2f}°C'
    ax2.set_title(str2)
    ax2.set_ylabel('Value')
    ax2.legend()
    ax2.grid()

    ax3.plot(Temp_out_water_he_interp.index[:Nsim], Xsim[2, :-1], label=r'$T_{in,tank}$', color='blue')
    ax3.plot(Temp_out_water_he_interp.index[:Nsim], Temp_out_water_he_interp['value'][:Nsim], label='Ground Truth',
             color='red', linestyle='dashed')
    str3 = f'Variance accounted for (VAF): {vaf3 * 100:.2f}\% \n Root Mean Squared Error: {err3:.2f}°C'
    ax3.set_title(str3)
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Value')
    ax3.legend()
    ax3.grid()

    minute_formatter = DateFormatter('%Y-%m-%d %H:%M')
    ax3.xaxis.set_major_formatter(minute_formatter)

    plt.xticks(rotation=45)
    # plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.suptitle('Comparison of Temperatures of Pipes', fontsize=16, fontweight='bold')
    # suptitle.set_y(1.02)
    plt.subplots_adjust(top=0.93)
    # plt.suptitle('Comparison of Tempertures of Tanks')

    # Show the plot
    plt.show()