import matplotlib.pyplot as plt
import matplotlib as mpl

# use latex for rendering text
# mpl.rcParams['text.usetex'] = True
# mpl.rcParams['font.family'] = 'serif'
# mpl.rcParams['font.serif'] = 'Computer Modern'
import numpy as np
import sys, matplotlib as mpl
color_dict = {
    'measured_PL': 'tab:blue',
    'measured_PV': 'sandybrown',
    'measured_Netload': 'dodgerblue',
    'forecasted_PL': 'tab:blue',
    'forecasted_PV': 'sandybrown',
    'forecasted_Netload': 'dodgerblue',
    'soc': 'violet',
    "Pnet_DA": 'cyan',
    "negPnet_DA": 'tab:blue',
    'posPnet_DA': 'tab:purple',
    'Pgreen_DA': 'forestgreen',
    'Pnongreen_DA': 'tomato',
    'daily_nongreen_status': 'black',
    'Pchg': 'mediumaquamarine',
    'Pdchg': 'orangered',
    'Pgreen_UGT1_DA': 'darkgreen',
    'Pgreen_UGT2_DA': 'mediumseagreen',
    'Pgreen_UGT3_DA': 'yellowgreen',
    'Remained_UGT1_quota': 'darkgreen',# but in dashh line
    'Remained_UGT2_quota': 'mediumseagreen',# but in dashh line
    'Remained_UGT3_quota': 'yellowgreen',# but in dashh line
    
}

# Map columns name with legend
legend_name = {
    'measured_PL': 'Load consumption [kW]',
    'measured_PV': 'PV generation [kW]',
    'measured_Netload': 'Net load [kW]',
    'forecasted_PL': 'Load consumption [kW]',
    'forecasted_PV': 'PV generation [kW]',
    'forecasted_Netload': 'Net load [kW]',
    'soc': r'SoC [%]',
    'Pnet_DA': 'Net power [kW]',
    'posPnet_DA': 'Imported power from grid [kW]',
    'negPnet_DA': 'Exported power [kW]',
    'Pgreen_DA': 'Green [kW]',
    'Pnongreen_DA': 'Non-green [kW]',
    'daily_nongreen_status': 'Non-green',
    'Pchg': 'Charge [kW]',
    'Pdchg': 'Discharge [kW]',
    'Pgreen_UGT1_DA': 'Import seller 1 [kW]',
    'Pgreen_UGT2_DA': 'Import seller 2 [kW]',
    'Pgreen_UGT3_DA': 'Import seller 3 [kW]',
    'x_green_UGT1_DA': 'Status UGT1 [-]',
    'x_green_UGT2_DA': 'Status UGT2 [-]',
    'x_green_UGT3_DA': 'Status UGT3 [-]',}
legend_name_nosell = legend_name.copy()
# Curtailed power for 'negPnet_DA' in nosell case
legend_name_nosell['negPnet_DA'] = 'Curtailed power [kW]'

# def plot_one_batch_one_green_operation(
#         result_df,
#         mode='deterministic',
#         colors_list=None,
#         figsize=(12, 6),
#         tick_fontsize=14,
#         label_fontsize=16,
#         legend_fontsize=14,
#         title_fontsize=16):
#     """
#     Plot one optimization batch (power flows, SOC, and operation status).
#     Shows x-tick labels on both subplots.
#     """

#     # Determine which columns to use
#     if mode == 'prescient':
#         y1st_1_cols = ['measured_Netload']
#     elif mode == 'deterministic':
#         y1st_1_cols = ['forecasted_Netload']
#     else:
#         raise ValueError("mode must be 'prescient' or 'deterministic'")
#     y1st_1_cols.extend(['Pchg', 'Pdchg'])
#     y2nd_1_cols = ['soc']
#     y1st_2_cols = ['Pgreen_DA', 'Pnongreen_DA', 'negPnet_DA']
#     y2nd_2_cols = ['daily_nongreen_status']

    

#     # Create two subplots sharing the x-axis
#     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

#     # First subplot (Power + SOC)
#     ax1_1 = ax1
#     ax1_2 = ax1.twinx()
#     for col in y1st_1_cols:
#         ax1_1.plot(result_df[col], label=legend_name[col], color=color_dict[col])
#     for col in y2nd_1_cols: 
#         ax1_2.plot(result_df[col], label=legend_name[col], color=color_dict[col])

#     ax1_1.set_ylabel('Power [kW]', fontsize=label_fontsize)
#     ax1_2.set_ylabel(r'SoC [%]', fontsize=label_fontsize)
#     ax1_1.tick_params(axis='y', labelsize=tick_fontsize)
#     ax1_2.tick_params(axis='y', labelsize=tick_fontsize)

#     # Set y-axis limits for SOC with, and create dash line for soc = 20% and 80% and 40%
#     ax1_2.set_ylim(result_df['soc'].min(), result_df['soc'].max())
#     ax1_2.axhline(20, color='gray', linestyle='--', linewidth=0.8)
#     ax1_2.axhline(80, color='gray', linestyle='--', linewidth=0.8)
#     # ax1_2.axhline(40, color='gray', linestyle='--', linewidth=0.8)

   

#     # Second subplot (dispatch + status)
#     ax2_1 = ax2
#     ax2_2 = ax2.twinx()
#     for col in y1st_2_cols: # Plot postPnet as dash line
#         if col == 'negPnet_DA':
#             ax2_1.plot(result_df.index, result_df[col], label=legend_name[col], color=color_dict[col], linestyle='--')
#         else:
#             ax2_1.plot(result_df.index, result_df[col], label=legend_name[col], color=color_dict[col])
#     for col in y2nd_2_cols:
#         ax2_2.plot(result_df.index, result_df[col], label=legend_name[col], color=color_dict[col])

#     ax2_1.set_ylabel('Power [kW]', fontsize=label_fontsize)
#     ax2_2.set_ylabel('Daily status', fontsize=label_fontsize)
#     ax2_1.tick_params(axis='y', labelsize=tick_fontsize)
#     ax2_2.tick_params(axis='y', labelsize=tick_fontsize)
#      # Show ytick at only 0,1 for status plot
#     ax2_2.set_yticks([0, 1])

#     # Build custom xticks (every 12 hrs from 15-min interval)
#     x_ticks_index = result_df.index[::48]
#     noon_time_index = [48+i*96 for i in range(len(result_df)//96)]
#     labels = []
#     for x in x_ticks_index:
#         dt = result_df['Datetime'].iloc[x]
#         if x == 0:  # add day label for the first tick
#             labels.append(f"\n{dt.strftime('%d%b%y')}")
#         elif x in noon_time_index:  # every 96 steps = 24h
#             labels.append(dt.strftime('%H%p'))
#         else:
#             labels.append('')

#     # Show legends for both y-axes, get_legend_handles_labels are for combining legends from both y-axes
#     lines_1, labels_1 = ax1_1.get_legend_handles_labels()
#     lines_2, labels_2 = ax1_2.get_legend_handles_labels()
#     ax1_1.legend(lines_1 + lines_2, labels_1 + labels_2,
#                  fontsize=legend_fontsize,
#                  loc='upper center',
#                  bbox_to_anchor=(0.5, -0.01),
#                  ncol=2,
#                  frameon=False)

#     lines_3, labels_3 = ax2_1.get_legend_handles_labels()
#     lines_4, labels_4 = ax2_2.get_legend_handles_labels()
#     ax2_1.legend(lines_3 + lines_4, labels_3 + labels_4,
#                  fontsize=legend_fontsize,
#                  loc='upper center',
#                  bbox_to_anchor=(0.5, -0.2),
#                  ncol=2,
#                  frameon=False)

#     # only set the x-ticks on the bottom subplot
#     ax2_1.set_xticks(x_ticks_index)
#     ax2_1.set_xticklabels(labels, fontsize=tick_fontsize)
#     for label in ax2_1.get_xticklabels():
#         label.set_ha('center')
#         label.set_va('top')
#         label.set_linespacing(1.5)
#     ax2_1.tick_params(axis='x', labelsize=tick_fontsize)
#     ax2_1.grid(visible=True, which='both', axis='x', linestyle='--', linewidth=0.5)

#     # but enable the grid on the top subplot as well
#     ax1_1.grid(visible=True, which='both', axis='x', linestyle='--', linewidth=0.5)


#     # Reduce thickness of boundary lines
#     for axis in [ax1_1, ax1_2, ax2_1, ax2_2]:

#         for spine in axis.spines.values():
#             spine.set_linewidth(0.1)
        

#     # Optionally tighten layout
#     fig.tight_layout()

#     return fig
def plot_one_batch_one_green_operation(
        result_df,
        mode='deterministic',
        colors_list=None,
        figsize=(12, 6),
        tick_fontsize=14,
        label_fontsize=16,
        legend_fontsize=14,
        title_fontsize=16, 
        case="sell"):
    """
    Plot one optimization batch (power flows, SOC, and operation status).
    Shows x-tick labels on both subplots.
    """

    # Determine which columns to use
    if mode == 'prescient':
        y1st_1_cols = ['measured_Netload']
    elif mode == 'deterministic':
        y1st_1_cols = ['forecasted_Netload']
    else:
        raise ValueError("mode must be 'prescient' or 'deterministic'")
    y1st_1_cols.extend(['Pchg', 'Pdchg'])
    y2nd_1_cols = ['soc']
    y1st_2_cols = ['Pgreen_DA', 'Pnongreen_DA', 'negPnet_DA']
    y2nd_2_cols = ['daily_nongreen_status']

    

    # Create two subplots sharing the x-axis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True, constrained_layout=True)

    # First subplot (Power + SOC)
    ax1_1 = ax1
    ax1_2 = ax1.twinx()
    if case == "nosell":
        legend_name_use = legend_name_nosell
    else:
        legend_name_use = legend_name

    for col in y1st_1_cols:
        ax1_1.plot(result_df[col], label=legend_name_use[col], color=color_dict[col], linewidth=0.8)
    for col in y2nd_1_cols: 
        ax1_2.plot(result_df[col], label=legend_name_use[col], color=color_dict[col], linewidth=0.8)

    ax1_1.set_ylabel('Power [kW]', fontsize=label_fontsize)
    ax1_2.set_ylabel(r'SoC [%]', fontsize=label_fontsize)
    ax1_1.tick_params(axis='y', labelsize=tick_fontsize)
    ax1_2.tick_params(axis='y', labelsize=tick_fontsize)

    # Set y-axis limits for SOC with, and create dash line for soc = 20% and 80% and 40%
    ax1_2.set_ylim(result_df['soc'].min(), result_df['soc'].max())
    ax1_2.axhline(20, color='gray', linestyle='--', linewidth=0.8)
    ax1_2.axhline(80, color='gray', linestyle='--', linewidth=0.8)
    # ax1_2.axhline(40, color='gray', linestyle='--', linewidth=0.8)

   

    # Second subplot (dispatch + status)
    ax2_1 = ax2
    for col in y1st_2_cols: # Plot postPnet as dash line
        if col == 'negPnet_DA':
            ax2_1.plot(result_df.index, result_df[col], label=legend_name_use[col], color=color_dict[col], linestyle='--', linewidth=0.8)
        else:
            ax2_1.plot(result_df.index, result_df[col], label=legend_name_use[col], color=color_dict[col], linewidth=0.8)

    # fill the background color for both non-green and green areas
    green_status = result_df['daily_green_status'] if 'daily_green_status' in result_df.columns else 1 - result_df['daily_nongreen_status']
    
    # fill green
    ax1_1.fill_between(result_df.index, 
                       y1 = result_df[y1st_1_cols].min().min(),
                       y2 = result_df[y1st_1_cols].max().max(),
                       where=(green_status ==1),
                       transform=ax1_1.get_xaxis_transform(),
                       interpolate=True,
                          color='lightgreen',
                            alpha=0.15,
                            label='_nolegend_')
    ax2_1.fill_between(result_df.index, 
                       y1 = result_df[y1st_2_cols].min().min(),
                       y2 = result_df[y1st_2_cols].max().max(),
                       where=(green_status ==1),
                       transform=ax2_1.get_xaxis_transform(),
                       interpolate=True,
                          color='lightgreen',
                            alpha=0.15,
                            label='_nolegend_')
    # fill non-green
    ax1_1.fill_between(result_df.index,
                          y1 = result_df[y1st_1_cols].min().min(),
                          y2 = result_df[y1st_1_cols].max().max(),
                          where=(green_status ==0),
                          transform=ax1_1.get_xaxis_transform(),
                          interpolate=True,
                              color='lightcoral',
                             alpha=0.15,
                             label='_nolegend_')
    ax2_1.fill_between(result_df.index,
                          y1 = result_df[y1st_2_cols].min().min(),
                          y2 = result_df[y1st_2_cols].max().max(),
                          where=(green_status ==0),
                          transform=ax2_1.get_xaxis_transform(),
                          interpolate=True,
                              color='lightcoral',
                             alpha=0.15,
                             label='_nolegend_')

    ax2_1.set_ylabel('Power [kW]', fontsize=label_fontsize)
    ax2_1.tick_params(axis='y', labelsize=tick_fontsize)


    # Build custom xticks (every 12 hrs from 15-min interval)
    x_ticks_index = result_df.index[::48]
    noon_time_index = [48+i*96 for i in range(len(result_df)//96)]
    labels = []
    for x in x_ticks_index:
        dt = result_df['Datetime'].iloc[x]
        if x == 0:  # add day label for the first tick
            labels.append(f"\n{dt.strftime('%d%b%y')}")
        elif x in noon_time_index:  # every 96 steps = 24h
            labels.append(dt.strftime('%H%p'))
        else:
            labels.append('')

    # Show legends for both y-axes, get_legend_handles_labels are for combining legends from both y-axes
    lines_1, labels_1 = ax1_1.get_legend_handles_labels()
    lines_2, labels_2 = ax1_2.get_legend_handles_labels()
    ax1_1.legend(lines_1 + lines_2, labels_1 + labels_2,
                 fontsize=legend_fontsize,
                 loc='upper center',
                 bbox_to_anchor=(0.5, -0.01),
                 ncol=2,
                 frameon=False)

    lines_3, labels_3 = ax2_1.get_legend_handles_labels()
    ax2_1.legend(lines_3, labels_3,
                fontsize=legend_fontsize,
                loc='upper center',
                bbox_to_anchor=(0.5, -0.25),
                ncol=2,
                frameon=False)

    # only set the x-ticks on the bottom subplot
    ax2_1.set_xticks(x_ticks_index)
    ax2_1.set_xticklabels(labels, fontsize=tick_fontsize)
    for label in ax2_1.get_xticklabels():
        label.set_ha('center')
        label.set_va('top')
        label.set_linespacing(1.5)
    ax2_1.tick_params(axis='x', labelsize=tick_fontsize)
    ax2_1.grid(visible=True, which='both', axis='x', linestyle='--', linewidth=0.5)

    # but enable the grid on the top subplot as well
    ax1_1.grid(visible=True, which='both', axis='x', linestyle='--', linewidth=0.5)


    # Reduce thickness of boundary lines
    for axis in [ax1_1, ax1_2, ax2_1]:

        for spine in axis.spines.values():
            spine.set_linewidth(0.1)
            spine.set_alpha(0.5)
            spine.set_color('gray')
        

    return fig
def plot_one_batch_three_green_operations(
        result_df,

        mode='deterministic',
        figsize=(12, 6),
        tick_fontsize=14,
        label_fontsize=16,
        legend_fontsize=14,
        resolution_in_hour=15/60,
        case="sell",
        green_allocation_quota = [20, 30, 300] # in kWh
 ):
    # Determine which columns to use
    if mode == 'prescient':
        y1st_1_cols = ['measured_Netload']
    elif mode == 'deterministic':
        y1st_1_cols = ['forecasted_Netload']
    else:
        raise ValueError("mode must be 'prescient' or 'deterministic'")
    y1st_1_cols.extend(['Pchg', 'Pdchg'])
    y2nd_1_cols = ['soc']
    if case == "nosell":
        legend_name_use = legend_name_nosell
    else:
        legend_name_use = legend_name
    # plot only the bottom subplot with three green operations
    fig,(ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    # First subplot (Power + SOC)
    ax1_1 = ax1
    ax1_2 = ax1.twinx()
    for col in y1st_1_cols:
        ax1_1.plot(result_df[col], label=legend_name_use[col], color=color_dict[col])
    for col in y2nd_1_cols: 
        ax1_2.plot(result_df[col], label=legend_name_use[col], color=color_dict[col])
    ax1_1.set_ylabel('Power [kW]', fontsize=label_fontsize)
    ax1_2.set_ylabel(r'SoC [%]', fontsize=label_fontsize)
    ax1_1.tick_params(axis='y', labelsize=tick_fontsize)
    ax1_2.tick_params(axis='y', labelsize=tick_fontsize)
    # Set y-axis limits for SOC with, and create dash line for soc = 20% and 80% and 40%
    ax1_2.set_ylim(result_df['soc'].min(), result_df['soc'].max())
    ax1_2.axhline(20, color='gray', linestyle='--', linewidth=0.8)
    ax1_2.axhline(80, color='gray', linestyle='--', linewidth=0.8)
    # Second subplot (dispatch + status)

    ax2_1 = ax2
    ax2_2 = ax2.twinx()
    y1st_2_cols = ['Pgreen_UGT1_DA', 'Pgreen_UGT2_DA', 'Pgreen_UGT3_DA'] + ['Pnongreen_DA'] + ['negPnet_DA']
    # y2nd_2_cols = ['daily_nongreen_status']
    for col in y1st_2_cols: # Plot postPnet as dash line
        if col == 'posPnet_DA':
            ax2_1.plot(result_df.index, result_df[col], label=legend_name_use[col], color=color_dict[col], linestyle='--')
        if col == 'negPnet_DA':
            ax2_1.plot(result_df.index, result_df[col], label=legend_name_use[col], color=color_dict[col], linestyle='--')
        else:
            ax2_1.plot(result_df.index, result_df[col], label=legend_name_use[col], color=color_dict[col])
    # for col in y2nd_2_cols:
    #     ax2_2.plot(result_df.index, result_df[col], label=legend_name[col], color=color_dict.get(col))
    used_seller1 = result_df['Pgreen_UGT1_DA'].cumsum() * resolution_in_hour
    remained_seller1 = green_allocation_quota[0] - used_seller1
    
    used_seller2 = result_df['Pgreen_UGT2_DA'].cumsum() * resolution_in_hour
    remained_seller2 = green_allocation_quota[1] - used_seller2
    
    used_seller3 = result_df['Pgreen_UGT3_DA'].cumsum() * resolution_in_hour
    remained_seller3 = green_allocation_quota[2] - used_seller3
    
    ax2_2.plot(remained_seller1, label='Remained seller 1 [kWh]', color=color_dict['Pgreen_UGT1_DA'], linestyle='--')
    ax2_2.plot(remained_seller2, label='Remained seller 2 [kWh]', color=color_dict['Pgreen_UGT2_DA'], linestyle='--')
    ax2_2.plot(remained_seller3, label='Remained seller 3 [kWh]', color=color_dict['Pgreen_UGT3_DA'], linestyle='--')
    
    ax2_1.set_ylabel('Power [kW]', fontsize=label_fontsize)
    ax2_2.set_ylabel('Energy [kWh]', fontsize=label_fontsize)
    ax2_1.tick_params(axis='y', labelsize=tick_fontsize)
    ax2_2.tick_params(axis='y', labelsize=tick_fontsize)
    
    ax2_1.set_ylim(0, max(result_df[y1st_2_cols].max()))
    # set y-range of remained quota plot to [0, min+10]
    ax2_2.set_ylim(0, min(green_allocation_quota)+10)
    
    # Build custom xticks (every 12 hrs from 15-min interval)
    x_ticks_index = result_df.index[::48]
    noontime_index = [48+i*96 for i in range(len(result_df)//96)]
    labels = []
    for i, x in enumerate(x_ticks_index):
        dt = result_df['Datetime'].iloc[x]
        if i==0:  # add day label for the first tick
            labels.append(f"\n{dt.strftime('%d%b%y')}")
        elif x in noontime_index:  # every 96 steps = 24h
            labels.append(dt.strftime('%H%p'))
        else:
            labels.append('')
    # Apply xticks and labels to both subplots
    # Show legends for both y-axes, get_legend_handles_labels are for combining legends from both y-axes
    lines_1, labels_1 = ax1_1.get_legend_handles_labels()
    lines_2, labels_2 = ax1_2.get_legend_handles_labels()
    ax1_1.legend(lines_1 + lines_2, labels_1 + labels_2,
                 fontsize=legend_fontsize,
                 loc='upper center',
                 bbox_to_anchor=(0.5, -0.1),
                 ncol=2,
                 frameon=False)
    # --- Get handles ---
    lines_3, labels_3 = ax2_1.get_legend_handles_labels()  # Import + Non-green
    lines_4, labels_4 = ax2_2.get_legend_handles_labels()  # Remained

    import_lines = lines_3[0:3]
    import_labels = labels_3[0:3]

    # Remained (3)
    remain_lines = lines_4[0:3]
    remain_labels = labels_4[0:3]

    # Non-green (1)
    nongreen_line = lines_3[3]
    nongreen_label = labels_3[3]
    
    # Exported (1)
    export_line = lines_3[4]
    export_label = labels_3[4]

    # --- Create invisible placeholders to lock grid shape ---
    from matplotlib.lines import Line2D

    empty_line = Line2D([], [], color='none', label='')

    # --- Column-wise assembly (3×3 grid) ---
    final_lines = [
        import_lines[0], remain_lines[0], nongreen_line,
        import_lines[1], remain_lines[1], export_line,
        import_lines[2], remain_lines[2], empty_line,
    ]

    final_labels = [
        import_labels[0], remain_labels[0], nongreen_label,
        import_labels[1], remain_labels[1], export_label,
        import_labels[2], remain_labels[2], '',
    ]

    # --- Draw legend ---
    ax2_1.legend(
        final_lines,
        final_labels,
        fontsize=legend_fontsize,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.40),
        ncol=3,
        columnspacing=0.3,
        handletextpad=0.8,
        frameon=False,
    )

    # only set the x-ticks on the bottom subplot
    ax2_1.set_xticks(x_ticks_index)
    ax2_1.set_xticklabels(labels, fontsize=tick_fontsize)
    for label in ax2_1.get_xticklabels():
        label.set_ha('center')
        label.set_va('top')
        label.set_linespacing(1.5)
    ax2_1.tick_params(axis='x', labelsize=tick_fontsize)
    ax2_1.grid(visible=True, which='both', axis='x', linestyle='--', linewidth=0.5)

    # but enable the grid on the top subplot as well
    for label in ax1_1.get_xticklabels():
        label.set_visible(False)
    ax1_1.grid(visible=True, which='both', axis='x', linestyle='--', linewidth=0.5)
    # fill the background color for both non-green and green areas
    green_status = result_df['daily_green_status'] if 'daily_green_status' in result_df.columns else 1 - result_df['daily_nongreen_status']
    # fill green
    ax1_1.fill_between(result_df.index,
                          y1 = result_df[y1st_1_cols].min().min(),
                          y2 = result_df[y1st_1_cols].max().max(),
                          where=(green_status ==1),
                          transform=ax1_1.get_xaxis_transform(),
                          interpolate=True,
                              color='lightgreen',
                             alpha=0.15,
                             label='_nolegend_')
    ax2_1.fill_between(result_df.index,
                            y1 = result_df[y1st_2_cols].min().min(),
                            y2 = result_df[y1st_2_cols].max().max(),
                            where=(green_status ==1),
                            transform=ax2_1.get_xaxis_transform(),
                            interpolate=True,
                                color='lightgreen',
                                 alpha=0.15,
                                 label='_nolegend_')
    # fill non-green
    ax1_1.fill_between(result_df.index,
                            y1 = result_df[y1st_1_cols].min().min(),
                            y2 = result_df[y1st_1_cols].max().max(),
                            where=(green_status ==0),
                            transform=ax1_1.get_xaxis_transform(),
                            interpolate=True,
                                color='lightcoral',
                                 alpha=0.15,
                                 label='_nolegend_')
    ax2_1.fill_between(result_df.index,
                            y1 = result_df[y1st_2_cols].min().min(),
                            y2 = result_df[y1st_2_cols].max().max(),
                            where=(green_status ==0),
                            transform=ax2_1.get_xaxis_transform(),
                            interpolate=True,
                                color='lightcoral',
                                 alpha=0.15,
                                 label='_nolegend_')

    # Reduce thickness of boundary lines
    for axis in [ax1_1, ax1_2, ax2_1, ax2_2]:
        for spine in axis.spines.values():
            spine.set_linewidth(0.1)
            spine.set_alpha(0.5)
            spine.set_color('gray')

    # Optionally tighten layout

    fig.tight_layout()
    return fig
def plot_one_batch_three_green_with_remained_quota(
        result_df,
        green_allocation_quota = [20, 30, 300], # in kWh
        resolution_in_hour=15/60,  # in hours
        mode='deterministic',
        colors_list=None,
        figsize=(12, 6),
        tick_fontsize=14,
        label_fontsize=16,
        legend_fontsize=14,
        title_fontsize=16):
    # Create 2x1 two y axis subplots, the first one is the same as the bottom of the first function, the second one is the remained quota
    fig, (ax1, ax2) = plt.subplots(2, 1,
                                   figsize=figsize,
                                   sharex=True)
    # First subplot (Power + SOC)
    ax1_1 = ax1
    ax1_2 = ax1.twinx()
    y1st_1_cols = ['Pnongreen_DA', 'Pgreen_UGT1_DA', 'Pgreen_UGT2_DA', 'Pgreen_UGT3_DA']    
    y2nd_1_cols = ['daily_nongreen_status']
    for col in y1st_1_cols:
        ax1_1.plot(result_df[col], label=legend_name[col], color=color_dict[col])
    for col in y2nd_1_cols: 
        ax1_2.plot(result_df[col], label=legend_name[col], color=color_dict[col])
    ax1_1.set_ylabel('Power [kW]', fontsize=label_fontsize)
    ax1_2.set_ylabel('Daily status', fontsize=label_fontsize)
    ax1_1.tick_params(axis='y', labelsize=tick_fontsize)
    ax1_2.tick_params(axis='y', labelsize=tick_fontsize)
    # Show ytick at only 0,1 for status plot
    ax1_2.set_yticks([0, 1])
    # Second subplot (remained quota)
    ax2_1 = ax2
    ax2_2 = ax2.twinx()
    used_UGT1 = result_df['Pgreen_UGT1_DA'].cumsum() * resolution_in_hour
    remained_UGT1 = green_allocation_quota[0] - used_UGT1
    used_UGT2 = result_df['Pgreen_UGT2_DA'].cumsum() * resolution_in_hour
    remained_UGT2 = green_allocation_quota[1] - used_UGT2
    used_UGT3 = result_df['Pgreen_UGT3_DA'].cumsum() * resolution_in_hour
    remained_UGT3 = green_allocation_quota[2] - used_UGT3
    for col in y1st_1_cols[1:]: # skip Pnongreen_DA
        ax2_1.plot(result_df[col], label=legend_name[col], color=color_dict[col])
    ax2_2.plot(remained_UGT1, label='Energy UGT1 [kWh]', color=color_dict['Pgreen_UGT1_DA'], linestyle='--')
    ax2_2.plot(remained_UGT2, label='Energy UGT2 [kWh]', color=color_dict['Pgreen_UGT2_DA'], linestyle='--')
    ax2_2.plot(remained_UGT3, label='Energy UGT3 [kWh]', color=color_dict['Pgreen_UGT3_DA'], linestyle='--')
    ax2_1.set_ylabel('Power [kW]', fontsize=label_fontsize)
    ax2_2.set_ylabel('Remained quota [kWh]', fontsize=label_fontsize)
    ax2_1.tick_params(axis='y', labelsize=tick_fontsize)    
    ax2_2.tick_params(axis='y', labelsize=tick_fontsize)
    ax2_2.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    # Build custom xticks (every 12 hrs from 15-min interval)
    x_ticks_index = result_df.index[::48]
    labels = []
    for i, x in enumerate(x_ticks_index):
        dt = result_df['Datetime'].iloc[x]
        if i % 2 == 0:  # every 96 steps = 24h
            labels.append(f"\n{dt.strftime('%d-%b-%y')}")
        else:
            labels.append(dt.strftime('%H:%M'))
    # Apply xticks and labels to both subplots
    # Show legends for both y-axes, get_legend_handles_labels are for combining legends
    lines_1, labels_1 = ax1_1.get_legend_handles_labels()
    lines_2, labels_2 = ax1_2.get_legend_handles_labels()
    ax1_1.legend(lines_1 + lines_2, labels_1 + labels_2,
                 fontsize=legend_fontsize,
                 loc='upper center',
                 bbox_to_anchor=(0.5, -0.3),
                 ncol=3,
                 frameon=False)
    lines_3, labels_3 = ax2_1.get_legend_handles_labels()
    lines_4, labels_4 = ax2_2.get_legend_handles_labels()
    ax2_1.legend(lines_3 + lines_4, labels_3 + labels_4,
                 fontsize=legend_fontsize,
                 loc='upper center',
                 bbox_to_anchor=(0.5, -0.3),
                 ncol=3,
                 frameon=False)
    for ax in [ax1_1, ax2_1]:
        ax.set_xticks(x_ticks_index)
        ax.set_xticklabels(labels, fontsize=tick_fontsize)
        for label in ax.get_xticklabels():
            label.set_ha('center')
            label.set_va('top')
            label.set_linespacing(1.5)
        ax.tick_params(axis='x', labelsize=tick_fontsize)
        ax.grid(visible=True, which='both', axis='x', linestyle='--', linewidth=0.5)
    # Ensure x-ticks are visible on both subplots
    ax1_1.tick_params(labelbottom=True)
    fig.tight_layout()
    return fig

def plot_REdays_net_cost(net_costs:list, 
                         figsize=(8,6),
                         tick_fontsize=14,
                         label_fontsize=16,
                         title_fontsize=16,
                         planning_date:str='2024-06-07'):
    
    # Create bar chart
    plt.figure(figsize=figsize)
    plt.bar(range(1, len(net_costs)+1), net_costs, color='skyblue')
    plt.xlabel('Number of anticipated RE days', fontsize=label_fontsize)
    plt.ylabel('Cost [THB]', fontsize=label_fontsize)
    plt.xticks(range(1, len(net_costs)+1), fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    plt.title(f'Net electricity cost from planning date {planning_date} for the next {len(net_costs)} days', fontsize=title_fontsize)
    # set the thinckness of boundary lines to 0.2
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(0.005)
    return plt.gcf()


def plot_boxplot_per_RE_days(
        data_to_plot:list,
        title:str='Net Cost per kWh of Net Load vs. Number of RE Days',
        ylabel:str='Net Cost per kWh of Net Load',
        figsize=(10, 6),
        tick_fontsize=14,
        label_fontsize=16,
        title_fontsize=16,
        ylim=[None, None]):
    # Create boxplot
    plt.figure(figsize=figsize)
    plt.boxplot(data_to_plot)
    plt.title(title, fontsize=title_fontsize)
    plt.xlabel("Number of anticipated RE100 days", fontsize=label_fontsize)
    plt.ylabel(ylabel, fontsize=label_fontsize)
    plt.xticks(range(1, len(data_to_plot)+1), fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    plt.title(title, fontsize=title_fontsize)
    plt.grid(visible=True, which='both', axis='y', linestyle='--', linewidth=0.5)
    plt.ylim(ylim)
    return plt.gcf()


# def plot_seasonal_stackbars(
#     rain_df,
#     winter_df,
#     summer_df,
#     figsize=(10, 14),
#     label_fontsize=20,
#     tick_fontsize=20,
#     title_fontsize=22,
#     legend_fontsize=20,
#     ylim=(0, 12),
#     color_dict=None,
# ):
#     # enable latex rendering (optional)
#     plt.rcParams["text.usetex"] = True

#     if color_dict is None:
#         color_dict = {
#             "measured_PV": "sandybrown",
#             "Pdchg": "orangered",
#             "Pgreen_DA": "forestgreen",
#             "Pnongreen_DA": "tomato",
#         }

#     season_specs = [
#         ("Rain", rain_df),
#         ("Winter", winter_df),
#         ("Summer", summer_df),
#     ]

#     fig, axs = plt.subplots(3, 1, figsize=figsize, sharex=True, sharey=True)

#     for ax, (title, df) in zip(axs, season_specs):
#         # stacked bars
#         b1 = ax.bar(df.index, df["PV"], label="PV", color=color_dict["measured_PV"])
#         b2 = ax.bar(
#             df.index,
#             df["Battery_Discharge"],
#             bottom=df["PV"],
#             label="Battery discharge",
#             color=color_dict["Pdchg"],
#         )
#         b3 = ax.bar(
#             df.index,
#             df["Imported_Renewable"],
#             bottom=df["PV"] + df["Battery_Discharge"],
#             label="Imported green",
#             color=color_dict["Pgreen_DA"],
#         )
#         # b4 = ax.bar(
#         #     df.index,
#         #     df["Imported_Grid"],
#         #     bottom=df["PV"] + df["Battery_Discharge"] + df["Imported_Renewable"],
#         #     label="Imported non-green",
#         #     color=color_dict["Pnongreen_DA"],
#         # )
#         l1, = ax.plot(df.index, df["Load"], label="Load", color="black", linewidth=2)

#         ax.set_title(title, fontsize=title_fontsize)
#         ax.set_ylabel("Power [kW]", fontsize=label_fontsize)
#         ax.tick_params(axis="x", labelsize=tick_fontsize)
#         ax.tick_params(axis="y", labelsize=tick_fontsize)
#         ax.set_ylim(ylim)
#         ax.grid(axis="x", alpha=0.2)

#     axs[-1].set_xlabel("Hour of day", fontsize=label_fontsize)

#     # one legend for the whole figure, bottom-center
#     handles, labels = axs[0].get_legend_handles_labels()
#     fig.legend(
#         handles,
#         labels,
#         loc="lower center",
#         bbox_to_anchor=(0.5, 0),
#         ncol=5,
#         fontsize=legend_fontsize,
#         frameon=False,
#     )

#     plt.tight_layout(rect=(0, 0.05, 1, 1))  # leave space for legend
#     return fig


def plot_seasonal_stackbars(
    rain_df,
    winter_df,
    summer_df,
    orientation="vertical",   # NEW
    figsize=(10, 14),
    label_fontsize=20,
    tick_fontsize=20,
    title_fontsize=22,
    legend_fontsize=20,
    ylim=(0, 12),
    color_dict=None,
):
    # enable latex rendering (optional)

    if color_dict is None:
        color_dict = {
            "measured_PV": "sandybrown",
            "Pdchg": "orangered",
            "Pgreen_DA": "forestgreen",
            "Pnongreen_DA": "tomato",
        }

    season_specs = [
        ("Rainy", rain_df),
        ("Winter", winter_df),
        ("Summer", summer_df),
    ]

    # --- layout control ---
    if orientation == "vertical":
        fig, axs = plt.subplots(3, 1, figsize=figsize, sharex=True, sharey=True)
    elif orientation == "horizontal":
        fig, axs = plt.subplots(1, 3, figsize=figsize, sharey=True)
    else:
        raise ValueError("orientation must be 'vertical' or 'horizontal'")

    # ensure axs is iterable in both cases
    axs = axs.flatten()

    for ax, (title, df) in zip(axs, season_specs):
        ax.bar(df.index, df["PV"],
               label="PV",
               color=color_dict["measured_PV"])

        ax.bar(df.index,
               df["Battery_Discharge"],
               bottom=df["PV"],
               label="Battery discharge",
               color=color_dict["Pdchg"])

        ax.bar(df.index,
               df["Imported_Renewable"],
               bottom=df["PV"] + df["Battery_Discharge"],
               label="Imported green",
               color=color_dict["Pgreen_DA"])

        ax.plot(df.index,
                df["Load"],
                label="Load",
                color="black",
                linewidth=0.5
                )

        ax.set_title(title, fontsize=title_fontsize)
        ax.set_ylim(ylim)
        ax.tick_params(axis="x", labelsize=tick_fontsize)
        ax.tick_params(axis="y", labelsize=tick_fontsize)
        ax.set_xticks([0, 6, 12, 18])
        ax.set_xticklabels([0, 6, 12, 18], fontsize=tick_fontsize)
        ax.grid(axis="x", alpha=0.2)

    # --- axis labels ---
    if orientation == "vertical":
        for ax in axs:
            ax.set_ylabel("MWh", fontsize=label_fontsize)
        axs[-1].set_xlabel("Hour of day", fontsize=label_fontsize)
    else:  # horizontal
        axs[0].set_ylabel("MWh", fontsize=label_fontsize)
        for ax in axs:
            ax.set_xlabel("Hour of day", fontsize=label_fontsize)

    # --- shared legend ---
    handles, labels = axs[0].get_legend_handles_labels()
    if orientation == "vertical":
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0),
            ncol=5,
            fontsize=legend_fontsize,
            frameon=False,
        )
    else:  # horizontal
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.05),
            ncol=4,
            fontsize=legend_fontsize,
            frameon=False,
        )
    # reduce border thickness
    for ax in axs:
        for spine in ax.spines.values():
            spine.set_linewidth(0.1)
            spine.set_alpha(0.5)
            spine.set_color('gray')

    plt.tight_layout(rect=(0, 0.05, 1, 1))
    return fig
