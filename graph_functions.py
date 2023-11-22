import plotly.graph_objs as go
import pandas as pd
import matplotlib.pyplot as plt
from math import pi
from scipy.stats import percentileofscore
import textwrap
import seaborn as sns
from highlight_text import fig_text
import data_proc as data
import matplotlib.colors as mcolors
import seaborn as sns
import numpy as np

def plot_player_comparison_per_90(df, player_of_interest, mins):
    # Get the position of the player of interest
    player_position = df[df['Player'] == player_of_interest]['Pos'].values[0]

    # Get the subset of the dataframe that corresponds to the player's position and has played more than 900 minutes
    df_position = df[(df['Pos'] == player_position) & (df['Min'])].copy()

    # Calculate the npxG/90 + xAG/90 for each player
    df_position['npxG+xAG_per_90'] = (df_position['npxG'] + df_position['xAG']) / (df_position['Min'] / 90)

    #filtering out anything ridiculous
    df_position = df_position[df_position['npxG+xAG_per_90'] < 3]

    # Get the list of unique clubs
    clubs = sorted(df_position['Squad'].unique())

    # Determine the maximum value for the x-axis across all players
    df_position = df_position[df_position['Min'] > mins]
    max_value = df_position['npxG+xAG_per_90'].max()

    # Number of rows and columns for the subplots
    n_rows = len(clubs) // 4 + (len(clubs) % 4 > 0)
    n_cols = 4

    # Create the figure with subplots arranged in rows of 4
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(20, n_rows*5))
    axs = axs.flatten()  # Flatten the axes array for easier iteration

    fig.suptitle('Players npxG + xAG per 90 Compared To Player From Their Own Team',
                    fontsize=16)

    # For each club, create a bar chart of the top 5 players in terms of npxG+xAG per 90
    for ax, club in zip(axs, clubs):
        # Get the subset of the dataframe that corresponds to the current club
        df_club = df_position[(df_position['Squad'] == club)]

        # Sort the players by npxG+xAG_per_90 in descending order and take the top 5
        top_players = df_club.sort_values('npxG+xAG_per_90', ascending=False).head(5)

        # Determine the color of the bars
        bar_colors = ['red' if player == player_of_interest else plt.cm.Greens(val/max_value) for player, val in zip(top_players['Player'], top_players['npxG+xAG_per_90'])]

        # Create the bar chart with the determined colors
        ax.barh(top_players['Player'], top_players['npxG+xAG_per_90'], color=bar_colors)

        # Set the title of the subplot to the name of the club
        ax.set_title(club)

        # Set the same maximum value for the x-axis in all subplots
        ax.set_xlim(0, max_value)

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    # Remove extra subplots if total clubs is not a multiple of 4
    if len(clubs) % 4 != 0:
        for ax in axs[len(clubs):]:
            fig.delaxes(ax)

    # Add padding between the subplots and set the layout to tight
    fig.tight_layout(pad=3.0)

    return fig

def plot_player_scatter(df, player_of_interest, min_minutes):
    # Get the position of the player of interest
    player_position = df[df['Player'] == player_of_interest]['Pos'].values[0]

    # Create a filtered DataFrame based on the player's position and minimum minutes
    filtered_df = df[(df['Pos'] == player_position) & (df['Min'] >= min_minutes)].copy()

    # Calculate per 90 stats
    filtered_df['xAG_per_90'] = filtered_df['xAG'] / (filtered_df['Min'] / 90)
    filtered_df['npxG_per_90'] = filtered_df['npxG'] / (filtered_df['Min'] / 90)

    #filtering out anything ridiculous
    filtered_df = filtered_df[filtered_df['xAG_per_90'] < 1.5]
    filtered_df = filtered_df[filtered_df['npxG_per_90'] < 1.5]

    # Create the scatter plot
    fig, ax = plt.subplots(figsize=(15, 12))

    # Add a grid
    ax.grid(True, linestyle='--', alpha=0.6)


    top_npxG_indices = filtered_df['npxG_per_90'].nlargest(10).index
    top_rows = filtered_df.loc[top_npxG_indices]
    top_xg_players = top_rows['Player'].values

    top_xa_indices = filtered_df['xAG_per_90'].nlargest(10).index
    top_rows2 = filtered_df.loc[top_xa_indices]
    top_xa_players = top_rows2['Player'].values

    if player_of_interest in filtered_df['Player'].values:
        colors = ['red' if player == player_of_interest 
                    else 'orange' if player in top_xg_players or player in top_xa_players
                        else 'green' for player in filtered_df['Player']]
    else:
        colors = 'green'

    max_minutes = filtered_df['Min'].max()
    normalized_minutes = (filtered_df['Min'] / max_minutes) * 500

    ax.scatter(filtered_df['xAG_per_90'], 
                filtered_df['npxG_per_90'], 
                s=normalized_minutes, #filtered_df['Min']/10,
                c=colors,
                edgecolor='black',  # Add an outline to the markers
                alpha=0.7)

    # Calculate and add the average lines
    average_x = filtered_df['xAG_per_90'].mean()
    average_y = filtered_df['npxG_per_90'].mean()
    plt.axhline(average_x, color='grey', linestyle='--', alpha=0.5)
    plt.axvline(average_y, color='grey', linestyle='--', alpha=0.5)

   # Annotate the player of interest
    if player_of_interest in filtered_df['Player'].values:
        player_data = filtered_df[filtered_df['Player'] == player_of_interest].iloc[0]
        ax.annotate(player_of_interest, 
                    (player_data['xAG_per_90'], player_data['npxG_per_90']), 
                    xytext=(0, -20), 
                    textcoords='offset points')

    # Exclude player_of_interest from top_xg_players
    top_xg_players = [player for player in top_xg_players if player != player_of_interest]

    # Annotate players in top_xg_players
    for player in top_xg_players:
        player_data = filtered_df[filtered_df['Player'] == player].iloc[0]
        ax.annotate(player, 
                    (player_data['xAG_per_90'], player_data['npxG_per_90']), 
                    xytext=(0, 10), 
                    textcoords='offset points')

    # Exclude player_of_interest and top_xg_players from top_xa_players
    top_xa_players = [player for player in top_xa_players if player != player_of_interest and player not in top_xg_players]

    # Annotate players in top_xa_players
    for player in top_xa_players:
        player_data = filtered_df[filtered_df['Player'] == player].iloc[0]
        ax.annotate(player, 
                    (player_data['xAG_per_90'], player_data['npxG_per_90']), 
                    xytext=(0, 10), 
                    textcoords='offset points')



    ax.annotate('Average of dataset',
                xy=(average_y, average_x),
                xytext=(0, (average_x/1.1)),
                arrowprops=dict(facecolor='black',
                                width=1,
                                headwidth=8,
                                connectionstyle='angle3, angleA=90, angleB=0'),
                verticalalignment='center'
    )

     # Set the title and labels
    ax.set_title(f'xAG per 90 against npxG per 90 for the {player_position} position with over {min_minutes} minutes (EPL)', 
                    size=30, pad=20)
    ax.set_xlabel('xAG per 90', size=16)
    ax.set_ylabel('npxG per 90', size=16)

    ax.set_xlabel('xAG per 90')
    ax.set_ylabel('npxG per 90')
    ax.set_title(f'xAG per 90 against npxG per 90 for the {player_position} position with over {min_minutes} minutes (EPL)')
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    return plt.gcf()  # Return the current figure


def plot_player_scatter_additional(df, player_of_interest, min_minutes, field1, field2):
    # Get the position of the player of interest
    player_position = df[df['Player'] == player_of_interest]['Pos'].values[0]

    # Create a filtered DataFrame based on the player's position and minimum minutes
    filtered_df = df[(df['Pos'] == player_position) & (df['Min'] >= min_minutes)].copy()

    # Calculate per 90 stats
    filtered_df['field1_per_90'] = filtered_df[field1] / (filtered_df['Min'] / 90)
    filtered_df['field2_per_90'] = filtered_df[field2] / (filtered_df['Min'] / 90)

    # Create the scatter plot
    fig, ax = plt.subplots(figsize=(15, 12))

    # Add a grid
    ax.grid(True, linestyle='--', alpha=0.6)

    #top players
    top_indices = filtered_df['field1_per_90'].nlargest(10).index
    top_rows = filtered_df.loc[top_indices]
    top_players = top_rows['Player'].values

    top_indices2 = filtered_df['field2_per_90'].nlargest(10).index
    top_rows2 = filtered_df.loc[top_indices2]
    top_players2 = top_rows2['Player'].values

    if player_of_interest in filtered_df['Player'].values:
        colors = ['red' if player == player_of_interest 
                    else 'orange' if player in top_players or player in top_players2
                        else 'green' for player in filtered_df['Player']]
    else:
        colors = 'green'

    max_minutes = filtered_df['Min'].max()
    normalized_minutes = (filtered_df['Min'] / max_minutes) * 500

    ax.scatter(filtered_df['field1_per_90'], 
                filtered_df['field2_per_90'], 
                s=normalized_minutes, #filtered_df['Min']/10,
                c=colors,
                edgecolor='black',  # Add an outline to the markers
                alpha=0.7)


   # Annotate the player of interest
    if player_of_interest in filtered_df['Player'].values:
        player_data = filtered_df[filtered_df['Player'] == player_of_interest].iloc[0]
        ax.annotate(player_of_interest, 
                    (player_data['field1_per_90'], player_data['field2_per_90']), 
                    xytext=(0, -20), 
                    textcoords='offset points')

    # Exclude player_of_interest from top_xg_players
    top_players = [player for player in top_players if player != player_of_interest]

    # Annotate players in top_xg_players
    for player in top_players:
        player_data = filtered_df[filtered_df['Player'] == player].iloc[0]
        ax.annotate(player, 
                    (player_data['field1_per_90'], player_data['field2_per_90']), 
                    xytext=(0, 10), 
                    textcoords='offset points')

    # Exclude player_of_interest and top_xg_players from top_xa_players
    top_players2 = [player for player in top_players2 if player != player_of_interest and player not in top_players]

    # Annotate players in top_xa_players
    for player in top_players2:
        player_data = filtered_df[filtered_df['Player'] == player].iloc[0]
        ax.annotate(player, 
                    (player_data['field1_per_90'], player_data['field2_per_90']), 
                    xytext=(0, 10), 
                    textcoords='offset points')



     # Set the title and labels
    ax.set_title(f'{field1} against {field2} for the {player_position} position with over {min_minutes} minutes (EPL)', 
                    size=20, pad=10)
    ax.set_xlabel(f'{field1}', size=16)
    ax.set_ylabel(f'{field2}', size=16)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    return plt.gcf()  # Return the current figure


def top_xa_xg(df, player_of_interest, min_minutes):

    # Get the position of the player of interest
    player_position = df[df['Player'] == player_of_interest]['Pos'].values[0]

    # Create a filtered DataFrame based on the player's position and minimum minutes
    df = df[(df['Pos'] == player_position) & (df['Min'] >= min_minutes)].copy()

    # Function to get top 20 players and add selected player if not in top 20
    def get_top_players_with_selected(metric):
        top_players = df.nlargest(20, metric)
        if player_of_interest not in top_players['Player'].values:
            player_of_interest_row = df[df['Player'] == player_of_interest]
            top_players = pd.concat([top_players, player_of_interest_row])
        return top_players

    # Get top 10 players for npxG and xAG along with the selected player
    top_npxg_players = get_top_players_with_selected('npxG')
    top_xag_players = get_top_players_with_selected('xAG')

    # Set up the matplotlib figure
    fig, axs = plt.subplots(ncols=2, figsize=(15, 6))

    # Draw side-by-side bar plots with selected player highlighted in red
    sns.barplot(x='npxG', y='Player', data=top_npxg_players.sort_values('npxG', ascending=False), ax=axs[0], 
                palette=['red' if player == player_of_interest else 'green' for player in top_npxg_players['Player']])
    sns.barplot(x='xAG', y='Player', data=top_xag_players.sort_values('xAG', ascending=False), ax=axs[1], 
                palette=['red' if player == player_of_interest else 'green' for player in top_xag_players['Player']])

    # Set titles and labels
    axs[0].set_title('Top npxG with Selected Player')
    axs[0].set_xlabel('npxG')
    axs[0].set_ylabel('Player')

    axs[1].set_title('Top xAG with Selected Player')
    axs[1].set_xlabel('xAG')
    axs[1].set_ylabel('Player')

    # Display the plots
    plt.tight_layout()
    return plt.gcf()  # Return the current figure



def swarm_plot(df, player_of_interest):
    # Get the position of the player of interest
    # player_position = df[df['Player'] == player_of_interest]['Pos'].values[0]
    # Filter DataFrame to include only players with position 'MF' and create a copy of it
    # Filter DataFrame to include only players with position 'MF' and create a copy of it
    player_position = df[df['Player'] == player_of_interest]['Pos'].values[0]
    mf_players = df[df['Pos'] == player_position].copy()

    # Add a new column to the DataFrame to identify the selected player
    mf_players.loc[:, 'Selected'] = mf_players['Player'] == player_of_interest

    # Set up the matplotlib figure with two subplots
    fig, axs = plt.subplots(nrows=2, figsize=(10, 8), sharex=True)

    # Create a strip plot for npxG with decreased marker size and dodge set to False
    sns.stripplot(x='npxG', y='Pos', hue='Selected', data=mf_players, ax=axs[0], size=5, palette={True: 'red', False: 'green'},
                jitter=True, dodge=False)
    axs[0].set_title(f'Distribution of npxG for Position {player_position}')
    axs[0].set_xlabel('npxG')
    axs[0].set_ylabel(f'Position {player_position}')

    # Create a strip plot for xAG with decreased marker size and dodge set to False
    sns.stripplot(x='xAG', y='Pos', hue='Selected', data=mf_players, ax=axs[1], size=5, palette={True: 'red', False: 'green'},
                jitter=True, dodge=False)
    axs[1].set_title(f'Distribution of xAG for Position {player_position}')
    axs[1].set_xlabel('xAG')
    axs[1].set_ylabel(f'Position {player_position}')

    # Remove the legend
    axs[0].get_legend().remove()
    axs[1].get_legend().remove()

    # Display the plots
    plt.tight_layout()
    return plt.gcf()  # Return the current figure


def shot_quality_team(df, team_selected): #, team):


    # # Load the updated teams' dataset
    # teams_shooting_file_path = '/path_to_your_file/current_teams_shooting.csv'
    # df = pd.read_csv(teams_shooting_file_path)

    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Create a scatter plot for Shots Conceded per 90 against npxG per Shot
    for i in range(df.shape[0]):
        team = df.iloc[i]
        color = 'red' if team['Squad'] == team_selected else 'green'  # Changed the color to green
        plt.scatter(x=team['npxG per Shot_y'], y=team['Sh per 90_y'], color=color, s=100)
        plt.text(x=team['npxG per Shot_y'], 
                y=team['Sh per 90_y'], 
                s=team['Squad'], 
                fontdict=dict(color='white',size=10)) #, 
                #bbox=dict(facecolor='yellow',alpha=0.5))

    # # Create a scatter plot for Shots Conceded per 90 against npxG per Shot
    # sns.scatterplot(x='npxG per Shot_y', y='Sh per 90_y', data=df, ax=ax, s=100, palette='green')

    # Calculate average lines
    average_shots_conceded_per_90 = df['Sh per 90_y'].mean()
    average_npxG_per_shot = df['npxG per Shot_y'].mean()

    # Adding average lines to divide the plot into four quadrants
    plt.axvline(x=average_npxG_per_shot, color='grey', linestyle='--')
    plt.axhline(y=average_shots_conceded_per_90, color='grey', linestyle='--')

    # # Adding labels for each team
    # for i in range(df.shape[0]):
    #     plt.text(x=df['npxG per Shot_y'].iloc[i], 
    #             y=df['Sh per 90_y'].iloc[i], 
    #             s=df['Squad'].iloc[i], 
    #             fontdict=dict(color='white',size=10))
    #             # , 
                # bbox=dict(facecolor='yellow',alpha=0.5))


    # Annotations and their positions
    annotations = {
        'top_right': 'Higher Defensive Vulnerability',
        'top_left': 'Allow More Shots of Lower Quality',
        'bottom_right': 'Allow Fewer but More Dangerous Shots',
        'bottom_left': 'Stronger Defensive Performance'
    }
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Adjusting the position of the annotations to be well within each quadrant
    quadrant_inner_positions = {
        'top_right': (average_npxG_per_shot + (xlim[1] - average_npxG_per_shot) * 0.75, 
                    average_shots_conceded_per_90 + (ylim[1] - average_shots_conceded_per_90) * 0.75),
        'top_left': (average_npxG_per_shot - (average_npxG_per_shot - xlim[0]) * 0.75, 
                    average_shots_conceded_per_90 + (ylim[1] - average_shots_conceded_per_90) * 0.75),
        'bottom_right': (average_npxG_per_shot + (xlim[1] - average_npxG_per_shot) * 0.75, 
                        average_shots_conceded_per_90 - (average_shots_conceded_per_90 - ylim[0]) * 0.75),
        'bottom_left': (average_npxG_per_shot - (average_npxG_per_shot - xlim[0]) * 0.75, 
                        average_shots_conceded_per_90 - (average_shots_conceded_per_90 - ylim[0]) * 0.75)
    }

    # Function to wrap text into specified number of words per line
    def wrap_text(text, words_per_line=3):
        words = text.split()
        lines = [' '.join(words[i:i + words_per_line]) for i in range(0, len(words), words_per_line)]
        return '\n'.join(lines)

    # Wrap annotations text and Adding annotations in each quadrant
    for quadrant, position in quadrant_inner_positions.items():
        wrapped_text = wrap_text(annotations[quadrant], words_per_line=3)
        plt.annotate(wrapped_text, xy=position, xycoords='data', 
                    fontsize=10, ha='center', va='center',
                    bbox=dict(boxstyle="round,pad=0.3", edgecolor="green", facecolor="black"))

    # Setting titles and labels
    ax.set_title('Shots Conceded per 90 against npxG per Shot for Teams')
    ax.set_xlabel('npxG per Shot')
    ax.set_ylabel('Shots Conceded per 90')

    # Display the plot
    plt.tight_layout()
    return plt.gcf()  # Return the current figure



def plot_team_scatter(df, team):
    # Set the size of the figure
    fig, ax = plt.subplots(figsize=(15, 12))

    # Add a grid
    ax.grid(True, linestyle='--', alpha=0.8)

    for i in range(len(df)):
        color = 'red' if df['Squad'][i] == team else 'green'

        ax.scatter(df['npxG per 90_x'][i], 
                df['npxG per 90_y'][i], 
                s=100,
                c=color,
                edgecolor='black',  # Add an outline to the markers
                alpha=0.7)

        # Add team names as annotations
        ax.annotate(df['Squad'][i],
                    (df['npxG per 90_x'][i], df['npxG per 90_y'][i]),
                    xytext=(-10, -20),
                    textcoords='offset points',
                    fontsize=12)


    # Calculate the averages for npxG and xA
    npxG_avg = df['npxG per 90_x'].mean()
    xA_avg = df['npxG per 90_y'].mean()
    plt.axhline(npxG_avg, color='grey', linestyle='--', alpha=0.5)
    plt.axvline(xA_avg, color='grey', linestyle='--', alpha=0.5)

    # Set the title and labels for the axes
    ax.set_title('Scatter Plot of npxG vs xA')
    ax.set_xlabel('npxG')
    ax.set_ylabel('xA')

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Invert y-axis
    ax.invert_yaxis()

    return plt.gcf()  # Return the current figure

# def radar_chart_player(df, player_of_interest, attributes, attribute_type):
#     # Get the position of the player of interest
#     player_position = df[df['Player'] == player_of_interest]['Pos'].values[0]

#     # Filter the DataFrame to include only rows for the same position
#     position_df = df[df['Pos'] == player_position]

#     # Get the values for the attributes for the player of interest
#     player_values = df[df['Player'] == player_of_interest][attributes].values.flatten()
#     # player_mins = (df[df['Player'] == player_of_interest]['Min'].values.flatten()) / 90
#     # player_values = player_values / player_mins

#     # Initialize an empty dictionary to store the maximum per 90 minutes values for each attribute
#     merged_df_per_90 = position_df.copy()

#     # Calculate the percentile rank for each attribute
#     percentile_values = [percentileofscore(merged_df_per_90[attribute], player_value) / 100 for attribute, player_value in zip(attributes, player_values)]

#     # The radar chart needs the data to be in a circular form, hence we add the first element to the end of the list
#     percentile_values += percentile_values[:1]

#     # Compute the angle of each axis in the plot
#     angles = [n / float(len(attributes)) * 2 * pi for n in range(len(attributes))]
#     angles += angles[:1]

#     # Initialize the spider plot
#     # fig, ax = plt.subplots(figsize=(5, 5))
#     plt.figure(figsize=(10, 10))
#     ax = plt.subplot(111, polar=True)

#     # Draw one axe per attribute and add labels
#     # get the length of the longest word in labels
#     max_len = max([len(word) for label in attributes for word in label.split()])
#     wrapped_attributes = ['\n'.join(textwrap.wrap(label, width=max_len)) for label in attributes]
#     plt.xticks(angles[:-1], wrapped_attributes, color='white', size=16)
#     # plt.xticks(angles[:-1], attributes, color='white', size=16)

#     # Draw the values
#     ax.plot(angles, percentile_values, linewidth=1, linestyle='dashed', color='green')

#     # Tidy up
#     # Fill the area, set y limits, title (plot)
#     ax.fill(angles, percentile_values, 'green', alpha=0.5)
#     ax.set_ylim([0, 1])
#     plt.title(f'{player_of_interest} - {attribute_type} Attribute Type',  size=20, pad=20, fontweight = 'bold')

#     return plt.gcf()


def radar_chart_player(df, player_of_interest, attributes, attribute_type):
    player_position = df[df['Player'] == player_of_interest]['Pos'].values[0]
    position_df = df[df['Pos'] == player_position]
    player_values = df[df['Player'] == player_of_interest][attributes].values.flatten()

    merged_df_per_90 = position_df.copy()
    percentile_values = [percentileofscore(merged_df_per_90[attribute], player_value) / 100 
                            for attribute, player_value 
                            in zip(attributes, player_values)]
    percentile_values += percentile_values[:1]

    # We will also add the first player value to the end of the list for the labels
    player_values = list(player_values)
    player_values += player_values[:1]

    angles = [n / float(len(attributes)) * 2 * pi for n in range(len(attributes))]
    angles += angles[:1]

    plt.figure(figsize=(10,8), dpi=300)
    ax = plt.subplot(111, polar=True)

    max_len = max([len(word) for label in attributes for word in label.split()])
    wrapped_attributes = ['\n'.join(textwrap.wrap(label, width=max_len)) for label in attributes]

    plt.xticks(angles[:-1], wrapped_attributes, color='white', size=16)

    # Draw the values
    ax.plot(angles, percentile_values, linewidth=1, linestyle='dashed', color='green')

    # Add labels for each point
    for angle, attribute, player_value in zip(angles, attributes, player_values):
        ax.text(angle, percentileofscore(merged_df_per_90[attribute], player_value) / 100, f"{player_value:.2f}",
                ha='center', va='center', bbox=dict(facecolor='white', edgecolor='dimgray', alpha=0.4))

    ax.fill(angles, percentile_values, 'green', alpha=0.5)
    ax.set_ylim([0, 1])
    plt.title(f'{player_of_interest} - {attribute_type} Attribute Type',  size=20, pad=20, fontweight = 'bold')

    return plt.gcf()


def player_bars(df, player_of_interest):

    # attributes = ["G-PK", "npxG", "Sh per 90", "Ast", "xAG", "Shot Creating Actions", 
    #                 "Progressive Passes", "Progressive Carries", "Carries Into Final 3rd", 
    #                 "Carries Into Penalty Area", "Successful Take-Ons"]

    # Get the position of the player of interest
    player_position = df[df['Player'] == player_of_interest]['Pos'].values[0]

    if player_position in ['MF','FW']:
        attributes = ["G-PK", "npxG", "Sh per 90", "Ast", "xAG", "Shot Creating Actions", 
                    "Progressive Passes", "Progressive Carries", "Carries Into Final 3rd", 
                    "Carries Into Penalty Area", "Successful Take-Ons"]
    else:
        attributes = ["G-PK", "npxG", "Ast", "xAG", "Tackles", "Tackles Won", "Interceptions",
                        "Tackles and Interceptions", "Clearances", "Aerial Duels Won",
                        "Aerial Duels Win Pct",
                        "Progressive Passes", "Progressive Carries", "Passes Completed (Long)", 
                        "Passes Attempted (Long)", "Switches"]

    # Get the subset of the dataframe that corresponds to the player's position
    df_position = df[df['Pos'] == player_position]

    # Calculate the maximum values for each attribute in the player's position
    max_values = df_position[attributes].max()

    # Get the player's values for each attribute
    player_values = df[df['Player'] == player_of_interest][attributes].values.flatten()

    # Calculate the ratio of the player's values to the maximum values
    ratios = player_values / max_values

    # Create a DataFrame to store the data
    data = {
        'Attribute': attributes,
        'PlayerValue': player_values,
        'MaxValue': max_values,
        'Ratio': ratios
    }

    df_graph = pd.DataFrame(data)

    # Set the order of the attributes as provided
    df_graph.set_index('Attribute', inplace=True)

    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot the ratios as a horizontal bar chart with a color map from light green to dark green
    bars = ax.barh(df_graph.index, df_graph['Ratio'], color=plt.cm.Greens(df_graph['Ratio']))

    # Add the player's values, the ratios, and the names of the players with the maximum values as text labels on the bars
    for bar, attr in zip(bars, df_graph.index):
        max_player = df_position[df_position[attr] == df_position[attr].max()]['Player'].values[0]
        player_value = df_graph.loc[attr, 'PlayerValue']
        max_value = df_graph.loc[attr, 'MaxValue']
        ratio = df_graph.loc[attr, 'Ratio']
        ax.text(bar.get_width(), bar.get_y() + bar.get_height()/2, 
                f'{player_value:.2f}, {ratio:.2f}, {max_player} {max_value:.2f}', 
                va='center')

    # Set the x-axis label and the plot title
    ax.set_xlabel('Ratio to Maximum')
    ax.set_title(f'Performance of {player_of_interest} Compared to Maximum in Position ({player_position})')

    # Invert the y-axis so that the attribute with the highest ratio is at the top
    ax.invert_yaxis()

    # Set the x-axis limits from 0 to 1
    ax.set_xlim(0, 1)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    return plt.gcf()


def display_player_info(df, player_of_interest):
    # df_reset = df.reset_index(drop=True)
    # player_data = df_reset[df_reset['Player'] == player_of_interest]
    player_data = df[df['Player'] == player_of_interest]

    # Create a data frame with key statistics
    data = {
        'Player': player_of_interest,
        'Position': player_data['Pos'].values[0],
        'Club': player_data['Squad'].values[0],
        'Age': player_data['Age'].values[0],
        'Minutes Played': player_data['Min'].values[0],
        'Minutes Played Pct': player_data['Minutes Played Pct'].values[0], #player_data['Min'].values[0] / 3420,  # Assuming a full season is 3420 minutes
        'Goals': player_data['Gls'].values[0],
        'Non-Penalty Goals': player_data['G-PK'].values[0],
        'Non-Penalty xG': player_data['npxG'].values[0],
        'Shots per 90': player_data['Sh per 90'].values[0],
        'Assists': player_data['Ast'].values[0],
        'xAG': player_data['xAG'].values[0]
    }

    info_df = pd.DataFrame(data, index=[0], columns=None)

    # Reset the index and then transpose the DataFrame
    info_df = info_df.reset_index(drop=True).transpose()

   # Convert the DataFrame to a string and strip the first line (column header)
    info_str = info_df.to_string(header=False)

    return info_str


def age_histogram(df, team):
    # Create a new figure and axes
    fig, ax1 = plt.subplots(figsize=(14,6))

    # Filter the data for Tottenham
    team_data = df[df['Squad'] == team]

    # Histogram for Tottenham
    sns.histplot(team_data['Age'], bins=10, color='green', kde=True, alpha=0.8, label=team, ax=ax1)

    # Create a secondary y-axis
    ax2 = ax1.twinx()

    # Histogram for all teams on the secondary y-axis
    sns.histplot(df['Age'], bins=10, color='pink', kde=True, alpha=0.2, label='All Teams', ax=ax2)

    # Set the title and labels
    plt.title(f'Distribution of Player Ages in {team} vs All Teams')
    ax1.set_xlabel('Age')
    ax1.set_ylabel(f'Frequency - {team}')
    ax2.set_ylabel('Frequency - All Teams')

    ax1.grid(True, linestyle='--', alpha=0.8)

    # Show the legend
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    return plt.gcf()


def teams_xg_xa_trend(df, team):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14,20))

    for i in range(len(df)):
        color = 'red' if df['Squad'][i] == team else 'green'
        color2 = 'red' if df['Squad'][i] == team else 'white'
        ax1.scatter(df['npxG per 90_x'][i], 
                    df['Gls per 90_x'][i], 
                    s=300, 
                    c=color, 
                    edgecolor='black', 
                    alpha=0.7, 
                    marker='o')
        ax1.annotate(df['Squad'][i],
                    (df['npxG per 90_x'][i], 
                    df['Gls per 90_x'][i]),
                    xytext=(-10, -20),
                    textcoords='offset points',
                    fontsize=12,
                    color=color2)
        ax1.grid(True, linestyle='--', alpha=0.8)

    sns.regplot(x='npxG per 90_x', 
                y='Gls per 90_x', 
                data=df, 
                scatter=False, 
                ax=ax1,
                color='green',
                ci=95)

    ax1.set_title('npxG per 90 vs Goals Scored per 90')
    ax1.set_xlabel('npxG per 90')
    ax1.set_ylabel('Goals Scored per 90')

    # Second subplot: 'npxG per 90_y' vs 'Gls per 90_y' with inverted axes
    for i in range(len(df)):
        color = 'red' if df['Squad'][i] == team else 'green'
        color2 = 'red' if df['Squad'][i] == team else 'white'
        ax2.scatter(df['npxG per 90_y'][i], 
                    df['Gls per 90_y'][i], 
                    s=300, 
                    c=color, 
                    edgecolor='black', 
                    alpha=0.7, 
                    marker='o')
        ax2.annotate(df['Squad'][i],
                    (df['npxG per 90_y'][i], 
                    df['Gls per 90_y'][i]),
                    xytext=(-10, -20),
                    textcoords='offset points',
                    fontsize=12,
                    color=color2)
        ax2.grid(True, linestyle='--', alpha=0.8)
    
    ax2.invert_xaxis()
    ax2.invert_yaxis()

    sns.regplot(x='npxG per 90_y', 
                y='Gls per 90_y', 
                data=df, 
                scatter=False, 
                ax=ax2,
                color='green',
                ci=95)

    ax2.set_title('npXA per 90 vs Goals Conceded per 90')
    ax2.set_xlabel('npXA per 90')
    ax2.set_ylabel('Goals Conceded per 90')

    # Adjust the space between the subplots
    plt.subplots_adjust(hspace=0.2)

    return plt.gcf()


def individual_team_rollingxg(df, team):
    df = df[df['team']==team]
    df = df.reset_index(drop=True)
    df['xg_roll'] = df['xg'].rolling(window = 10, min_periods = 10).mean()
    df['xga_roll'] = df['xga'].rolling(window = 10, min_periods = 10).mean()
    fig = plt.figure(figsize=(10, 5), dpi = 300)
    ax = plt.subplot(111)

    ax.grid(True, linestyle='--', alpha=0.8)

    line_1 = ax.plot(df.index, df['xga_roll'], label = "xG conceded", color='red')
    line_2 = ax.plot(df.index, df['xg_roll'], label = "xG created",color='green')
    ax.set_ylim(0)

    # Fill between
    ax.fill_between(df.index, df['xg_roll'], df['xga_roll'], where = df['xg_roll'] < df['xga_roll'], 
                    interpolate = True, alpha = 0.5,zorder = 3,color='red')
    ax.fill_between(df.index, df['xg_roll'], df['xga_roll'], where = df['xg_roll'] >= df['xga_roll'], 
                    interpolate = True, alpha = 0.5, color='green')

    # Add a line to mark the division between seasons
    min_val_seasons = ax.get_ylim()[0]
    max_val_seasons = ax.get_ylim()[1]
    reset_points = df[df['index'] > df['index'].shift(-1)].index.tolist()
    for point in reset_points:
        ax.plot([point, point],[min_val_seasons, max_val_seasons], ls=":", lw=1.5, color="white", zorder=2)

    # Title and subtitle for the legend
    fig_text(x = 0.12, y = 0.98, s = team, color = "white", weight = "bold", size = 10, annotationbbox_kw={"xycoords": "figure fraction"})
    fig_text(x = 0.12, y = 0.95,
                s = "Expected goals <created> and <conceded> | 10-match rolling average\nEPL seasons 20/21, 21/22, 22/23 and 23/24",
                highlight_textprops = [{"color": line_2[0].get_color(), "weight": "bold"},{"color": line_1[0].get_color(), "weight": "bold"}],
                color = "white", size = 6, annotationbbox_kw={"xycoords": "figure fraction"})

    ax.legend()

    return plt.gcf()

def plot_fixture_difficulty(team, window = 4):
    fixtures_diff_df = data.fixtures_data(team, window)
    rank_rolling_avg = fixtures_diff_df["rank_rolling_avg"].values

    # Create the plot
    fig, ax = plt.subplots(figsize=(15, 7))  # Explicitly create a figure and axes
    sc = ax.scatter(fixtures_diff_df.index, fixtures_diff_df['rank_rolling_avg'], c=rank_rolling_avg, 
                     cmap='RdYlGn', s=200, edgecolors='black')
    ax.plot(fixtures_diff_df.index, fixtures_diff_df['rank_rolling_avg'], '--', color='grey')
    ax.set_title(f'Fixture Difficulty Rating - {team}')
    ax.set_xlabel('Match number')
    ax.set_ylabel(f'Fixture Difficulty {window}-Match Rolling Average')

    # Add colorbar explicitly tied to the scatter plot collection
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label('Difficulty Rolling Average')

    return fig


def fixture_heatmap(current_gameweek = 1, match_num = 6):
    # Function to compute the correct average rank for a team's fixtures
    def compute_corrected_average_rank(fixtures):
        """Compute the average rank for a team's fixtures."""
        ranks = []
        for fixture in fixtures:
            # Extract opponent from the fixture string using a more robust method
            opponent = ' '.join(fixture.split(" ")[:-1])
            rank_val = team_rank_mapping.get(opponent, None)
            if rank_val is not None:
                ranks.append(rank_val)
        return sum(ranks) / len(ranks) if ranks else None

    # Using the provided code and integrating the compute_average_rank function
    def get_combined_and_ordered_fixtures(team, df, matches, start_gameweek=1):
        #Get the home and away fixtures
        home_matches = df[df['Home'] == team].copy()
        home_matches['Fixture'] = home_matches['Away'] + " (H)"
        
        away_matches = df[df['Away'] == team].copy()
        away_matches['Fixture'] = away_matches['Home'] + " (A)"
        
        combined = pd.concat([home_matches, away_matches])
        combined_sorted = combined.sort_index()

        # Adjust start and end indices based on the current game week
        #will automate
        start_index = start_gameweek - 1
        end_index = start_index + matches

        return combined_sorted['Fixture'].iloc[start_index:end_index].tolist()
        # return combined_sorted['Fixture'].head(matches).tolist()

    # Define a custom colormap with softer shades of red and green
    soft_red_to_green = plt.cm.get_cmap("RdYlGn", 20)

    def color_based_on_rank_soft(val):
        # Extract team from the cell value
        team = ' '.join(val.split(" ")[:-1])
        rank = team_rank_mapping.get(team)
        
        # Return background color based on rank
        if rank is not None:
            color = soft_red_to_green((rank - 1) / 20)  # Assuming there are 20 teams, adjust if needed
            return f'color: black; background-color: {mcolors.rgb2hex(color)}'
        return ''

    fixtures_diff_df = data.fixtures_data_overall()

    # Extract the combined and ordered fixtures for each team
    fixtures_dict = {}
    for team in fixtures_diff_df['Away'].unique():
        fixtures_dict[team] = get_combined_and_ordered_fixtures(team = team,  
                                                                df = fixtures_diff_df, 
                                                                matches = match_num,
                                                                start_gameweek = current_gameweek)

    # Ensure each team has 6 fixtures, pad with "N/A" if necessary
    for team, fixtures in fixtures_dict.items():
        if len(fixtures) < match_num:
            fixtures += ["N/A"] * (match_num - len(fixtures))
            fixtures_dict[team] = fixtures


    # Convert dictionary to DataFrame and dynamically set columns
    df_fixtures = pd.DataFrame(fixtures_dict).T
    columns = [f'Fixture {i}' for i in range(current_gameweek, current_gameweek + match_num)]
    df_fixtures.columns = columns

    # Create a mapping of team to rank from the dataset
    team_rank_mapping = dict(zip(fixtures_diff_df['Away'], fixtures_diff_df['rank_y']))

    # Recompute the average rank for each team's fixtures using the corrected function
    # df_fixtures['Average_Fixture_Rank'] = df_fixtures[columns].apply(compute_corrected_average_rank, axis=1)
    # df_fixtures['Average_Fixture_Rank'] = df_fixtures.apply(lambda row: compute_corrected_average_rank(row[columns].tolist()), axis=1)
    df_fixtures['Average_Fixture_Rank'] = df_fixtures[columns].apply(compute_corrected_average_rank, axis=1)



    # Sort the teams based on average rank
    df_fixtures_sorted_corrected = df_fixtures.sort_values(by='Average_Fixture_Rank', ascending=False)

    # print(df_fixtures_sorted_corrected.tail(5))

    # Removing the row with NaN values
    styled_df_cleaned = df_fixtures_sorted_corrected[~df_fixtures_sorted_corrected.index.isna()]


    # Apply the coloring to the dataframe with the adjusted colormap and text color
    styled_df_soft = styled_df_cleaned.iloc[:,:-1].style.applymap(color_based_on_rank_soft, 
                                                                    subset=columns)
    return styled_df_soft

#################################################################################################################################
#################################################################################################################################
#################################################################################################################################
#################################################################################################################################
def plotly_team_scatter(df, team):
    colors = ['red' if squad == team else 'green' for squad in df['Squad']]

    fig = go.Figure(data=go.Scatter(
        x=df['npxG per 90_x'], 
        y=df['npxG per 90_y'],
        mode='markers+text',  # We add 'text' to the mode to display the labels
        marker=dict(
            color=colors,
            size=12,
            line=dict(
                color='black',
                width=2
            )
        ),
        text=df['Squad'],  # This will set the labels
        textposition="top center",  # You can change this to position the labels as you need
        textfont=dict(
            color="#E8E8E8"  # Set text color to light gray
        ),
        hoverlabel=dict(
            bgcolor=colors,
            bordercolor='black',
            font=dict(color='white', size=14),
            namelength=-1,
        ),
        hovertext=['Team: {}<br>npxG: {:.2f}<br>xA: {:.2f}'.format(squad, i, j) for squad, i, j in zip(df['Squad'], df['npxG per 90_x'], df['npxG per 90_y'])],
        hoverinfo="text",
        showlegend=False
    ))

    # Add average lines
    npxG_avg = df['npxG per 90_x'].mean()
    xA_avg = df['npxG per 90_y'].mean()
    fig.add_shape(type='line', line=dict(dash='dash', color='#E8E8E8'), x0=npxG_avg, x1=npxG_avg, y0=df['npxG per 90_y'].min(), y1=df['npxG per 90_y'].max())
    fig.add_shape(type='line', line=dict(dash='dash', color='#E8E8E8'), y0=xA_avg, y1=xA_avg, x0=df['npxG per 90_x'].min(), x1=df['npxG per 90_x'].max())

    # Update layout
    fig.update_layout(
        title='Scatter Plot of npxG vs xA',
        titlefont=dict(
            size=20,
            color="#E8E8E8"
        ),
        xaxis_title='npxG',
        yaxis_title='xA',
        plot_bgcolor='#383838',  # set background color to dark grey
        paper_bgcolor='#383838',  # set paper (outside plot) background color to dark grey
        width=700,  # Set figure width
        height=600,  # Set figure height
        xaxis=dict(
            color="#E8E8E8",
            gridcolor="#474747",
            zerolinecolor="#474747",
            titlefont=dict(
                color="#E8E8E8"
            )
        ),
        yaxis=dict(
            color="#E8E8E8",
            gridcolor="#474747",
            zerolinecolor="#474747",
            titlefont=dict(
                color="#E8E8E8"
            )
        )
    )

    # Invert y-axis
    fig.update_yaxes(autorange="reversed")

    return fig

def individual_team_rollingxg_plotly(df, team, start_date, end_date):
    df = df[df['team'] == team]
    df = df.reset_index(drop=True)
    df['xg_roll'] = df['xg'].rolling(window = 10, min_periods = 10).mean()
    df['xga_roll'] = df['xga'].rolling(window = 10, min_periods = 10).mean()

    # Filter df by start_date and end_date
    df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]

    fig = plt.figure(figsize=(10, 5), dpi = 300)
    ax = plt.subplot(111)

    ax.grid(True, linestyle='--', alpha=0.8)

    line_1 = ax.plot(df.index, df['xga_roll'], label = "xG conceded")
    line_2 = ax.plot(df.index, df['xg_roll'], label = "xG created")
    ax.set_ylim(0)

    ax.fill_between(df.index, df['xg_roll'], df['xga_roll'], where = df['xg_roll'] < df['xga_roll'], interpolate = True, alpha = 0.5,zorder = 3)
    ax.fill_between(df.index, df['xg_roll'], df['xga_roll'], where = df['xg_roll'] >= df['xga_roll'], interpolate = True, alpha = 0.5)

    min_val_seasons = ax.get_ylim()[0]
    max_val_seasons = ax.get_ylim()[1]
    reset_points = df[df['index'] > df['index'].shift(-1)].index.tolist()
    for point in reset_points:
        ax.plot([point, point],[min_val_seasons, max_val_seasons], ls=":", lw=1.5, color="white", zorder=2)

    ax.legend()

    return plt.gcf()

def plot_tables_exp_act(df):

    teams = df['team'].unique().tolist()

    actual_positions = df['Rk'].unique().tolist()
    expected_positions = df['Rank'].unique().tolist() 

    # Reverse the lists
    teams = teams[::-1]
    actual_positions = actual_positions[::-1]
    expected_positions = expected_positions[::-1]

    # Create a scatter plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot actual positions (in red) and expected positions (in yellow)
    ax.scatter(actual_positions, teams, color='green', s=300, label="Actual Position")
    ax.scatter(expected_positions, teams, color='red', s=300, label="Expected Position")

    # Draw arrows between actual and expected positions
    for team, actual, expected in zip(teams, actual_positions, expected_positions):
        ax.annotate("", xy=(expected, team), xytext=(actual, team),
                    arrowprops=dict(arrowstyle="->", lw=1.5))
        ax.text(actual, team, str(actual), color='white', ha='center', va='center', weight='bold')
        ax.text(expected, team, str(expected), color='black', ha='center', va='center', weight='bold')

    # Set labels, title and adjust x-axis
    ax.set_xlabel("Position")
    ax.set_title("Actual vs Expected Positions (EPL 2023-24)")
    ax.set_xlim(0, 21)  # Set x-axis limits from 1 to 20
    ax.legend(loc="upper right")

    plt.tight_layout()

    return plt.gcf() 

def predicted_data(df, team, start_week):
    # Parameters
    team = team
    start_week = start_week
    end_week = start_week + 9

    # Bring in the predicted data
    df_predicted_goals_all_fixtures = df

    # Select rows for the team selected within the gameweek range
    df_selected = df_predicted_goals_all_fixtures[
        ((df_predicted_goals_all_fixtures['Home'] == team) | (df_predicted_goals_all_fixtures['Away'] == team)) &
        (df_predicted_goals_all_fixtures['Week'].between(start_week, end_week))]

    # Get teams home and away data
    df_selected_home = df_selected[df_selected['Home'] == team].set_index('Week')
    df_selected_away = df_selected[df_selected['Away'] == team].set_index('Week')

    df_selected_home['Predicted Goals (Home)'] = df_selected_home['Predicted Goals (Home)'].round(1)
    df_selected_home['Predicted Goals (Away)'] = df_selected_home['Predicted Goals (Away)'].round(1)
    df_selected_away['Predicted Goals (Home)'] = df_selected_away['Predicted Goals (Home)'].round(1)
    df_selected_away['Predicted Goals (Away)'] = df_selected_away['Predicted Goals (Away)'].round(1)

    # Set up predicted goals scored for each game week
    predicted_goals_scored = []
    opponents_list = []
    for week in range(start_week, end_week + 1):
        if week in df_selected_home.index:
            predicted_goals_scored.append(df_selected_home.loc[week, 'Predicted Goals (Home)'])
            opponents_list.append(df_selected_home.loc[week, 'Away'])
        elif week in df_selected_away.index:
            predicted_goals_scored.append(df_selected_away.loc[week, 'Predicted Goals (Away)'])
            opponents_list.append(df_selected_away.loc[week, 'Home'])
        else:
            predicted_goals_scored.append(None)
            opponents_list.append("No Match")

    # Set up predicted goals conceded for each game week
    predicted_goals_conceded = []
    opponents_list = [] 
    for week in range(start_week, end_week + 1):
        if week in df_selected_home.index:
            predicted_goals_conceded.append(df_selected_home.loc[week, 'Predicted Goals (Away)'])
            opponents_list.append(df_selected_home.loc[week, 'Away'])
        elif week in df_selected_away.index:
            predicted_goals_conceded.append(df_selected_away.loc[week, 'Predicted Goals (Home)'])
            opponents_list.append(df_selected_away.loc[week, 'Home'])
        else:
            predicted_goals_conceded.append(None)
            opponents_list.append("No Match")

    # Set up clean sheet percentages for each game week
    clean_sheet_pct = []
    opponents_list = []
    for week in range(start_week, end_week + 1):
        if week in df_selected_home.index:
            clean_sheet_pct.append(df_selected_home.loc[week, 'Home Clean Sheet %'])
            opponents_list.append(df_selected_home.loc[week, 'Away'])
        elif week in df_selected_away.index:
            clean_sheet_pct.append(df_selected_away.loc[week, 'Away Clean Sheet %'])
            opponents_list.append(df_selected_away.loc[week, 'Home'])
        else:
            clean_sheet_pct.append(None)
            opponents_list.append("No Match")

    # Create dataframes for each of the rows needed
    df_goals_scored = pd.DataFrame(predicted_goals_scored, index=opponents_list, columns=['Predicted Goals Scored']).transpose()
    df_goals_conceded = pd.DataFrame(predicted_goals_conceded, index=opponents_list, columns=['Predicted Goals Conceded']).transpose()
    df_cleansheets_pct = pd.DataFrame(clean_sheet_pct, index=opponents_list, columns=['Predicted Clean Sheet %']).transpose()

    # Plot the heatmap
    plt.figure(figsize=(14, 4)) 

    # 1st plot
    plt.subplot(3, 1, 1)
    sns.heatmap(df_goals_scored, annot=True, cmap="Greens", cbar=False, linewidths=.5)
    plt.xticks([]) 
    plt.yticks(rotation=0, fontsize=12)
    plt.title(f'Predicted Data For {team} (Gameweeks {start_week} to {end_week})')

    # 2nd plot
    plt.subplot(3, 1, 2) 
    sns.heatmap(df_goals_conceded, annot=True, cmap="Greens", cbar=False, linewidths=.5)
    plt.xticks([]) 
    plt.yticks(rotation=0, fontsize=12)

    # 3rd plot
    plt.subplot(3, 1, 3)
    sns.heatmap(df_cleansheets_pct, annot=True, cmap="Greens", cbar=False, linewidths=.5)
    plt.xticks(ticks=np.arange(0.5, len(opponents_list)), labels=opponents_list, rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=12)

    plt.tight_layout(h_pad=0.1)

    return plt.gcf() 

def predict_goals(df, start_week):
    # start_week = 13
    end_week = start_week + 7

    # Filter for the weeks wanted
    df = df[df['Week'].between(start_week, end_week)]
    df = df.round(1)
    teams = np.unique(df[['Home', 'Away']].values) 
    goals_scored_pivot = pd.DataFrame(index=teams)

    # Predicted goals for each team
    for week in range(start_week, end_week):
        goals_scored_pivot[week] = 0  # Initialise
        week_games = df[df['Week'] == week]
        for _, row in week_games.iterrows():
            goals_scored_pivot.at[row['Home'], week] += row['Predicted Goals (Home)']
            goals_scored_pivot.at[row['Away'], week] += row['Predicted Goals (Away)']

    # Calculate the total predicted goals for each team
    goals_scored_pivot['Total'] = goals_scored_pivot.sum(axis=1)

    # Sort by the total (and remove)
    goals_scored_pivot_sorted = goals_scored_pivot.sort_values('Total', ascending=False)
    goals_scored_pivot_sorted = goals_scored_pivot_sorted.drop(columns='Total')

    # Plot the heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(goals_scored_pivot_sorted, annot=True, cmap="Greens", cbar=False, linewidths=1, linecolor='black')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.xlabel('Gameweek')
    plt.ylabel('Team')
    plt.title('Predicted Goals Scored')

    return plt.gcf()

def predict_goals_conceded(df, start_week):
    end_week = start_week + 7

    # Filter for the weeks wanted
    df = df[df['Week'].between(start_week, end_week)]
    df = df.round(1)
    teams = np.unique(df[['Home', 'Away']].values) 
    goals_scored_pivot = pd.DataFrame(index=teams)

    # Predicted goals for each team
    for week in range(start_week, end_week):
        goals_scored_pivot[week] = 0  # Initialise
        week_games = df[df['Week'] == week]
        for _, row in week_games.iterrows():
            goals_scored_pivot.at[row['Home'], week] += row['Predicted Goals (Away)']
            goals_scored_pivot.at[row['Away'], week] += row['Predicted Goals (Home)']

    # Calculate the total predicted goals for each team
    goals_scored_pivot['Total'] = goals_scored_pivot.sum(axis=1)

    # Sort by the total (and remove)
    goals_scored_pivot_sorted = goals_scored_pivot.sort_values('Total', ascending=True)
    goals_scored_pivot_sorted = goals_scored_pivot_sorted.drop(columns='Total')

    # Plot the heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(goals_scored_pivot_sorted, annot=True, cmap="Greens", cbar=False, linewidths=1, linecolor='black')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.xlabel('Gameweek')
    plt.ylabel('Team')
    plt.title('Predicted Goals Conceded')

    return plt.gcf()

def predict_clean_sheets(df, start_week):
    end_week = start_week + 7

    # Filter for the weeks wanted
    df = df[df['Week'].between(start_week, end_week)]
    df = df.round(1)
    teams = np.unique(df[['Home', 'Away']].values) 
    goals_scored_pivot = pd.DataFrame(index=teams)

    # Predicted goals for each team
    for week in range(start_week, end_week):
        goals_scored_pivot[week] = 0  # Initialise
        week_games = df[df['Week'] == week]
        for _, row in week_games.iterrows():
            goals_scored_pivot.at[row['Home'], week] += row['Home Clean Sheet %']
            goals_scored_pivot.at[row['Away'], week] += row['Away Clean Sheet %']

    # Calculate the total predicted goals for each team
    goals_scored_pivot['Total'] = goals_scored_pivot.sum(axis=1)

    # Sort by the total (and remove)
    goals_scored_pivot_sorted = goals_scored_pivot.sort_values('Total', ascending=False)
    goals_scored_pivot_sorted = goals_scored_pivot_sorted.drop(columns='Total')

    # Plot the heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(goals_scored_pivot_sorted, annot=True, cmap="Greens", cbar=False, linewidths=1, linecolor='black')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.xlabel('Gameweek')
    plt.ylabel('Team')
    plt.title('Predicted Clean Sheet %')

    return plt.gcf()


def weekly_points(df_weekly, df_overall, name):

    df_to_use = df_weekly[df_weekly['full_name'] == name]

    player_info = df_overall[['full_name', 'pos_short', 'now_cost', 'total_points']]
    player_info = player_info[player_info['full_name'] == name]
    total_points = player_info['total_points'].values[0]
    position = player_info['pos_short'].values[0]
    cost = player_info['now_cost'].values[0]

    points = df_to_use['total_points'].to_list()  
    gameweeks = list(range(1, len(points) + 1))

    # Create figure and plot space
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.bar(gameweeks, points, color='green')

    for i, pt in enumerate(points):
        ax.text(i + 1, pt + 0.5, f"{pt} PTS", ha='center')

    # Set labels and title
    ax.set_ylabel('Points')
    ax.set_title(f'Player Performance Across Gameweeks: {name} - £{cost}m - Total Points: {total_points} - Position: {position}')

    # Set x-axis labels as game week numbers
    ax.set_xticks(gameweeks)
    ax.set_xticklabels(gameweeks)

    # Set y-axis limit
    max_points = max(points)
    if min(points) < 0: 
        min_points = min(points) - 2
    else: min_points = 0

    ax.set_ylim(min_points, max_points + 2)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.2)

    return plt.gcf()



# def player_contributions(player,seasons):
#     import data_proc as data

#     df_20_21 = data.players_2020_2021 
#     df_21_22 = data.players_2021_2022
#     df_22_23 = data.merged_df


#     # Assuming you have data frames df_2021, df_2022, df_2023
#     dataframes = [df_20_21, df_21_22, df_22_23]
#     data = {}

#     # Define the player we are interested in
#     player_name = player

#     # Check which seasons the player is in
#     seasons = []
#     for i, df in enumerate(dataframes, start=20_21):
#         if player_name in df['Player'].unique():
#             seasons.append(str(i))
#             data[str(i)] = df



#     # Define the columns of interest based on actual column names in the dataset
#     interesting_data = {}

#     # Extract the interesting data for the specified player
#     for season in seasons:
#         player_data = data[season][data[season]['Player'] == player_name]
#         interesting_data[season] = {
#             'G': player_data['Gls'].values[0],
#             'A': player_data['Ast'].values[0],
#             'npxG': player_data['npxG'].values[0],
#             'xAG': player_data['xAG'].values[0],
#         }

#     ### project data for next season
#     # Initialize the sums for each category
#     sum_G = 0
#     sum_A = 0
#     sum_npxG = 0
#     sum_xAG = 0

#     # Calculate the sum for each category
#     for season in interesting_data.values():
#         sum_G += season['G']
#         sum_A += season['A']
#         sum_npxG += season['npxG']
#         sum_xAG += season['xAG']

#     # Calculate the average for each category
#     num_seasons = len(interesting_data)
#     avg_G = sum_G / num_seasons
#     avg_A = sum_A / num_seasons
#     avg_npxG = sum_npxG / num_seasons
#     avg_xAG = sum_xAG / num_seasons

#     # # Add the projections for the 23/24 season to our data
#     # interesting_data['23_24'] = {
#     #     'G': avg_G,
#     #     'A': avg_A,
#     #     'npxG': avg_npxG,
#     #     'xAG': avg_xAG,
#     # }

#     # Add the projections for the next season to our data
#     next_season = str(int(seasons[-1]) + 1)
#     interesting_data[next_season] = {
#         'G': avg_G,
#         'A': avg_A,
#         'npxG': avg_npxG,
#         'xAG': avg_xAG,
#     }
#     seasons.append(next_season)


#     # # Define the seasons in the correct order
#     # seasons = ['20_21', '21_22', '22_23', '23_24']

#     # Prepare the data for the plots
#     G_values = [interesting_data[season]['G'] for season in seasons]
#     A_values = [interesting_data[season]['A'] for season in seasons]
#     npxG_values = [interesting_data[season]['npxG'] for season in seasons]
#     xAG_values = [interesting_data[season]['xAG'] for season in seasons]

#     # Calculate the actual and expected contributions
#     actual_contributions = [G + A for G, A in zip(G_values, A_values)]
#     expected_contributions = [npxG + xAG for npxG, xAG in zip(npxG_values, xAG_values)]

#     # Determine the maximum value for the y-axis
#     max_y_value = max(max(actual_contributions), max(expected_contributions), 
#                         max(G_values), max(npxG_values), max(A_values), max(xAG_values)) + 5

#     # Create the scatter plot
#     fig, ax = plt.subplots(figsize=(15, 12))

#     # Add a grid
#     ax.grid(True, linestyle='--', alpha=0.6)

#     ax.plot(seasons, 
#             actual_contributions, 
#             marker="^", 
#             linestyle='dashed', 
#             color='green',  
#             markeredgecolor='black',
#             markersize=20, 
#             linewidth=1.5, 
#             alpha=0.7,
#             label='Actual Contributions (G + A)')

#     ax.plot(seasons, 
#             expected_contributions, 
#             marker="^", 
#             linestyle='dashed', 
#             color='red',  
#             markeredgecolor='black',
#             markersize=20, 
#             linewidth=1.5, 
#             alpha=0.7,
#             label='Expected Contributions (npxG + xA)')

#     # Add labels above each data point
#     for i, txt in enumerate(actual_contributions):
#         ax.annotate(f"{txt:.2f}", 
#                         (seasons[i], actual_contributions[i]), 
#                         textcoords="offset points", 
#                         xytext=(0,10), 
#                         ha='center', 
#                         fontsize=12, 
#                         color='lightgrey')

#     # Add labels above each data point
#     for i, txt in enumerate(expected_contributions):
#         ax.annotate(f"{txt:.2f}", 
#                         (seasons[i], expected_contributions[i]), 
#                         textcoords="offset points", 
#                         xytext=(0,10), 
#                         ha='center', 
#                         fontsize=12, 
#                         color='lightgrey')

#     labels = ['2020-2021','2021-2022','2022-2023','2023-2024 (Predicted)']
#     ax.set_xticklabels(labels)
#     ax.set_xlabel('Season')
#     ax.set_ylabel('Contributions')
#     plt.ylim(0, max_y_value)
#     plt.legend()

#     return plt.gcf()