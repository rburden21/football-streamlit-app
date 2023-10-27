import data_proc as data
import graph_functions as gp
import matplotlib.pyplot as plt
import streamlit as st

# Set the default style parameters for all charts
plt.rcParams['figure.facecolor'] = '#383838'  # Dark gray background color
plt.rcParams['axes.facecolor'] = '#383838'  # Dark gray axes background color
plt.rcParams['axes.edgecolor'] = '#E8E8E8'  # Light gray axes edge color
plt.rcParams['axes.labelcolor'] = '#E8E8E8'  # Light gray label color
plt.rcParams['xtick.color'] = '#E8E8E8'  # Light gray xtick color
plt.rcParams['ytick.color'] = '#E8E8E8'  # Light gray ytick color
plt.rcParams['text.color'] = '#E8E8E8'  # Light gray text color
plt.rcParams['grid.color'] = '#474747'  # Gray grid line color
plt.rcParams['grid.alpha'] = 0.8  # Grid line alpha   

axes_settings = {
        "spines.top" : False,
        "spines.right" : False,
        "titlesize" : 18
}

plt.rc("axes", **axes_settings)

gen_attributes = ['G-PK','npxG','Sh','Ast','xAG','npxG+xAG', 'Shot Creating Actions','Progressive Passes', 'PrgR']

def_attributes = ['Tackles Won', 'Dribblers Tackled', 'Dribblers Tackled Pct', 'Blocks', 'Shots Blocked', 'Passes Blocked',
                    'Tackles and Interceptions', 'Errors Leading to Shots', 'Aerial Duels Won', 'Aerial Duels Win Pct']

pos_attributes = ['Touches', 'Successful Take-Ons', 'Successful Take-On %','Carries', 'Progressive Carrying Distance', 
                    'Progressive Carries', 'Miscontrols', 'Disposessed', 'Passes Recieved']

shoot_attributes = ['Gls per 90', 'Sh', 'SoT', 'Sh per 90', 'SoT per 90', 'Average Shot Distance', 'npxG per Shot',
                    'G-xG', 'npG - npxG']

pass_attributes = ['Passes Completed', 'Pass Completion Pct', 'Total Passing Dist', 'Pass Completion Pct (Short)',
                    'Pass Completion Pct (Medium)','Pass Completion Pct (Long)', 'xA', 'Key Passes', 
                    'Passes Into Final Third', 'Progressive Passes']

gk_attributes = []

df_player = data.current_players
df_player_archive = data.players_archive
df_teams = data.merged_df_teams
df_teams_indvidual = data.df_individual
df_teams_attacking = data.team_shooting_df
df_teams_tables = data.df_table_exp_act

# df_player = df_player.reset_index(drop=True)
df_player = df_player.drop(columns=['index'])
duplicated_columns = df_player.columns[df_player.columns.duplicated()].tolist()
df_player = df_player.loc[:, ~df_player.columns.duplicated()]
#final lists
all_players = df_player[['Player']].iloc[:, 0].sort_values().unique()
teams = df_player[['Squad']].iloc[:, 0].sort_values().unique()


st.set_page_config(page_title='Dashboard', page_icon='', layout='wide')

def page_one():
    st.title("Individual Player Statistics")
    
    # Custom CSS to adjust the width of the main content area
    st.markdown(
        """
        <style>
        .reportview-container {
            max-width: 95%;  # You can adjust this value as you see fit
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

                              
    player_of_interest = st.sidebar.selectbox('Select a player', options=all_players)

    # # Determine which seasons the selected player appears in
    # seasons = []
    # if player_of_interest in df_2021['Player'].unique():
    #     seasons.append('2021')
    # if player_of_interest in df_2022['Player'].unique():
    #     seasons.append('2022')
    # if player_of_interest in df_2023['Player'].unique():
    #     seasons.append('2023')
    # if player_of_interest in df_2024['Player'].unique():
    #     seasons.append('2024')

    # Create a slider for the minimum minutes
    min_minutes = st.sidebar.slider(
        'Minimum minutes', 
        min_value=0, 
        max_value=int(data.players_archive['Min'].max()), 
        value=100,  # initial value
        step=100
        )

# #drop down for year if player exists there
#     # Use Streamlit to create a season selection widget in the sidebar
#     selected_season = st.selectbox('Select a season', options=seasons)
#     # Depending on the selected season, use the corresponding data frame to create the scatter plot
#     if selected_season == '2021':
#         df = df_2021
#     elif selected_season == '2022':
#         df = df_2022
#     elif selected_season == '2023':
#         df = df_2023
#     elif selected_season == '2024':
#         df = df_2024

    data_container1 = st.container()
    data_container2 = st.container()
    # data_container3 = st.container()

    with data_container1:
        container1, container2 = st.columns(2)
        with container1:
            st.header("Player Statistics")
            player_info = gp.display_player_info(df_player, player_of_interest)
            st.text(player_info)
        with container2:
            fig1 = gp.radar_chart_player(df_player, player_of_interest, gen_attributes, 'General')
            st.pyplot(fig1)  # Assume fig1 is the figure for the scatter plot

    # fig21 = gp.player_contributions(player_of_interest, seasons)
    # st.pyplot(fig21)


    # # Create a list of the available options
    # # Define the available options
    # options = ["General", "Defensive", "Possession", "Shooting", "Passing"]

    # # Create the dropdown menu and store the selected value
    # selected_chart = st.selectbox("Select a chart", options)

    # # Check which chart was selected and show the appropriate plot
    # if selected_chart == "General":
    #     fig15 = gp.radar_chart_player(df, player_of_interest, gen_attributes, 'General')
    #     st.pyplot(fig15)  # Assume fig1 is the figure for the scatter plot

    # elif selected_chart == "Defensive":
    #     fig11 = gp.radar_chart_player(df, player_of_interest, def_attributes, 'Defensive')
    #     st.pyplot(fig11)  # Assume fig1 is the figure for the scatter plot

    # elif selected_chart == "Possession":
    #     fig12 = gp.radar_chart_player(df, player_of_interest, pos_attributes, 'Possession')
    #     st.pyplot(fig12)  # Assume fig1 is the figure for the scatter plot

    # elif selected_chart == "Shooting":
    #     fig13 = gp.radar_chart_player(df, player_of_interest, shoot_attributes, 'Shooting')
    #     st.pyplot(fig13)

    # elif selected_chart == "Passing":
    #     fig14 = gp.radar_chart_player(df, player_of_interest, pass_attributes, 'Passing')
    #     st.pyplot(fig14)

    # else:
    #     st.info("Please select a chart from the dropdown menu")



    # fig2 = plot_player_scatter(merged_df, player_of_interest)
    fig2 = gp.plot_player_scatter(df_player, player_of_interest, min_minutes)
    st.pyplot(fig2)  # Assume fig1 is the figure for the scatter plot


    field1_list = ['npxG', 'xAG']
    field2_list = ['Sh','SoT','Touches (Att Pen)', 'Carries Into Final 3rd','Passes Into Final Third',
                    'Passes Into Pen Area','Crosses Into Pen Area','Key Passes', 'Shot Creating Actions',
                    'Goal Creating Actions', 'xG Minus xGA (On Pitch)']

    selected_field1 = st.selectbox('Select field for x axis', field1_list)
    selected_field2 = st.selectbox('Select field for y axis', field2_list)
    # field1 = 'npxG'
    # field2 = 'Touches (Att Pen)'
    fig201 = gp.plot_player_scatter_additional(df_player, player_of_interest, min_minutes, selected_field1, selected_field2)
    st.pyplot(fig201)  # Assume fig1 is the figure for the scatter plot

    fig202 = gp.top_xa_xg(df_player, player_of_interest, min_minutes)
    st.pyplot(fig202)

    fig203 = gp.swarm_plot(df_player, player_of_interest)
    st.pyplot(fig203)

    fig3 = gp.player_bars(df_player, player_of_interest)
    st.pyplot(fig3)

    fig4 = gp.plot_player_comparison_per_90(df_player, player_of_interest, min_minutes)
    st.pyplot(fig4)

def page_two():
    st.title("Team Statistics")

    # Use Streamlit to create a player selection widget in the sidebar
    team_of_interest = st.sidebar.selectbox(
        'Select a Team',
        options= teams
    )
    
    year = '2023-2024'
    # cols = ['date', 'round', 'venue', 'team', 'opponent', 'result', 'gf', 'ga', 'xg', 'xa', 'possession']
    st.text(f'{year} {team_of_interest} match data')
    # team_data = df_teams_indvidual[cols][(df_teams_indvidual['team'] == team_of_interest) 
    #                                     & (df_teams_indvidual['year'] == '2023-2024')]
    cols = ['date', 'round', 'venue', 'team', 'opponent', 'result', 'gf', 'ga', 'xg', 'xga', 'possession']
    team_data = df_teams_indvidual[cols][(df_teams_indvidual['team'] == team_of_interest) 
                                            & (df_teams_indvidual['year'] == year)]
    st.dataframe(team_data)

    # Display the dataframe
    # st.dataframe(df_individual[df_individual['team']==team_of_interest])

    fig5 = gp.plot_team_scatter(df_teams, team_of_interest)
    st.pyplot(fig5)

    fig511 = gp.plot_tables_exp_act(df_teams_tables)
    st.pyplot(fig511)

    fig501 = gp.shot_quality_team(df_teams_attacking, team_of_interest) #, team_of_interest)
    st.pyplot(fig501)


    # # To plot the figure in streamlit
    # fig51 = gp.plotly_team_scatter(data.merged_df_teams, team_of_interest)
    # st.plotly_chart(fig51)

    fig7 = gp.teams_xg_xa_trend(df_teams,team_of_interest)
    st.pyplot(fig7)

    fig8 = gp.individual_team_rollingxg(df_teams_indvidual, team_of_interest)
    st.pyplot(fig8)

    fig81 = gp.plot_fixture_difficulty(team_of_interest, window = 4)
    st.pyplot(fig81)

    styled_df = gp.fixture_heatmap(current_gameweek = 10, match_num = 8)
    st.dataframe(styled_df)

    fig6 = gp.age_histogram(df_player, team_of_interest)
    st.pyplot(fig6)


pages = {
    "Individual Player Statistics": page_one,
    "Team Statistics": page_two,
}

st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", list(pages.keys()))
page = pages[selection]
page()