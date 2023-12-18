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

start_week = 18

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
df_predicted_data = data.df_predicted_goals_all_fixtures
df_weekly_data = data.df_weekly_fpl
df_overall_data = data.df_overall_fpl

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

    # Sidebar for page 1                              
    player_of_interest = st.sidebar.selectbox('Select a player', options=all_players)

    # Create a slider for the minimum minutes
    min_minutes = st.sidebar.slider(
        'Minimum minutes', 
        min_value=0, 
        max_value=int(data.players_archive['Min'].max()), 
        value=400,  # initial value
        step=100
        )

    # Define columns for layout
    tab1, tab2, tab3, tab4 = st.tabs(["Analysis", "Scatter Plots", "Player Comparisons", "FPL Data"])

    # Inside each column, you can use st.image to add an image and st.button to add a button
    with tab1:
        st.header("Player Analysis")
        data_container1 = st.container()
        data_container2 = st.container()

        with data_container1:
            container1, container2 = st.columns(2)
            with container1:
                st.header("Player Statistics")
                player_info = gp.display_player_info(df_player, player_of_interest)
                st.text(player_info)
            with container2:
                fig1 = gp.radar_chart_player(df_player, player_of_interest, gen_attributes, 'General')
                st.pyplot(fig1) 

    with tab2:
        st.header("Scatter Plots Analysis")

        fig2 = gp.plot_player_scatter(df_player, player_of_interest, min_minutes)
        st.pyplot(fig2) 


        field1_list = ['npxG', 'xAG']
        field2_list = ['Sh','SoT','Touches (Att Pen)', 'Carries Into Final 3rd','Passes Into Final Third',
                        'Passes Into Pen Area','Crosses Into Pen Area','Key Passes', 'Shot Creating Actions',
                        'Goal Creating Actions', 'xG Minus xGA (On Pitch)']

        selected_field1 = st.selectbox('Select field for x axis', field1_list)
        selected_field2 = st.selectbox('Select field for y axis', field2_list)

        fig201 = gp.plot_player_scatter_additional(df_player, player_of_interest, min_minutes, selected_field1, selected_field2)
        st.pyplot(fig201)

    with tab3:
        st.header("Player Comparisons")

        fig202 = gp.top_xa_xg(df_player, player_of_interest, min_minutes)
        st.pyplot(fig202)

        fig203 = gp.swarm_plot(df_player, player_of_interest)
        st.pyplot(fig203)

        fig3 = gp.player_bars(df_player, player_of_interest)
        st.pyplot(fig3)

        fig4 = gp.plot_player_comparison_per_90(df_player, player_of_interest, min_minutes)
        st.pyplot(fig4)

    with tab4:
        st.header("FPL Data")

        st.text("Overall FPL Data")
        st.dataframe(df_overall_data)

        fig400 = gp.weekly_points(df_weekly_data, df_overall_data, player_of_interest)
        st.pyplot(fig400)

        fig401 = gp.weekly_xg(df_weekly_data, df_overall_data, player_of_interest)
        st.pyplot(fig401)

        fig402 = gp.weekly_xA(df_weekly_data, df_overall_data, player_of_interest)
        st.pyplot(fig402)

        fig403 = gp.weekly_fpltransfers(df_weekly_data, player_of_interest)
        st.pyplot(fig403)
        

def page_two():
    st.title("Team Statistics")

    # Use Streamlit to create a player selection widget in the sidebar
    team_of_interest = st.sidebar.selectbox(
        'Select a Team',
        options= teams
    )
    
    tab1, tab2, tab3 = st.tabs(["Individual Team Data", "Overal Team Data", "Overall Predictive Data"])


    with tab1:
        st.header("Individual Team Data\n")
        year = '2023-2024'
        st.text(f'{year} {team_of_interest} match data')
        cols = ['date', 'round', 'venue', 'team', 'opponent', 'result', 'gf', 'ga', 'xg', 'xga', 'possession']
        team_data = df_teams_indvidual[cols][(df_teams_indvidual['team'] == team_of_interest) 
                                                & (df_teams_indvidual['year'] == year)]
        st.dataframe(team_data)

        fig8 = gp.individual_team_rollingxg(df_teams_indvidual, team_of_interest)
        st.pyplot(fig8)

        fig81 = gp.plot_fixture_difficulty(team_of_interest, window = 4)
        st.pyplot(fig81)

        fig6 = gp.age_histogram(df_player, team_of_interest)
        st.pyplot(fig6)

    with tab2:
        st.header("Overal Team Data")  
    
        styled_df = gp.fixture_heatmap(current_gameweek = start_week, match_num = 6)
        st.dataframe(styled_df)

        fig5 = gp.plot_team_scatter(df_teams, team_of_interest)
        st.pyplot(fig5)

        fig501 = gp.shot_quality_team(df_teams_attacking, team_of_interest) #, team_of_interest)
        st.pyplot(fig501)

        fig7 = gp.teams_xg_xa_trend(df_teams,team_of_interest)
        st.pyplot(fig7)

    with tab3:
        st.header("Predictive Data")

        fig520 = gp.predicted_data(df_predicted_data, team_of_interest, start_week = start_week)
        st.pyplot(fig520)  

        fig511 = gp.plot_tables_exp_act(df_teams_tables)
        st.pyplot(fig511)

        tab31, tab32, tab33 = st.tabs(["Predicted Goals Scored", "Predicted Goals Conceded", "Predicted Clean Sheet %"])

        with tab31:
            st.header("Team Predicted Goals Scored")

            fig3001 = gp.predict_goals(df_predicted_data, start_week = start_week)
            st.pyplot(fig3001)

        with tab32:
            st.header("Team Predicted Goals Conceded")

            fig3002 = gp.predict_goals_conceded(df_predicted_data, start_week = start_week)
            st.pyplot(fig3002)

        with tab33:
            st.header("Team Predicted Clean Sheet %")

            fig3002 = gp.predict_clean_sheets(df_predicted_data, start_week = start_week)
            st.pyplot(fig3002)


pages = {
    "Individual Player Statistics": page_one,
    "Team Statistics": page_two,
}

st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", list(pages.keys()))
page = pages[selection]
page()