import numpy as np
import os
import pandas as pd
import sqlite3
import utils as utils

folder_db = '/data/databases/'
folder_raw = '/data/raw_files/'

# Get the directory of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))
# Generate the path to the database
db_path = os.path.join(current_directory + folder_db, "fbref_data_players_latest.db")
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Write a SQL query to join all the tables
query = """
SELECT *
FROM general
LEFT JOIN keepers ON general.Player = keepers.Player
LEFT JOIN keepers_adv ON general.Player = keepers_adv.Player
LEFT JOIN shooting ON general.Player = shooting.Player
LEFT JOIN passing ON general.Player = passing.Player
LEFT JOIN passing_types ON general.Player = passing_types.Player
LEFT JOIN gca ON general.Player = gca.Player
LEFT JOIN defense ON general.Player = defense.Player
LEFT JOIN possession ON general.Player = possession.Player
LEFT JOIN playingtime ON general.Player = playingtime.Player
LEFT JOIN misc ON general.Player = misc.Player
"""

# Execute the query and load the result into a DataFrame
current_players = pd.read_sql_query(query, conn)

# Close the connection
conn.close()

db_path = os.path.join(current_directory + folder_db, "fbref_data_players_archive.db")
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Write a SQL query to join all the tables
query = """
SELECT *
FROM general
"""

# Execute the query and load the result into a DataFrame
players_archive = pd.read_sql_query(query, conn)

# Close the connection
conn.close()

# All teams in dataset
# # teams = current_players['Squad'].unique()

# Combine the dataframes for the for and against team data


# conn = sqlite3.connect('fbref_data_teams.db')
# df = pd.read_sql_query('SELECT * FROM teams_general', conn)
# conn.close()

# # Connect to the SQLite database
# conn = sqlite3.connect('fbref_data_teams_against.db')
# df2 = pd.read_sql_query('SELECT * FROM teams_against_general', conn)
# conn.close()

db_path = os.path.join(current_directory + folder_db, "fbref_for_team_data_overall_2023-2024.db")
conn = sqlite3.connect(db_path)
df1 = pd.read_sql_query('SELECT * FROM teams_for_general', conn)
conn.close()
df1

db_path = os.path.join(current_directory + folder_db, "fbref_against_team_data_overall_2023-2024.db")
conn = sqlite3.connect(db_path)
df2 = pd.read_sql_query('SELECT * FROM teams_against_general', conn)
conn.close()
df2

merged_df_teams = df1.merge(df2, left_on='Squad', right_on='Squad', how='left')

# TEAM SHOOTING DATA

table = 'shooting' #table_names[3]
status = ['for','against']
season = '2023-2024' 

db_name1 = f'fbref_{status[0]}_team_data_overall_{season}.db'
db_name2 = f'fbref_{status[1]}_team_data_overall_{season}.db'

db_path1 = os.path.join(current_directory + folder_db, db_name1)
db_path2 = os.path.join(current_directory + folder_db, db_name2)

conn = sqlite3.connect(db_path1)
df1 = pd.read_sql_query(f'SELECT * FROM teams_{status[0]}_{table}', conn)
conn.close()
df1

conn = sqlite3.connect(db_path2)
df2 = pd.read_sql_query(f'SELECT * FROM teams_{status[1]}_{table}', conn)
conn.close()
df2

team_shooting_df = df1.merge(df2, left_on='Squad', right_on='Squad', how='left')




# Teams Data - Matchday

# Used to calculate rolling xg and xa
db_path = os.path.join(current_directory + folder_db, "fbref_data_team_individual.db")
conn = sqlite3.connect(db_path)
df_individual = pd.read_sql_query('SELECT * FROM team_individual_tbl', conn)
conn.close()

# FPL Data
db_path = os.path.join(current_directory + folder_db, "fpl_data.db")
conn = sqlite3.connect(db_path)
df_fpl = pd.read_sql_query('SELECT * FROM fpl_data', conn)
conn.close()


# Fixtures data
def fixtures_data(team, window=4):
    # url = 'https://fbref.com/en/comps/9/schedule/Premier-League-Scores-and-Fixtures'

    # fixtures_df = utils.fetch_and_process_fbref_data(url, 0)

    db_path = os.path.join(current_directory + folder_db, "fixtures.db")
    conn = sqlite3.connect(db_path)
    fixtures_df = pd.read_sql_query('SELECT * FROM fixtures', conn)
    conn.close()

    data_path = os.path.join(current_directory + folder_raw)
    df = pd.read_csv(data_path + '/team_ranks.csv')
    df = df.sort_values('rank')

    team_mapping = {'Wolverhampton': 'Wolves'} #wolves is a specific case
    df['name'] = df['name'].replace(team_mapping)
    df['name'] = df['name'].apply(lambda x: utils.replace_with_fuzzy_match(x, fixtures_df['Home'].unique()))

    team_fixtures = fixtures_df[(fixtures_df['Home'] == team) | (fixtures_df['Away'] == team)]
    team_fixtures['team'] = team
    team_fixtures['opposition'] = np.where(team_fixtures['team'] == team_fixtures['Home'], 
                                            team_fixtures['Away'], team_fixtures['Home'])

    fixtures_merged = pd.merge(team_fixtures, df, how='left', left_on='opposition', right_on='name')

    # Compute the forward rolling average of the 'difficulty' column
    fixtures_merged['rank_rolling_avg'] = fixtures_merged['rank'].rolling(window).mean().shift(-window+1)

    # Manually adjust the last 4 entries
    for i in range(window-1):
        fixtures_merged.loc[len(fixtures_merged)-(i+1), 'rank_rolling_avg'] = fixtures_merged['rank'].iloc[-(i+1):].mean()


    # Compute the forward rolling sum of the 'difficulty' column
    fixtures_merged['rank_rolling_sum'] = fixtures_merged['rank'].rolling(window).sum().shift(-window+1)

    # Manually adjust the last 4 entries
    for i in range(window-1):
        fixtures_merged.loc[len(fixtures_merged)-(i+1), 'rank_rolling_sum'] = fixtures_merged['rank'].iloc[-(i+1):].sum()

    # Check the DataFrame
    return fixtures_merged


def fixtures_data_overall():
    # url = 'https://fbref.com/en/comps/9/schedule/Premier-League-Scores-and-Fixtures'
    # url = 'https://fbref.com/en/comps/9/schedule/Premier-League-Scores-and-Fixtures'

    db_path = os.path.join(current_directory + folder_db, "fixtures.db")
    conn = sqlite3.connect(db_path)
    fixtures_df = pd.read_sql_query('SELECT * FROM fixtures', conn)
    conn.close()

    data_path = os.path.join(current_directory + folder_raw)
    df = pd.read_csv(data_path + '/team_ranks.csv')

    team_mapping = {'Wolverhampton': 'Wolves'} #wolves is a specific case
    df['name'] = df['name'].replace(team_mapping)
    df['name'] = df['name'].apply(lambda x: utils.replace_with_fuzzy_match(x, fixtures_df['Home'].unique()))

    fixtures_merged = pd.merge(fixtures_df, df, how='left', left_on='Home', right_on='name')
    fixtures_merged = pd.merge(fixtures_merged, df, how='left', left_on='Away', right_on='name')
    return fixtures_merged

# Updates player names in fbref data to fpl names if they are there

# 
# players_2023_2024 = utils.update_positions(players_2023_2024, df_fpl)
# merged_df = utils.update_positions(merged_df, df_fpl)
# players_2021_2022 = utils.update_positions(players_2021_2022, df_fpl)
# players_2020_2021 = utils.update_positions(players_2020_2021, df_fpl)


# # df1 = merged_df
# df2 = players_2021_2022
# df3 = players_2020_2021

# # Create a Pandas Excel writer using XlsxWriter as the engine.
# writer = pd.ExcelWriter("fullyears_players.xlsx", engine="xlsxwriter")

# # Write each dataframe to a different worksheet.
# df1.to_excel(writer, sheet_name="22_23")
# df2.to_excel(writer, sheet_name="21_22")
# df3.to_excel(writer, sheet_name="20_22")

# # Close the Pandas Excel writer and output the Excel file.
# writer.close()