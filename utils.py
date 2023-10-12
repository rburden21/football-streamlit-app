import requests
import pandas as pd
from unidecode import unidecode
from fuzzywuzzy import fuzz, process
import re

def convert_to_years(age_str):
    try:
        # Split the string on hyphen and return the first part as integer
        return int(age_str.split('-')[0])
    except:
        return age_str

def fetch_and_process_fbref_data(url, type=2):
    # Fetch the data
    response = requests.get(url).text.replace('<!--', '').replace('-->', '')
    
    if type == 0: 
        header = 0 
    else: 
        header = 1

    # Parse the tables from the HTML response
    df = pd.read_html(response, header=header)[type]
    
    # Clean up the data
    if type == 2:
        df = df[~df['Player'].isin(['Player'])]
        df['Nation'] = df['Nation'].str.extract('([A-Z]{3})')
        df['Pos'] = df['Pos'].str.split(',').str[0]
        df = df.drop('Matches', axis=1)
        df.fillna(0, inplace=True)
        
        # Reset the index
        df = df.reset_index(drop=True)    
    return df

# Function to replace special characters using unidecode
def replace_special_characters(name):
    if isinstance(name, str):
        name = unidecode(name)
        
        # Additional manual replacements for characters not handled by unidecode
        manual_replacements = {
            'á': 'a', 'ä': 'a',
            'é': 'e', 'ë': 'e',
            'í': 'i', 'ï': 'i',
            'ó': 'o', 'ö': 'o',
            'ú': 'u', 'ü': 'u',
            'ý': 'y', 'ÿ': 'y',
            'Ø': 'O', 'ø': 'o',
            'š': 's', 'ç': 'c',
            # Add more replacements if necessary
        }
        
        for original, replacement in manual_replacements.items():
            name = name.replace(original, replacement)

    return name

def safe_unidecode(value):
    try:
        if isinstance(value, str):
            return unidecode(value)
    except:
        pass
    return value


def has_special_characters(name):
    return bool(re.search(r'[^\x00-\x7F]+', name))

def replace_with_fuzzy_match(original, choices, scorer=fuzz.token_sort_ratio):
    new_val, score = process.extractOne(original, choices, scorer=scorer)
    return new_val

def match_name_parts(row, df_fpl):
    player_name = row['Player']
    squad = row['Squad']
    original_position = row['Pos']  # Store the original position
    
    # Try to find a match in the final_fpl dataframe
    for _, fpl_row in df_fpl.iterrows():
        if fpl_row['name'] != squad:
            continue
        if player_name in fpl_row['full_name']:
            return fpl_row['pos_short']

    # If no match is found, return the original position
    return original_position

def update_positions(df, df_fpl):
    # Apply the name matching and updating process
    df['Pos'] = df.apply(lambda row: match_name_parts(row, df_fpl), axis=1)
    
    # Return the updated dataframe
    return df

def replace_special_chars(df):
    replacement_dict = {
        'Bénie Adama Traore': 'Benie Adama Traore',
        'Nathan Aké': 'Nathan Ake',
        'Miguel Almirón': 'Miguel Almiron',
        'Julián Álvarez': 'Julian Alvarez',
        'Saïd Benrahma': 'Said Benrahma',
        'Julio César Enciso': 'Julio Cesar Enciso',
        'Vladimír Coufal': 'Vladimir Coufal',
        'Odsonne Édouard': 'Odsonne Edouard',
        'Pervis Estupiñán': 'Pervis Estupinan',
        'Pascal Groß': 'Pascal Gross',
        'Marc Guéhi': 'Marc Guehi',
        'Bruno Guimarães': 'Bruno Guimaraes',
        'Joško Gvardiol': 'Josko Gvardiol',
        'Raúl Jiménez': 'Raul Jimenez',
        'João Pedro': 'Joao Pedro',
        'Issa Kaboré': 'Issa Kabore',
        'Cheikhou Kouyaté': 'Cheikhou Kouyate',
        'Mateo Kovačić': 'Mateo Kovacic',
        'Saša Lukić': 'Sasa Lukic',
        'Emiliano Martínez': 'Emiliano Martinez',
        'Aleksandar Mitrović': 'Aleksandar Mitrovic',
        'Lucas Paquetá': 'Lucas Paqueta',
        'Fabian Schär': 'Fabian Schar',
        'Tomáš Souček': 'Tomas Soucek',
        'Jurriën Timber': 'Jurrien Timber',
        'Joël Veltman': 'Joel Veltman',
        'Martin Ødegaard': 'Martin Odegaard',
        'Adrián': 'Adrian',
        'Sergio Agüero': 'Sergio Aguero',
        'Rayan Aït Nouri': 'Rayan Ait Nouri',
        'Thiago Alcántara': 'Thiago Alcantara',
        'Carlos Vinícius': 'Carlos Vinicius',
        'João Virgínia': 'Joao Virginia',
        'Matěj Vydra': 'Matej Vydra',
        'Okay Yokuşlu': 'Okay Yokuslu'
    }

    df['Player'] = df['Player'].replace(replacement_dict)
    return df