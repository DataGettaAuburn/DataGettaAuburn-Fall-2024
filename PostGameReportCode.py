#!/usr/bin/env python
# coding: utf-8

# Full Processing Code for TrackMan Reports
# First copy the most recent trackman file to the end of the TrackMan_NoStuff Document.

# Heres the Stuff+ code

# In[21]:


import pandas as pd
import numpy as np
import datetime
import os
import shutil
from tkinter import filedialog
from tkinter import Tk

# Create a GUI for file dialog
root = Tk()
root.withdraw()  # Hide the main window
file_path = filedialog.askopenfilename()  # Show the file open dialog and get the selected file path

# Load the data
df = pd.read_csv(file_path)


# In[22]:



# Rename the column
df = df.rename(columns={"Top/Bottom": "Top.Bottom"})

# Define today's date
today = datetime.date.today()
start_date = today.strftime('%Y-%m-%d')
end_date = start_date

# Define pitch colors
pitch_colors = {
    'Fastball': '#d22d49',
    'Fastballs': '#d22d49',
    'FourSeamFastBall': '#d22d49',
    'TwoSeamFastBall': '#de6a04',
    'Two-Seam': '#de6a04',
    'Sinker': '#fe9d00',
    'Cutter': '#933f2c',
    'Slider': '#eee716',
    'Split-Finger': '#3bacac',
    'Splitter': '#888888',
    'ChangeUp': '#1dbe3a',
    'Sweeper': '#ddb33a',
    'Curveball': '#00d1ed',
    'Other': '#888888'
}


# In[23]:



# Append the filtered data to the master CSV file
master = r"C:\Users\adamn\Desktop\TrackMan_NoStuff_Master.csv"

master_file = pd.read_csv(master)


# In[24]:



#df.to_csv(master, mode='a', index=False, header=False)


# In[25]:



# Move the original file to the specified directory
new_location = r"C:\Users\adamn\Desktop\All Trackman CSVs"
#shutil.move(file_path, os.path.join(new_location, os.path.basename(file_path)))

print("The selected CSV file has been appended to the master file and moved to the specified directory.")


# In[26]:


# Convert the 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'])
# Define the pitcher you want to filter by
filtered_pitcher = 'Watts, Dylan'  # replace with pitcher's name or 'all'

mask = ((df['PitcherTeam'].isin(['AUB_TIG', 'AUB_PRC', 'AUB'])))

# Use the mask to filter the dataframe
df = df.loc[mask]


# In[27]:


min_plate_x = -0.86
max_plate_x = 0.86
max_plate_z = 3.55
min_plate_z = 1.77

# In[28]:


df['nVAA'] = df['VertApprAngle'] - (-13.73 + (df['RelSpeed'] * 0.06312) + ((df['PlateLocHeight'] * 1.067)))


# In[29]:


from matplotlib.gridspec import GridSpec

# Create a unique identifier for each plate appearance
df['PlateAppearanceID'] = df['Date'].astype(str) + "_" + df['Pitcher'] + "_" + df['Top.Bottom'] + "_" + df['Inning'].astype(str) + "_" + df['PAofInning'].astype(str)

def two_three_success(group):
    if any(group.iloc[:3]['PitchCall'] == 'HitByPitch'):
        return "fail"
    elif len(group) < 4:
        if len(group) >= 3 and group.iloc[2]['Balls'] == 2 and group.iloc[2]['PitchCall'] == 'InPlay':
            return "fail"
        else:
            return "success"
    elif len(group) >= 4 and group.iloc[3]['Strikes'] == 2:
        return "success"
    else:
        return "fail"



def ab_eff_success(group):
    return "success" if len(group) <= 4 else "fail"

def in_zone(row):
    if min_plate_x <= row['PlateLocSide'] <= max_plate_x and min_plate_z <= row['PlateLocHeight'] <= max_plate_z:
        return 1
    else:
        return 0

# Apply the in_zone function to the dataframe
df['in_zone'] = df.apply(in_zone, axis=1)

# Group by 'PlateAppearanceID' and calculate 'ab_efficiency' and 'success_rate'
plate_appearance_grouped = df.groupby('PlateAppearanceID')
two_three_success = plate_appearance_grouped.apply(two_three_success).reset_index().rename(columns={0: 'two_three_success'})
ab_eff_success = plate_appearance_grouped.apply(ab_eff_success).reset_index().rename(columns={0: 'ab_eff_success'})

# Merge these results back into df
df = pd.merge(df, two_three_success, on='PlateAppearanceID', how='left')
df = pd.merge(df, ab_eff_success, on='PlateAppearanceID', how='left')


# Create a dictionary to map pitch types to 'Fastball'
pitch_type_dict = {'Splitter': 'ChangeUp', 'Fastball': 'Fastball', 'FourSeamFastBall': 'Fastball', 'TwoSeamFastBall': 'Fastball', 'Sinker': 'Fastball', 'Cutter': 'Fastball'}

# Map the 'TaggedPitchType' column using the dictionary
df['GeneralPitchType'] = df['TaggedPitchType'].map(pitch_type_dict).fillna(df['TaggedPitchType'])

# Initialize an empty DataFrame to store the metrics for each pitcher
metrics = pd.DataFrame()

# Iterate over each unique pitcher
for pitcher in df['Pitcher'].unique():
    # Filter the DataFrame for the current pitcher
    df_pitcher = df[df['Pitcher'] == pitcher]
    
    # Calculate the metrics for the current pitcher
    FPS = ((df_pitcher['PitchofPA'] == 2) & (df_pitcher['Strikes'] == 1)).mean() * 100
    
    # Calculate the number of plate appearances that ended in a walk
    walks = df_pitcher.groupby('PlateAppearanceID').apply(lambda group: 1 if group['KorBB'].iloc[-1] == 'Walk' else 0).sum()
    
    # Calculate the total number of plate appearances
    total_plate_appearances = df_pitcher['PlateAppearanceID'].nunique()
    
    # Calculate BB%
    BB_percentage = (walks / total_plate_appearances) * 100
    
    two_thirds = (df_pitcher.groupby('PlateAppearanceID')['two_three_success'].first() == "success").mean() * 100
    AB_Efficiency = (df_pitcher.groupby('PlateAppearanceID')['ab_eff_success'].first() == "success").mean() * 100
    
    Zone_percentage = df_pitcher[~((df_pitcher['Strikes'] == 2) & (df_pitcher['Balls'] <= 1))]['in_zone'].mean() * 100
    
    # Calculate Zone% for each pitch type for the current pitcher
    pitch_type_metrics = {}
    for pitch_type in df_pitcher['GeneralPitchType'].unique():
        pitch_type_metrics[f'{pitch_type}_Zone%'] = round(df_pitcher[(df_pitcher['GeneralPitchType'] == pitch_type) & ~((df_pitcher['Strikes'] == 2) & (df_pitcher['Balls'] <= 1))]['in_zone'].mean() * 100, 2)
    
    # Append the metrics for the current pitcher to the DataFrame
    metrics = metrics.append(pd.Series({'FPS': FPS, '2/3': two_thirds, 'AB Efficiency': AB_Efficiency, 'BB%': BB_percentage, 'Zone%': Zone_percentage, **pitch_type_metrics}, name=pitcher))

# Replace NaN values with '-'
metrics.fillna('-', inplace=True)

metrics = metrics.round(2)
# Sort the DataFrame by index
metrics = metrics.sort_index()



# In[30]:



# Apply the function to the dataframe
df['in_zone'] = df.apply(in_zone, axis=1)

# Group 'Fastball' and 'FourSeamFastball' together
df['TaggedPitchType'] = df['TaggedPitchType'].replace({'Fastball': 'Fastball', 'FourSeamFastBall': 'Fastball'})

df['IVB'] = df['InducedVertBreak']

# Calculate averages and round to 3 decimal places
grouped = df.groupby(['Pitcher', 'TaggedPitchType']).mean()[['RelSpeed', 'IVB', 'HorzBreak', 'SpinRate']].round(1)

# Calculate in_zone percentage for each pitch type for each pitcher
grouped_in_zone = (df.groupby(['Pitcher', 'TaggedPitchType'])['in_zone'].mean() * 100).round(0)

# Calculate min, max for each metric excluding in_zone
grouped_min = df.drop(columns='in_zone').groupby(['Pitcher', 'TaggedPitchType']).min()[['RelSpeed', 'IVB', 'HorzBreak', 'SpinRate']].round(1)
grouped_max = df.drop(columns='in_zone').groupby(['Pitcher', 'TaggedPitchType']).max()[['RelSpeed', 'IVB', 'HorzBreak', 'SpinRate']].round(1)

# Calculate averages for 'RelHeight' and 'Extension'
grouped_avg = df.groupby(['Pitcher', 'TaggedPitchType']).mean()[['RelHeight', 'Extension']].round(1)

# Filter dataframe for only fastballs
df_fastballs = df[df['TaggedPitchType'].isin(['Fastball', 'FourSeamFastBall', 'Sinker', 'TwoSeamFastBall'])]

# Define the zone based on 'PlateLocHeight'
def define_zone(row):
    if row['PlateLocHeight'] > 2.2:
        return 'upper'
    elif 1.9 <= row['PlateLocHeight'] <= 2.2:
        return 'middle'
    elif 0.0 <= row['PlateLocHeight'] < 1.9:
        return 'low'

df_fastballs['zone'] = df_fastballs.apply(define_zone, axis=1)

# Calculate average 'VertApprAngle' for each zone without considering the pitch type
grouped_vaa = df_fastballs.groupby(['Pitcher', 'zone']).mean()['VertApprAngle'].round(1)

# Unstack the multi-index dataframe to get each zone as a separate column
grouped_vaa = grouped_vaa.unstack(level=-1)

# Rename the columns
grouped_vaa.columns = ['VAA' + col for col in grouped_vaa.columns]

# Replace grouped_success_rate, grouped_ab_efficiency with the columns of 
# metrics dataframe and drop other columns from metrics dataframe.
metrics_final = metrics[['2/3','AB Efficiency']]
metrics_final.rename(columns={'2/3': 'success_rate','AB Efficiency': 'ab_efficiency'}, inplace=True)

# Combine all the grouped dataframes
grouped_final = pd.concat([grouped, grouped_min.add_suffix('_min'), grouped_max.add_suffix('_max'), grouped_avg, grouped_in_zone.rename('in_zone%'), metrics_final, grouped_vaa], axis=1)

# Rename the columns
grouped_final.rename(columns={'RelSpeed': 'Velo', 'HorzBreak': 'HB', 'SpinRate': 'Spin'}, inplace=True)
grouped_final.rename(columns={'RelSpeed_mean': 'Velo_mean', 
                              'RelSpeed_max': 'Velo_max',
                              'RelSpeed_min': 'Velo_min',
                              'HorzBreak_mean': 'HB_mean',
                              'HorzBreak_max': 'HB_max',
                              'HorzBreak_min': 'HB_min',
                              'SpinRate_mean': 'Spin_mean',
                              'SpinRate_max': 	'Spin_max',
                             	'SpinRate_min': 	'Spin_min'}, inplace=True)

# Get unique PitcherIds
pitcher_ids = df['Pitcher'].unique()

# Define the order of the columns
cols_order = [
   	'Velo',
   	'Velo_max',
   	'Velo_min',
   	'IVB',
   	'IVB_max',
   	'IVB_min',
   	'HB',
   	'HB_max',
   	'HB_min',
   	'Spin',
   	'Spin_max',
   	'Spin_min',
   	'in_zone%',
   	'RelHeight',
   	'Extension',
   	'success_rate',
   	'ab_efficiency',
   	'VAAupper',
   	'VAAmiddle',
   	'VAAlow'
]

grouped_final = grouped_final[cols_order]



# In[31]:


# Calculate the count of each TaggedPitchType for each Pitcher
pitch_type_counts = df.groupby(['Pitcher', 'TaggedPitchType']).size().reset_index(name='count')

# Set the index of pitch_type_counts to be the same as grouped_final
pitch_type_counts.set_index(['Pitcher', 'TaggedPitchType'], inplace=True)


# Ensure that the index names of grouped_final and pitch_type_counts are the same
grouped_final.index.names = pitch_type_counts.index.names

# Join the counts with the grouped_final DataFrame
grouped_final = pitch_type_counts.join(grouped_final)

# Reorder the columns to put 'counts' first
cols_order = ['count'] + [col for col in grouped_final.columns if col != 'count']
grouped_final = grouped_final[cols_order]

# Convert 'count' to integer
grouped_final['count'] = grouped_final['count'].round(0).astype(int)



# Makes the actual plots and save the pdf
# Make sure you change the pdf name to represent the actual day that these reports are from

# In[32]:


def home_plate_drawing(ax4):
    ax4.plot([-0.708, 0.708], [0.15, 0.15], color='black', linewidth=1)
    ax4.plot([-0.708, -0.708], [0.15, 0.3], color='black', linewidth=1)
    ax4.plot([-0.708, 0], [0.3, 0.5], color='black', linewidth=1)
    ax4.plot([0, 0.708], [0.5, 0.3], color='black', linewidth=1)
    ax4.plot([0.708, 0.708], [0.3, 0.15], color='black', linewidth=1)

# In[33]:


zone_PDF_Path = fr"C:\Users\adamn\Desktop\Auburn Research\KinaTrax\PostGameTrackmanReports\0.Grouped\Zone_Percent_Chart_{end_date}.pdf"

pdf_path = fr"C:\Users\adamn\Desktop\Auburn Research\KinaTrax\PostGameTrackmanReports\0.Grouped\pitcher_plots_{end_date}.pdf"


# In[34]:


# Create a new column 'PitchCount' that represents the pitch count for each pitcher
df['PitchCount'] = df.groupby(['Pitcher','TaggedPitchType']).cumcount() + 1


# In[35]:


# Define the pitch calls that indicate a swing
swing_calls = ['StrikeSwinging', 'InPlay', 'FoulBallNotFieldable', 'FoulBall', 'FoulBallFieldable']

# Create a new 'Swing' column
df['Swing'] = df['PitchCall'].isin(swing_calls)


# In[36]:


from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle

with PdfPages(pdf_path) as pdf:
    # Loop over each pitcher
    for id in pitcher_ids:
        # Filter data for current PitcherId
        current_pitcher_data = df[df['Pitcher'] == id]
        
        # Calculate centroid and round to 3 decimal places
        centroid = current_pitcher_data.groupby('TaggedPitchType')[['HorzBreak', 'InducedVertBreak']].mean().round(3)
        
        # Get pitcher name
        pitcher_name = current_pitcher_data['Pitcher'].iloc[0]

        # Get date
        pitch_date = current_pitcher_data['Date'].iloc[0].date()
        
        # Create a new figure with a custom layout of subplots
        fig = plt.figure(figsize=(22, 17))
        gs = GridSpec(4, 4, figure=fig, height_ratios=[.05, .75, 1 , 3.25])
        # Adjust the spacing between subplots
        plt.subplots_adjust(hspace=0.3)
        
        
        # Add the pitcher's name and date at the top left of the page
        ax0 = fig.add_subplot(gs[0, 0])
        ax0.text(0.5, 0.5, f"{pitcher_name} - {pitch_date}", ha='center', va='center', fontsize=14)
        ax0.axis('off')
        
        # Print efficiency and 2 out of 3 success rate at the top right of the page

        success_rate = round(metrics.loc[id, '2/3'], 1).item()
        ab_efficiency = round(metrics.loc[id, 'AB Efficiency'], 1).item()

        # Check if 'VAAupper', 'VAAmiddle', and 'VAAlow' exist before trying to access them
        vaa_upper = grouped_vaa.loc[id]['VAAupper'].item() if 'VAAupper' in grouped_vaa.columns else 'N/A'
        vaa_middle = grouped_vaa.loc[id]['VAAmiddle'].item() if 'VAAmiddle' in grouped_vaa.columns else 'N/A'
        vaa_low = grouped_vaa.loc[id]['VAAlow'].item() if 'VAAlow' in grouped_vaa.columns else 'N/A'

        #ax1 = fig.add_subplot(gs[0, 3])
        #ax1.text(0.5, 0.5, f" 2 out of 3 Success Rate: {success_rate}%\nAB Efficiency: {ab_efficiency}%\nVAA_upper: {vaa_upper}\nVAA_middle: {vaa_middle}\nVAA_lower: {vaa_low}", ha='center', va='center', fontsize=10)
        #ax1.axis('off')

        # Create a table in the second row spanning both columns
        table_data = grouped_final.drop(columns=['success_rate', 'ab_efficiency', 'VAAupper', 'VAAmiddle', 'VAAlow']).loc[id]
        
        ax2 = fig.add_subplot(gs[1, :])
        table = ax2.table(cellText=table_data.values, colLabels=table_data.columns, rowLabels=table_data.index, cellLoc='center', loc='center')
        
        # Set the font size of the table
        table.auto_set_font_size(False)
        table.set_fontsize(10)

        # Scale the table to take up the whole width of the page
        table.scale(1, 1.5)

         # Turn off axis for table subplot
        ax2.axis('off')


        # Create the second subplot (Pitch Movement Plot)
        ax3 = fig.add_subplot(gs[3, 0:2])
        sns.scatterplot(data=current_pitcher_data, x='HorzBreak', y='InducedVertBreak', s=120, hue='TaggedPitchType', palette=pitch_colors, ax=ax3)
        ax3.set_title('Pitch Movement Plot')
        ax3.set_xlabel('Horizontal Break (in)')
        ax3.set_ylabel('Induced Vertical Break (in)')
        ax3.set_xlim(-25, 25)
        ax3.set_ylim(-25, 25)
        # Add axes lines at the origin 
        ax3.axhline(0, color='black', linewidth=0.5)
        ax3.axvline(0, color='black', linewidth=0.5)

        # Add centroid points to the second subplot
        for i in range(len(centroid)):
            ax3.scatter(centroid.iloc[i, 0], centroid.iloc[i, 1], color=pitch_colors[centroid.index[i]], s=150, marker='x')

        # Create the third subplot (Pitch Location)
        markers = {"StrikeCalled": "o", "StrikeSwinging": "v", "BallCalled": "^", "HitByPitch": "<", "FoulBallNotFieldable": ">", "FoulBallFieldable": ">", "FoulBall": ">", "InPlay": "s", "BallinDirt": "D", "BallInDirt": "D"}
        ax4 = fig.add_subplot(gs[3, 2:4])
        sns.scatterplot(data=current_pitcher_data, x='PlateLocSide', y='PlateLocHeight', s=120, style='PitchCall', markers=markers, hue='TaggedPitchType', palette=pitch_colors, ax=ax4)
        ax4.set_title('Pitch Location (Pitcher View)')
        ax4.set_xlabel('Plate Side (ft)')
        ax4.set_ylabel('Plate Height (ft)')
        ax4.set_xlim(-3, 3)
        ax4.set_ylim(0, 6)
        home_plate_drawing(ax4)

        # Add a rectangle to the third subplot
        rectangle = Rectangle((min_plate_x, min_plate_z), max_plate_x - min_plate_x, max_plate_z - min_plate_z, fill=False)
        ax4.add_patch(rectangle)

        # Create the fourth subplot (Velocity by Pitch Type)
        ax5 = fig.add_subplot(gs[2, 0:2])
        sns.lineplot(data=current_pitcher_data, x='PitchCount', y='RelSpeed', hue='TaggedPitchType', palette=pitch_colors, ax=ax5)
        ax5.set_title('Velocity by Pitch Type')
        ax5.set_xlabel('Pitch Number')
        ax5.set_ylabel('Velo')

        # Filter data for left-handed and right-handed batters
        left_batters = current_pitcher_data[current_pitcher_data['BatterSide'] == 'Left']
        right_batters = current_pitcher_data[current_pitcher_data['BatterSide'] == 'Right']

        # Create the fifth subplot (Pitch Type Pie Chart for Left-Handed Batters)
        ax6 = fig.add_subplot(gs[2, 2])
        # Get colors for each pitch type
        left_pitch_usage = left_batters['TaggedPitchType'].value_counts()
        left_colors = [pitch_colors[pitch_type] for pitch_type in left_pitch_usage.index]
        ax6.pie(left_pitch_usage, labels=left_pitch_usage.index, colors=left_colors, autopct='%1.1f%%', startangle=140)
        ax6.set_title('Pitch Usage Percentage (Left-Handed Batters)')

        # Create the sixth subplot (Pitch Type Pie Chart for Right-Handed Batters)
        ax7 = fig.add_subplot(gs[2,3])
        # Get colors for each pitch type
        right_pitch_usage = right_batters['TaggedPitchType'].value_counts()
        right_colors = [pitch_colors[pitch_type] for pitch_type in right_pitch_usage.index]
        ax7.pie(right_pitch_usage, labels=right_pitch_usage.index, colors=right_colors, autopct='%1.1f%%', startangle=140)
        ax7.set_title('Pitch Usage Percentage (Right-Handed Batters)')


        # Calculate the values for the new table
        total_pitches = len(current_pitcher_data)
        swing_pitches = len(current_pitcher_data[current_pitcher_data['Swing'] == 1])
        first_pitches = current_pitcher_data[current_pitcher_data['PitchofPA'] == 1]
        all_first_pitches = len(first_pitches)


        # Define the condition for a successful first pitch
        conditions = (
            # If the PitchCall is not 'BallCalled' or 'InPlay'
            ((first_pitches['PitchCall'] != 'BallCalled') & (first_pitches['PitchCall'] != 'InPlay')) |
            # Or if the PitchCall is 'InPlay' and the PlayResult is 'Out'
            ((first_pitches['PitchCall'] == 'InPlay') & (first_pitches['PlayResult'] == 'Out'))
        )



        # Apply the conditions to the dataframe
        successful_first_pitches = first_pitches[conditions]

        # Get the number of successful first pitches
        first_pitch_strikes = len(successful_first_pitches)


        fps_percent = round((first_pitch_strikes/all_first_pitches)*100,1)




        iz_pitches = len(current_pitcher_data[current_pitcher_data['in_zone'] == 1])
        oz_pitches = len(current_pitcher_data[current_pitcher_data['in_zone'] == 0])
        iz_swings = len(current_pitcher_data[(current_pitcher_data['in_zone'] == 1) & (current_pitcher_data['Swing'] == 1)])
        iz_whiffs = len(current_pitcher_data[(current_pitcher_data['in_zone'] == 1) & (current_pitcher_data['PitchCall'] == 'StrikeSwinging')])
        whiff_percent = round(len(current_pitcher_data[current_pitcher_data['PitchCall'] == 'StrikeSwinging']) / swing_pitches * 100, 1)

        # Correct calculation for In-Zone Whiff%
        if iz_swings > 0:
            iz_whiff_percent = round(iz_whiffs / iz_swings * 100, 1)
        else:
            iz_whiff_percent = 'NA'

        chase_percent = round(len(current_pitcher_data[(current_pitcher_data['Swing'] == 1) & (current_pitcher_data['in_zone'] == 0)]) / oz_pitches * 100, 1)

        pitch_types = ['Fastball', 'FourSeamFastBall', 'TwoSeamFastBall', 'Sinker']
        avg_nVAA = round(current_pitcher_data[current_pitcher_data['TaggedPitchType'].isin(pitch_types)]['nVAA'].mean(),3)


        # Create a DataFrame for the new table
        new_table_data = pd.DataFrame({'Total Pitches': [total_pitches], 'AB Efficency': [ab_efficiency], 'FPS%': [fps_percent],  '2 of 3 Success': [success_rate] ,
        'Whiff%': [whiff_percent], 'IZ Whiff%': [iz_whiff_percent], 'Chase%': [chase_percent],'FB nVAA': [avg_nVAA],'FB VAA Upper': [vaa_upper], 'FB VAA Mid': [vaa_middle], 'FB VAA Lower': [vaa_low],})

        # Create a new table in the third row spanning both columns
        ax8 = fig.add_subplot(gs[0, 1:4])
        new_table = ax8.table(cellText=new_table_data.values, colLabels=new_table_data.columns, cellLoc='center', loc='center')

        # Set the font size of the new table
        new_table.auto_set_font_size(False)
        new_table.set_fontsize(10)

        # Scale the new table to take up the whole width of the page
        new_table.scale(1, 1.5)

        # Turn off axis for new table subplot
        ax8.axis('off')




        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)





# In[37]:


import os

# Define the directory where you want to save the PDFs
base_dir = r"C:\Users\adamn\Desktop\Auburn Research\KinaTrax\PostGameTrackmanReports"

# Loop over each pitcher
for id in pitcher_ids:
    # Filter data for current PitcherId
    current_pitcher_data = df[df['Pitcher'] == id]
    
    # Get pitcher name
    pitcher_name = current_pitcher_data['Pitcher'].iloc[0]

    # Get date
    pitch_date = current_pitcher_data['Date'].iloc[0].date()

    # Create a directory for the pitcher if it doesn't exist
    pitcher_dir = os.path.join(base_dir, pitcher_name)
    os.makedirs(pitcher_dir, exist_ok=True)

    # Define the path for the PDF
    pdf_path = os.path.join(pitcher_dir, f"{pitcher_name}_{pitch_date}.pdf")

    with PdfPages(pdf_path) as pdf:
            # Filter data for current PitcherId
            current_pitcher_data = df[df['Pitcher'] == id]
            
            # Calculate centroid and round to 3 decimal places
            centroid = current_pitcher_data.groupby('TaggedPitchType')[['HorzBreak', 'InducedVertBreak']].mean().round(3)
            
            # Get pitcher name
            pitcher_name = current_pitcher_data['Pitcher'].iloc[0]

            # Get date
            pitch_date = current_pitcher_data['Date'].iloc[0].date()
            
            # Create a new figure with a custom layout of subplots
            fig = plt.figure(figsize=(22, 17))
            gs = GridSpec(4, 4, figure=fig, height_ratios=[.05, .75, 1 , 3.25])
            # Adjust the spacing between subplots
            plt.subplots_adjust(hspace=0.3)
            
            
            # Add the pitcher's name and date at the top left of the page
            ax0 = fig.add_subplot(gs[0, 0])
            ax0.text(0.5, 0.5, f"{pitcher_name} - {pitch_date}", ha='center', va='center', fontsize=14)
            ax0.axis('off')
            
            # Print efficiency and 2 out of 3 success rate at the top right of the page

            success_rate = round(metrics.loc[id, '2/3'], 1).item()
            ab_efficiency = round(metrics.loc[id, 'AB Efficiency'], 1).item()

            # Check if 'VAAupper', 'VAAmiddle', and 'VAAlow' exist before trying to access them
            vaa_upper = grouped_vaa.loc[id]['VAAupper'].item() if 'VAAupper' in grouped_vaa.columns else 'N/A'
            vaa_middle = grouped_vaa.loc[id]['VAAmiddle'].item() if 'VAAmiddle' in grouped_vaa.columns else 'N/A'
            vaa_low = grouped_vaa.loc[id]['VAAlow'].item() if 'VAAlow' in grouped_vaa.columns else 'N/A'

            #ax1 = fig.add_subplot(gs[0, 3])
            #ax1.text(0.5, 0.5, f" 2 out of 3 Success Rate: {success_rate}%\nAB Efficiency: {ab_efficiency}%\nVAA_upper: {vaa_upper}\nVAA_middle: {vaa_middle}\nVAA_lower: {vaa_low}", ha='center', va='center', fontsize=10)
            #ax1.axis('off')

            # Create a table in the second row spanning both columns
            table_data = grouped_final.drop(columns=['success_rate', 'ab_efficiency', 'VAAupper', 'VAAmiddle', 'VAAlow']).loc[id]
            
            ax2 = fig.add_subplot(gs[1, :])
            table = ax2.table(cellText=table_data.values, colLabels=table_data.columns, rowLabels=table_data.index, cellLoc='center', loc='center')
            
            # Set the font size of the table
            table.auto_set_font_size(False)
            table.set_fontsize(10)

            # Scale the table to take up the whole width of the page
            table.scale(1, 1.5)

            # Turn off axis for table subplot
            ax2.axis('off')


            # Create the second subplot (Pitch Movement Plot)
            ax3 = fig.add_subplot(gs[3, 0:2])
            sns.scatterplot(data=current_pitcher_data, x='HorzBreak', y='InducedVertBreak', s=120, hue='TaggedPitchType', palette=pitch_colors, ax=ax3)
            ax3.set_title('Pitch Movement Plot')
            ax3.set_xlabel('Horizontal Break (in)')
            ax3.set_ylabel('Induced Vertical Break (in)')
            ax3.set_xlim(-25, 25)
            ax3.set_ylim(-25, 25)
            # Add axes lines at the origin 
            ax3.axhline(0, color='black', linewidth=0.5)
            ax3.axvline(0, color='black', linewidth=0.5)

            # Add centroid points to the second subplot
            for i in range(len(centroid)):
                ax3.scatter(centroid.iloc[i, 0], centroid.iloc[i, 1], color=pitch_colors[centroid.index[i]], s=150, marker='x')

            # Create the third subplot (Pitch Location)
            markers = {"StrikeCalled": "o", "StrikeSwinging": "v", "BallCalled": "^", "HitByPitch": "<", "FoulBallNotFieldable": ">", "FoulBallFieldable": ">", "FoulBall": ">", "InPlay": "s", "BallinDirt": "D", "BallInDirt": "D"}
            ax4 = fig.add_subplot(gs[3, 2:4])
            sns.scatterplot(data=current_pitcher_data, x='PlateLocSide', y='PlateLocHeight', s=120, style='PitchCall', markers=markers, hue='TaggedPitchType', palette=pitch_colors, ax=ax4)
            ax4.set_title('Pitch Location (Pitcher View)')
            ax4.set_xlabel('Plate Side (ft)')
            ax4.set_ylabel('Plate Height (ft)')
            ax4.set_xlim(-3, 3)
            ax4.set_ylim(0, 6)
            home_plate_drawing(ax4)

            # Add a rectangle to the third subplot
            rectangle = Rectangle((min_plate_x, min_plate_z), max_plate_x - min_plate_x, max_plate_z - min_plate_z, fill=False)
            ax4.add_patch(rectangle)

            # Create the fourth subplot (Velocity by Pitch Type)
            ax5 = fig.add_subplot(gs[2, 0:2])
            sns.lineplot(data=current_pitcher_data, x='PitchCount', y='RelSpeed', hue='TaggedPitchType', palette=pitch_colors, ax=ax5)
            ax5.set_title('Velocity by Pitch Type')
            ax5.set_xlabel('Pitch Number')
            ax5.set_ylabel('Velo')

            # Filter data for left-handed and right-handed batters
            left_batters = current_pitcher_data[current_pitcher_data['BatterSide'] == 'Left']
            right_batters = current_pitcher_data[current_pitcher_data['BatterSide'] == 'Right']

            # Create the fifth subplot (Pitch Type Pie Chart for Left-Handed Batters)
            ax6 = fig.add_subplot(gs[2, 2])
            # Get colors for each pitch type
            left_pitch_usage = left_batters['TaggedPitchType'].value_counts()
            left_colors = [pitch_colors[pitch_type] for pitch_type in left_pitch_usage.index]
            ax6.pie(left_pitch_usage, labels=left_pitch_usage.index, colors=left_colors, autopct='%1.1f%%', startangle=140)
            ax6.set_title('Pitch Usage Percentage (Left-Handed Batters)')

            # Create the sixth subplot (Pitch Type Pie Chart for Right-Handed Batters)
            ax7 = fig.add_subplot(gs[2,3])
            # Get colors for each pitch type
            right_pitch_usage = right_batters['TaggedPitchType'].value_counts()
            right_colors = [pitch_colors[pitch_type] for pitch_type in right_pitch_usage.index]
            ax7.pie(right_pitch_usage, labels=right_pitch_usage.index, colors=right_colors, autopct='%1.1f%%', startangle=140)
            ax7.set_title('Pitch Usage Percentage (Right-Handed Batters)')


            # Calculate the values for the new table
            total_pitches = len(current_pitcher_data)
            swing_pitches = len(current_pitcher_data[current_pitcher_data['Swing'] == 1])
            first_pitches = current_pitcher_data[current_pitcher_data['PitchofPA'] == 1]
            all_first_pitches = len(first_pitches)


            # Define the condition for a successful first pitch
            conditions = (
                # If the PitchCall is not 'BallCalled' or 'InPlay'
                ((first_pitches['PitchCall'] != 'BallCalled') & (first_pitches['PitchCall'] != 'InPlay')) |
                # Or if the PitchCall is 'InPlay' and the PlayResult is 'Out'
                ((first_pitches['PitchCall'] == 'InPlay') & (first_pitches['PlayResult'] == 'Out'))
            )



            # Apply the conditions to the dataframe
            successful_first_pitches = first_pitches[conditions]

            # Get the number of successful first pitches
            first_pitch_strikes = len(successful_first_pitches)


            fps_percent = round((first_pitch_strikes/all_first_pitches)*100,1)




            iz_pitches = len(current_pitcher_data[current_pitcher_data['in_zone'] == 1])
            oz_pitches = len(current_pitcher_data[current_pitcher_data['in_zone'] == 0])
            iz_swings = len(current_pitcher_data[(current_pitcher_data['in_zone'] == 1) & (current_pitcher_data['Swing'] == 1)])
            iz_whiffs = len(current_pitcher_data[(current_pitcher_data['in_zone'] == 1) & (current_pitcher_data['PitchCall'] == 'StrikeSwinging')])
            whiff_percent = round(len(current_pitcher_data[current_pitcher_data['PitchCall'] == 'StrikeSwinging']) / swing_pitches * 100, 1)

            # Correct calculation for In-Zone Whiff%
            if iz_swings > 0:
                iz_whiff_percent = round(iz_whiffs / iz_swings * 100, 1)
            else:
                iz_whiff_percent = 'NA'

            chase_percent = round(len(current_pitcher_data[(current_pitcher_data['Swing'] == 1) & (current_pitcher_data['in_zone'] == 0)]) / oz_pitches * 100, 1)

            pitch_types = ['Fastball', 'FourSeamFastBall', 'TwoSeamFastBall', 'Sinker']
            avg_nVAA = round(current_pitcher_data[current_pitcher_data['TaggedPitchType'].isin(pitch_types)]['nVAA'].mean(),3)


            # Create a DataFrame for the new table
            new_table_data = pd.DataFrame({'Total Pitches': [total_pitches], 'AB Efficency': [ab_efficiency], 'FPS%': [fps_percent],  '2 of 3 Success': [success_rate] ,
            'Whiff%': [whiff_percent], 'IZ Whiff%': [iz_whiff_percent], 'Chase%': [chase_percent],'FB nVAA': [avg_nVAA],'FB VAA Upper': [vaa_upper], 'FB VAA Mid': [vaa_middle], 'FB VAA Lower': [vaa_low],})

            # Create a new table in the third row spanning both columns
            ax8 = fig.add_subplot(gs[0, 1:4])
            new_table = ax8.table(cellText=new_table_data.values, colLabels=new_table_data.columns, cellLoc='center', loc='center')

            # Set the font size of the new table
            new_table.auto_set_font_size(False)
            new_table.set_fontsize(10)

            # Scale the new table to take up the whole width of the page
            new_table.scale(1, 1.5)

            # Turn off axis for new table subplot
            ax8.axis('off')




            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)


# Start the zone rates pdf

# In[38]:


import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# Convert 'Date' column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Group "Fastball" and "FourSeamFastBall"
df['TaggedPitchType'] = df['TaggedPitchType'].replace(['Fastball', 'FourSeamFastBall', 'TwoSeamFastBall','Cutter'], 'GroupedFastball')

def in_zone(row):
    if min_plate_x <= row['PlateLocSide'] <= max_plate_x and min_plate_z <= row['PlateLocHeight'] <= max_plate_z:
        return 1
    else:
        return 0

df = df[~((df['Strikes'] == 2) & (df['Balls'] <= 1))]
# Apply the function to the dataframe
df['in_zone'] = df.apply(in_zone, axis=1)

# Separate by 'Pitcher'
df_pitcher = df.groupby('Pitcher')

# Separate by 'TaggedPitchType'
df_pitch_type = df.groupby('TaggedPitchType')

# Make sure to change the name of these PDFs

# In[39]:



# Create a PdfPages object for the plots
with PdfPages(zone_PDF_Path) as pdf:

    fig = plt.figure(figsize=(11, 8.5))

    # Add a subplot for the text
    ax_text = plt.subplot2grid((4, 2), (0, 0), colspan=2)

    # Get min and max date
    min_date = df['Date'].min()
    max_date = df['Date'].max()

    # Check if there is only one unique date in the dataframe
    if min_date == max_date:
        date_text = f"Date: {min_date.strftime('%Y-%m-%d')}"
    else:
        date_text = f"Dates: {min_date.strftime('%Y-%m-%d')} - {max_date.strftime('%Y-%m-%d')}"

    # Add the print statements to the figure
    text_to_add = f"{date_text}\n" \
                  f"The overall in zone % in pitcher or even counts is {df['in_zone'].mean() * 100:.2f}\n" \
                  f"The zone % for fastballs is {df[df['TaggedPitchType'] == 'GroupedFastball']['in_zone'].mean() * 100:.2f}\n" \
                  f"The pitcher with the lowest zone % is {(df_pitcher['in_zone'].mean() * 100).idxmin()} with a zone % of {(df_pitcher['in_zone'].mean() * 100).min():.2f}\n" \
                  f"The pitcher with the highest zone % is {(df_pitcher['in_zone'].mean() * 100).idxmax()} with a zone % of {(df_pitcher['in_zone'].mean() * 100).max():.2f}"

    ax_text.text(0.5, 0.5, text_to_add, horizontalalignment='center', verticalalignment='center', fontsize=12)
    ax_text.axis('off')

    # Add a subplot for the first plot
    ax_plot1 = plt.subplot2grid((4, 2), (1, 0), rowspan=3)

    # Plot percentage of 'in_zone' by 'Pitcher'
    (df_pitcher['in_zone'].mean() * 100).plot(kind='barh', ax=ax_plot1)
    ax_plot1.set_title('Percentage of Pitches In Zone by Pitcher')
    ax_plot1.set_xlabel('Percentage')
    ax_plot1.set_ylabel('')

    # Add a subplot for the second plot
    ax_plot2 = plt.subplot2grid((4, 2), (1, 1), rowspan=3)

    # Plot percentage of 'in_zone' by 'TaggedPitchType'
    (df_pitch_type['in_zone'].mean() * 100).plot(kind='barh', ax=ax_plot2)
    ax_plot2.set_title('Percentage of Pitches In Zone by TaggedPitchType')
    ax_plot2.set_xlabel('Percentage')
    ax_plot2.set_ylabel('')

    plt.tight_layout()
    
    pdf.savefig(fig, orientation='landscape')
    plt.close(fig)

