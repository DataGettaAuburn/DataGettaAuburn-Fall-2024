import pandas as pd
import numpy as np
import datetime
import os
import shutil
from tkinter import filedialog
from tkinter import Tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle

# Create a GUI for file dialog
root = Tk()
root.withdraw()
file_path = filedialog.askopenfilename()

# Load the data
df = pd.read_csv(file_path)

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

# Convert 'Date' column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Group "Fastball" and "FourSeamFastBall"
df['TaggedPitchType'] = df['TaggedPitchType'].replace(['Fastball', 'FourSeamFastBall', 'TwoSeamFastBall', 'Cutter'], 'GroupedFastball')

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

# Define the directory where you want to save the PDFs ***This will need to change once properly implemented***
base_dir = r"C:\\Users\\adamn\\Desktop\\Auburn Research\\KinaTrax\\PostGameTrackmanReports"

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
        # Calculate centroid and round to 3 decimal places
        centroid = current_pitcher_data.groupby('TaggedPitchType')[['HorzBreak', 'InducedVertBreak']].mean().round(3)

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