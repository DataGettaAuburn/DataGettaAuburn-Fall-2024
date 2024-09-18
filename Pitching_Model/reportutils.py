import pandas as pd
from heatmap import heatmap, div_heatmap, save_to_image


def gen_report_text(avg_vertical_breaks, avg_horizontal_breaks, avg_release_heights):
    """

    """
    # creates a text output summarizing information about pitchers, 
    # their pitch types, and average break measurements. 
    text_output = ""
    for (name, pitches), avg_vert_break in avg_vertical_breaks.items():
        avg_horz_break = avg_horizontal_breaks.get((name, pitches), "N/A")
        avg_release_height = avg_release_heights.get((name, pitches), "N/A")
        text_output += f"Pitcher: {name}, Pitch Type: {pitches}, Average Vertical Break: {avg_vert_break}, Avg Horizontal Break = {avg_horz_break}, Avg Release Height = {avg_release_height}\n"
    return text_output


def gen_report_images(df: pd.DataFrame, folder: str, name: str, p_type: str):
    """

    """

    # these are the ranges for the x and y axis. cuttoff were chosen
    # somewhat arbitrarily based on the observed spread of the data
    rx = (1, 3.5)
    ry = (-1.5, 1.5)

    # calculates the resolution of the heatmap. res is the maximum resolution
    # of either axis. resx and resy are the calculated resolutions for the x and y
    # where 
    # the aspect ratio is preserved.
    res = 20
    resx = int(res / max(rx[1] - rx[0], ry[1] - ry[0]) * (rx[1] - rx[0]))
    resy = int(res / max(rx[1] - rx[0], ry[1] - ry[0]) * (ry[1] - ry[0]))

    values = [(row['PlateLocHeight'], row['PlateLocSide'], row['PitchCall']) for index, row in df.iterrows()]
    values_all = [(row['PlateLocHeight'], row['PlateLocSide'], 1) for index, row in df.iterrows()]
    hmap = heatmap(values, resx, resy, spr=0.25, range_x=rx, range_y=ry)
    hmap_all = heatmap(values_all, resx, resy, spr=0.25, range_x=rx, range_y=ry)
    hmap_ratio = div_heatmap(hmap, hmap_all)

    save_to_image(hmap_all, folder + name + '_' + p_type + '_AllPitches', 'All Pitches', (rx[0], rx[1], ry[0], ry[1]))
    save_to_image(hmap, folder + name + '_' + p_type + '_SuccessfulPitches', 'Successful Pitches', (rx[0], rx[1], ry[0], ry[1]))
    save_to_image(hmap_ratio, folder + name + '_' + p_type + '_SuccessRatio', 'Success Ratio', (rx[0], rx[1], ry[0], ry[1]))

# takes the dataframe as the parameter in order to split the data by pitcher and later
# by pitch type
def gen_report_data(df: pd.DataFrame, folder: str, all_pitches_only: bool = False):
    """
    Generates all necessary data for the report

    df : the dataframe containing the entire dataset
    folder : location image files will be saved
    all_pitches_only : whether or not to include all types of pitches (helps with runtime)
    """

    avg_vertical_breaks = {}
    avg_horizontal_breaks = {}
    avg_release_heights = {}

    # loops through each unique pitcher on the 2024 roster
    names = ["Myers, Carson", "Bauman, Tanner", "Graves, Griffin", "Sofield, Drew", "Keplinger, Konner", "Copeland, Konner", "Crotchfelt, Zach", "Nelson, Drew", "Schorr, Ben", "Watts, Dylan", "Carlson, Parker", "Herberholz, Christian", "Cannon, Will", "McBride, Conner", "Tilly, Cam", "Armstrong, John", "Petrovic, Alex", "Gonzalez, Joseph", "Allsup, Chase", "Keshock, Cameron", "Murphy, Hayden"]

    for name in names:
        # makes sure the name is actually a string
        # there was a bug where the name was nan for some reason
        if not isinstance(name, str):
            continue

        # this was used to avoid having to redo everything after fixing the nan bug
        """
        # checks if the file already exists
        filename = folder + name + '_All_AllPitches'
        try:
            img = Image.open(filename + '.png')
            img.close()
            continue
        except FileNotFoundError:
            pass
        print(name)
        """

        # creates new dataset
        name_df = df[df['Pitcher'] == name]

        """
        # Gets (general) text output and saves to text file corresponding to their name
        text_output = gen_report_text()
        f = open(name + ".txt", "w")
        f.write(text_output)
        f.close()
        """

        for pitches in name_df['TaggedPitchType'].unique():
            pitch_df = name_df[name_df['TaggedPitchType'] == pitches]
            avg_vertical_break = pitch_df['InducedVertBreak'].mean()
            avg_horizontal_break = pitch_df['HorzBreak'].mean()
            avg_release_height = pitch_df['RelHeight'].mean()
            avg_vertical_breaks[(name, pitches)] = avg_vertical_break
            avg_horizontal_breaks[(name, pitches)] = avg_horizontal_break
            avg_release_heights[(name, pitches)] = avg_release_height

        gen_report_images(name_df, folder, name, 'All')

        if not all_pitches_only:
            # loops through all the kinds of pitches for that person
            pitch_type = name_df[('TaggedPitchType')].unique()
            for pitches in pitch_type:
                # creates new dataset
                pitch_df = name_df[name_df['TaggedPitchType'] == pitches]

                """
                avg_vertical_break = np.mean(pitch_df['VertBreak'])
                avg_vertical_breaks[(name, pitches)] = avg_vertical_break

                # gets pitch specific text output and saves to text file corresponding to their name and pitch
                text_output_pitch = gen_report_text(avg_vertical_breaks)
                f = open(name + pitches + ".txt", "w")
                f.write(text_output_pitch)
                f.close()
                """

                gen_report_images(pitch_df, folder, name, pitches)
    text_output = gen_report_text(avg_vertical_breaks, avg_horizontal_breaks, avg_release_heights)


def report_to_latex(report_name: str, folder: str, players: list[str], pitch_types: list[str]):
    """

    """

    f = open(report_name + ".tex", "w")

    s = """
    \\documentclass{article}
    \\usepackage[margin=0.5in]{geometry}
    \\usepackage{graphicx}

    \\title{Pitcher Report}

    \\begin{document}

    \\maketitle
    """
    f.write(s)
    for i, p in enumerate(players):
        # checks if p is a string
        if not isinstance(p, str):
            continue
        # LaTeX commands to the file handle
        f.write('\\newpage')
        s = f"""
        \\section*{{{p}}}
        \\begin{{figure}}[h!]
            \\centering
            \\includegraphics[width=0.32\\textwidth]{{{folder}/{p}_All_AllPitches.png}}
            \\includegraphics[width=0.32\\textwidth]{{{folder}/{p}_All_SuccessfulPitches.png}}
            \\includegraphics[width=0.32\\textwidth]{{{folder}/{p}_All_SuccessRatio.png}}
        \\end{{figure}}
        """
        f.write(s)
        for p_type in pitch_types:
            s = f"""
            \\begin{{figure}}[h!]
                \\centering
                \\includegraphics[width=0.32\\textwidth]{{{folder}/{p}_{p_type}_AllPitches.png}}
                \\includegraphics[width=0.32\\textwidth]{{{folder}/{p}_{p_type}_SuccessfulPitches.png}}
                \\includegraphics[width=0.32\\textwidth]{{{folder}/{p}_{p_type}_SuccessRatio.png}}
            \\end{{figure}}
            """
            f.write(s)
    f.write('\\end{document}')
    f.close()
