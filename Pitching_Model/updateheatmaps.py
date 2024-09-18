import psycopg2
import numpy as np

from heatmap import heatmap, normalize

# Reads in the connection info and connects to the database
# The connection info is stored in a file called db_info.txt
# db_info.txt should be in the same directory as this file
# For each argument needed for the psycopg2.connect() function
# db_info should contain line separated key value pairs in the format
# key=value
args = dict()
with open('db_info.txt', 'r') as f:
    for line in f:
        key, value = line.split('=')
        args[key] = value.strip()
con = psycopg2.connect(**args)
cur = con.cursor()

# these are the ranges for the x and y axis.
rx = (1, 3.5)
ry = (-1.5, 1.5)

# these are the ranges for the strike zone
strike_x = (-0.86, 0.86)
strike_y = (1.59, 3.13)

# calculates the resolution of the heatmap. res is the maximum resolution
# of either axis. resx and resy are the calculated resolutions for 
# the x and y where the aspect ratio is preserved. Currently with the
# given ranges and max res, the heatmaps are 20X16
res = 20
resx = int(res / max(rx[1] - rx[0], ry[1] - ry[0]) * (rx[1] - rx[0]))
resy = int(res / max(rx[1] - rx[0], ry[1] - ry[0]) * (ry[1] - ry[0]))

# Gets the names of all teams
cur.execute("""SELECT DISTINCT "PitcherTeam" FROM trackman_pitcher;""")
team_names = cur.fetchall()
team_names = [row[0] for row in team_names]
# team_names = ['AUB_TIG']  # Uncomment this to only run for the AU Team

pitch_types = ['Fastball', 'Slider', 'ChangeUp', 'FourSeamFastBall', 'Sinker', 'Curveball', 'Cutter', 'Splitter', 'TwoSeamFastBall']
success = ['StrikeCalled', 'StrikeSwinging', 'FoulBall']    # These are the pitch calls that are considered as successes

for team_name in team_names:
    team_name = team_name.replace("'", "''")
    # get pitcher names
    cur.execute(f"""SELECT DISTINCT "Pitcher" FROM trackman_pitcher WHERE "PitcherTeam" = '{team_name}';""")
    pitcher_names = cur.fetchall()
    pitcher_names = [row[0] for row in pitcher_names]
    for pitcher in pitcher_names:
        pitcher = pitcher.replace("'", "''")
        # The heatmaps for all pitch types combined are calculated as the sum of 
        # the heatmaps for each pitch type. This saves a bunch of time since every 
        # pitch doesn't need to be queried a second time for the all pitch types heatmap
        all_types_hmap = np.zeros((resy, resx))
        all_types_hmap_all = np.zeros((resy, resx))
        for pitch_type in pitch_types:
            try:
                cur.execute(f"""SELECT trackman_pitcher."PlateLocHeight", trackman_pitcher."PlateLocSide", trackman_metadata."PitchCall" 
                            FROM trackman_pitcher 
                            INNER JOIN trackman_metadata ON trackman_pitcher."PitchUID" = trackman_metadata."PitchUID"
                            WHERE trackman_pitcher."Pitcher" = '{pitcher}' 
                            AND trackman_pitcher."PitcherTeam" = '{team_name}'
                            AND trackman_pitcher."TaggedPitchType" = '{pitch_type}'
                            AND trackman_pitcher."PlateLocHeight" != 'NaN'
                            AND trackman_pitcher."PlateLocSide" != 'NaN';""")
                values = cur.fetchall()

                if len(values) == 0:
                    # print(f"No data for {pitcher} and {pitch_type}")
                    continue

                values = [(float(row[0]), float(row[1]), 1.0 if row[2] in success else 0.0) for row in values]
                values_all = [(row[0], row[1], 1) for row in values]

                hmap = heatmap(values, resx, resy, spr=0.55, range_x=rx, range_y=ry)
                hmap_all = heatmap(values_all, resx, resy, spr=0.55, range_x=rx, range_y=ry)
                hmap_ratio = hmap / (hmap_all + 1e-6)
                
                # adds to the all pitch types heatmaps
                all_types_hmap += hmap
                all_types_hmap_all += hmap_all

                # Normalizes, flattens, and converts the heatmaps to statndard python lists
                hmap = normalize(hmap).flatten().tolist()
                hmap_all = normalize(hmap_all).flatten().tolist()
                hmap_ratio = normalize(hmap_ratio).flatten().tolist()

                # Write the heatmaps to the database
                cur.execute(f"""INSERT INTO heatmap_model_values ("Pitcher", "PitcherTeam", "PitchType", "AllPitches", "SuccessfulPitches", "PitchRatio")
                                VALUES ('{pitcher}', '{team_name}', '{pitch_type}', ARRAY {hmap_all}::DECIMAL[], ARRAY {hmap}::DECIMAL[], ARRAY {hmap_ratio}::DECIMAL[])
                                ON CONFLICT ("Pitcher", "PitcherTeam", "PitchType") 
                                DO UPDATE SET "AllPitches" = EXCLUDED."AllPitches",
                                "SuccessfulPitches" = EXCLUDED."SuccessfulPitches",
                                "PitchRatio" = EXCLUDED."PitchRatio";""")
                con.commit()
            except Exception as e:
                print(e)
                print(f"Error with {pitcher} and {pitch_type}")
        
        all_types_hmap_ratio = all_types_hmap / (all_types_hmap_all + 1e-6)
        # Normalizes, flattens, and converts the heatmaps to standard python lists
        all_types_hmap = normalize(all_types_hmap).flatten().tolist()
        all_types_hmap_all = normalize(all_types_hmap_all).flatten().tolist()
        all_types_hmap_ratio = normalize(all_types_hmap_ratio).flatten().tolist()

        if sum(all_types_hmap_all) == 0:
            # print(f"No data for {pitcher} and AllPitchTypes")
            continue

        try:
            # Write the heatmaps for all pitch types to the database
            cur.execute(f"""INSERT INTO heatmap_model_values ("Pitcher", "PitcherTeam", "PitchType", "AllPitches", "SuccessfulPitches", "PitchRatio")
                            VALUES ('{pitcher}', '{team_name}', 'AllPitchTypes', ARRAY {all_types_hmap_all}::DECIMAL[], ARRAY {all_types_hmap}::DECIMAL[], ARRAY {all_types_hmap_ratio}::DECIMAL[])
                            ON CONFLICT ("Pitcher", "PitcherTeam", "PitchType") 
                            DO UPDATE SET "AllPitches" = EXCLUDED."AllPitches",
                            "SuccessfulPitches" = EXCLUDED."SuccessfulPitches",
                            "PitchRatio" = EXCLUDED."PitchRatio";""")
            con.commit()
        except Exception as e:
                print(e)
                print(f"Error with {pitcher} and AllPitchTypes")

# Close communication with the database
cur.close()
con.close()
