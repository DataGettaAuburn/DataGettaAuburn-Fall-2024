import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from heatmap import heatmap
from reportutils import gen_report_data, report_to_latex

## Read in data first and separate based on columns we will actually use
baseball_data = pd.read_csv('TrackMan_NoStuff_Master.csv')
baseball_data = baseball_data[['Pitcher', 'Batter', 'PitchCall', 'PlateLocHeight', 'PlateLocSide', 'TaggedPitchType', 'InducedVertBreak', 'HorzBreak', 'RelHeight']]
print(baseball_data.tail())

## Encode favorable vs unfavorable outcomes
baseball_data.loc[baseball_data["PitchCall"] == "StrikeCalled", "PitchCall"] = 1
baseball_data.loc[baseball_data["PitchCall"] == "StrikeSwinging", "PitchCall"] = 1
baseball_data.loc[baseball_data["PitchCall"] == "BallCalled", "PitchCall"] = 0
baseball_data.loc[baseball_data["PitchCall"] == "FoulBall", "PitchCall"] = 1
baseball_data.loc[baseball_data["PitchCall"] == "InPlay", "PitchCall"] = 0
baseball_data.loc[baseball_data["PitchCall"] == "HitByPitch", "PitchCall"] = 0
baseball_data.loc[baseball_data["PitchCall"] == "BallinDirt", "PitchCall"] = 0
baseball_data.loc[baseball_data["PitchCall"] == "Undefined", "PitchCall"] = 0
baseball_data.loc[baseball_data["PitchCall"] == "BallInDirt", "PitchCall"] = 0

## Get the boundaries for the heat maps
minh = min(baseball_data['PlateLocHeight'])
maxh = max(baseball_data['PlateLocHeight'])
minw = min(baseball_data['PlateLocSide'])
maxw = max(baseball_data['PlateLocSide'])
print('min H: ', minh)
print('max H: ', maxh)
print('min W: ', minw)
print('max W: ', maxw)

## Make heat maps for only people on roster and certain types of pitches
names = ["Myers, Carson", "Bauman, Tanner", "Graves, Griffin", "Sofield, Drew", "Keplinger, Konner", "Copeland, Konner", "Crotchfelt, Zach", "Nelson, Drew", "Schorr, Ben", "Watts, Dylan", "Carlson, Parker", "Herberholz, Christian", "Cannon, Will", "McBride, Conner", "Tilly, Cam", "Armstrong, John", "Petrovic, Alex", "Gonzalez, Joseph", "Allsup, Chase", "Keshock, Cameron", "Murphy, Hayden"]
pitch_types = ["Fastball", "Slider", "ChangeUp", "FourSeamFastBall", "Sinker", "Curveball", "Cutter", "Splitter", "TwoSeamFastBall"]

## Generate the report
gen_report_data(baseball_data, 'ImageFolder\\', True)
report_to_latex('testreport', 'ImageFolder', names, pitch_types)
