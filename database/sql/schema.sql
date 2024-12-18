-- Creates the schema for the database
CREATE TABLE IF NOT EXISTS "players" (
  "PlayerName" varchar,
  "TeamName" varchar,
  PRIMARY KEY ("PlayerName", "TeamName")
);

CREATE TABLE IF NOT EXISTS "teams" (
  "TeamName" varchar UNIQUE PRIMARY KEY,
  "DisplayName" varchar,
  "Conference" varchar
);

CREATE TABLE IF NOT EXISTS "conferences" (
  "ConferenceName" varchar UNIQUE PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS "trackman_metadata" (
  "PitchUID" uuid UNIQUE PRIMARY KEY DEFAULT (uuid_generate_v4()),
  "GameDate" date,
  "PitchTime" time,
  "Inning" int,
  "TopBottom" varchar,
  "Outs" int,
  "Balls" int,
  "Strikes" int,
  "PitchCall" varchar,
  "KorBB" varchar,
  "TaggedHitType" varchar,
  "PlayResult" varchar,
  "OutsOnPlay" varchar,
  "RunsScored" varchar,
  "RunnersAt" varchar,
  "HomeTeam" varchar,
  "AwayTeam" varchar,
  "Stadium" varchar,
  "Level" varchar,
  "League" varchar,
  "GameID" varchar,
  "GameUID" varchar,
  "UTCDate" date,
  "UTCtime" time,
  "LocalDateTime" date,
  "UTCDateTime" date,
  "AutoHitType" varchar,
  "System" varchar,
  "HomeTeamForeignID" varchar,
  "AwayTeamForeignID" varchar,
  "GameForeignID" varchar,
  "PlayID" varchar
);

CREATE TABLE IF NOT EXISTS "trackman_pitcher" (
  "PitchUID" uuid UNIQUE PRIMARY KEY DEFAULT (uuid_generate_v4()),
  "PitchNo" int,
  "PAofInning" int,
  "PitchofPA" int,
  "Pitcher" varchar,
  "PitcherID" int,
  "PitcherThrows" varchar,
  "PitcherTeam" varchar,
  "PitcherSet" varchar,
  "TaggedPitchType" varchar,
  "AutoPitchType" varchar,
  "RelSpeed" decimal,
  "VertRelAngle" decimal,
  "HorzRelAngle" decimal,
  "SpinRate" decimal,
  "SpinAxis" decimal,
  "Tilt" varchar,
  "RelHeight" decimal,
  "RelSide" decimal,
  "Extension" decimal,
  "VertBreak" decimal,
  "InducedVert" decimal,
  "HorzBreak" decimal,
  "PlateLocHeight" decimal,
  "PlateLocSide" decimal,
  "ZoneSpeed" decimal,
  "VertApprAngle" decimal,
  "HorzApprAngle" decimal,
  "ZoneTime" decimal,
  "pfxx" decimal,
  "pfxz" decimal,
  "x0" decimal,
  "y0" decimal,
  "z0" decimal,
  "vx0" decimal,
  "vy0" decimal,
  "vz0" decimal,
  "ax0" decimal,
  "ay0" decimal,
  "az0" decimal,
  "SpeedDrop" decimal,
  "PitchLastMeasuredX" decimal,
  "PitchLastMeasuredY" decimal,
  "PitchLastMeasuredZ" decimal,
  "PitchTrajectoryXc0" decimal,
  "PitchTrajectoryXc1" decimal,
  "PitchTrajectoryXc2" decimal,
  "PitchTrajectoryYc0" decimal,
  "PitchTrajectoryYc1" decimal,
  "PitchTrajectoryYc2" decimal,
  "PitchTrajectoryZc0" decimal,
  "PitchTrajectoryZc1" decimal,
  "PitchTrajectoryZc2" decimal,
  "PitchReleaseConfidence" varchar,
  "PitchLocationConfidence" varchar,
  "PitchMovementConfidence" varchar
);


CREATE TABLE IF NOT EXISTS "trackman_catcher" (
  "PitchUID" uuid UNIQUE PRIMARY KEY DEFAULT (uuid_generate_v4()),
  "Catcher" varchar,
  "CatcherID" int,
  "CatcherThrows" varchar,
  "CatcherTeam" varchar,
  "ThrowSpeed" decimal,
  "PopTime" decimal,
  "ExchangeTime" decimal,
  "TimeToBase" decimal,
  "CatchPositionX" decimal,
  "CatchPositionY" decimal,
  "CatchPositionZ" decimal,
  "ThrowPositionX" decimal,
  "ThrowPositionY" decimal,
  "ThrowPositionZ" decimal,
  "BasePositionX" decimal,
  "BasePositionY" decimal,
  "BasePositionZ" decimal,
  "ThrowTrajectoryXc0" decimal,
  "ThrowTrajectoryXc1" decimal,
  "ThrowTrajectoryXc2" decimal,
  "ThrowTrajectoryYc0" decimal,
  "ThrowTrajectoryYc1" decimal,
  "ThrowTrajectoryYc2" decimal,
  "ThrowTrajectoryZc0" decimal,
  "ThrowTrajectoryZc1" decimal,
  "ThrowTrajectoryZc2" decimal,
  "CatcherThrowCatchConfidence" varchar,
  "CatcherThrowReleaseConfidence" varchar,
  "CatcherThrowLocationConfidence" varchar
);

CREATE TABLE IF NOT EXISTS "trackman_batter" (
  "PitchUID" uuid UNIQUE PRIMARY KEY DEFAULT (uuid_generate_v4()),
  "Batter" varchar,
  "BatterID" int,
  "BatterSide" varchar,
  "BatterTeam" varchar,
  "ExitSpeed" decimal,
  "Angle" decimal,
  "Direction" decimal,
  "HitSpinRate" decimal,
  "PositionAt110X" decimal,
  "PositionAt110Y" decimal,
  "PositionAt110Z" decimal,
  "Distance" decimal,
  "LastTracked" decimal,
  "Bearing" decimal,
  "HangTime" decimal,
  "EffectiveVelo" decimal,
  "MaxHeight" decimal,
  "MeasuredDuration" decimal,
  "ContactPositionX" decimal,
  "ContactPositionY" decimal,
  "ContactPositionZ" decimal,
  "HitSpinAxis" decimal,
  "HitTrajectoryXc0" decimal,
  "HitTrajectoryXc1" decimal,
  "HitTrajectoryXc2" decimal,
  "HitTrajectoryXc3" decimal,
  "HitTrajectoryXc4" decimal,
  "HitTrajectoryXc5" decimal,
  "HitTrajectoryXc6" decimal,
  "HitTrajectoryXc7" decimal,
  "HitTrajectoryXc8" decimal,
  "HitTrajectoryYc0" decimal,
  "HitTrajectoryYc1" decimal,
  "HitTrajectoryYc2" decimal,
  "HitTrajectoryYc3" decimal,
  "HitTrajectoryYc4" decimal,
  "HitTrajectoryYc5" decimal,
  "HitTrajectoryYc6" decimal,
  "HitTrajectoryYc7" decimal,
  "HitTrajectoryYc8" decimal,
  "HitTrajectoryZc0" decimal,
  "HitTrajectoryZc1" decimal,
  "HitTrajectoryZc2" decimal,
  "HitTrajectoryZc3" decimal,
  "HitTrajectoryZc4" decimal,
  "HitTrajectoryZc5" decimal,
  "HitTrajectoryZc6" decimal,
  "HitTrajectoryZc7" decimal,
  "HitTrajectoryZc8" decimal,
  "HitLaunchConfidence" varchar,
  "HitLandingConfidence" varchar
);

-- practice_trackman_metadata table
CREATE TABLE IF NOT EXISTS "practice_trackman_metadata" (
  "PitchUID" uuid UNIQUE PRIMARY KEY DEFAULT (uuid_generate_v4()),
  "GameDate" date,
  "PitchTime" time,
  "Inning" int,
  "TopBottom" varchar,
  "Outs" int,
  "Balls" int,
  "Strikes" int,
  "PitchCall" varchar,
  "KorBB" varchar,
  "TaggedHitType" varchar,
  "PlayResult" varchar,
  "OutsOnPlay" varchar,
  "RunsScored" varchar,
  "Notes" varchar,
  "HomeTeam" varchar,
  "AwayTeam" varchar,
  "Stadium" varchar,
  "Level" varchar,
  "League" varchar,
  "GameID" varchar,
  "GameUID" uuid,
  "UTCDate" date,
  "UTCTime" time,
  "LocalDateTime" timestamp,
  "UTCDateTime" timestamp,
  "AutoHitType" varchar,
  "System" varchar,
  "HomeTeamForeignID" varchar,
  "AwayTeamForeignID" varchar,
  "GameForeignID" varchar,
  "PlayID" varchar
);

-- practice_trackman_pitcher table
CREATE TABLE IF NOT EXISTS "practice_trackman_pitcher" (
  "PitchUID" uuid UNIQUE PRIMARY KEY DEFAULT (uuid_generate_v4()),
  "PitchNo" int,
  "PAofInning" int,
  "PitchofPA" int,
  "Pitcher" varchar,
  "PitcherID" int,
  "PitcherThrows" varchar,
  "PitcherTeam" varchar,
  "PitcherSet" varchar,
  "TaggedPitchType" varchar,
  "AutoPitchType" varchar,
  "RelSpeed" decimal,
  "VertRelAngle" decimal,
  "HorzRelAngle" decimal,
  "SpinRate" decimal,
  "SpinAxis" decimal,
  "Tilt" varchar,
  "RelHeight" decimal,
  "RelSide" decimal,
  "Extension" decimal,
  "VertBreak" decimal,
  "InducedVertBreak" decimal,
  "HorzBreak" decimal,
  "PlateLocHeight" decimal,
  "PlateLocSide" decimal,
  "ZoneSpeed" decimal,
  "VertApprAngle" decimal,
  "HorzApprAngle" decimal,
  "ZoneTime" decimal,
  "pfxx" decimal,
  "pfxz" decimal,
  "x0" decimal,
  "y0" decimal,
  "z0" decimal,
  "vx0" decimal,
  "vy0" decimal,
  "vz0" decimal,
  "ax0" decimal,
  "ay0" decimal,
  "az0" decimal,
  "SpeedDrop" decimal,
  "PitchLastMeasuredX" decimal,
  "PitchLastMeasuredY" decimal,
  "PitchLastMeasuredZ" decimal,
  "PitchTrajectoryXc0" decimal,
  "PitchTrajectoryXc1" decimal,
  "PitchTrajectoryXc2" decimal,
  "PitchTrajectoryYc0" decimal,
  "PitchTrajectoryYc1" decimal,
  "PitchTrajectoryYc2" decimal,
  "PitchTrajectoryZc0" decimal,
  "PitchTrajectoryZc1" decimal,
  "PitchTrajectoryZc2" decimal,
  "PitchReleaseConfidence" varchar,
  "PitchLocationConfidence" varchar,
  "PitchMovementConfidence" varchar
);

-- practice_trackman_batter table
CREATE TABLE IF NOT EXISTS "practice_trackman_batter" (
  "PitchUID" uuid UNIQUE PRIMARY KEY DEFAULT (uuid_generate_v4()),
  "Batter" varchar,
  "BatterID" int,
  "BatterSide" varchar,
  "BatterTeam" varchar,
  "ExitSpeed" decimal,
  "Angle" decimal,
  "Direction" decimal,
  "HitSpinRate" decimal,
  "PositionAt110X" decimal,
  "PositionAt110Y" decimal,
  "PositionAt110Z" decimal,
  "Distance" decimal,
  "LastTracked" decimal,
  "Bearing" decimal,
  "HangTime" decimal,
  "EffectiveVelo" decimal,
  "MaxHeight" decimal,
  "MeasuredDuration" decimal,
  "ContactPositionX" decimal,
  "ContactPositionY" decimal,
  "ContactPositionZ" decimal,
  "HitSpinAxis" decimal,
  "HitTrajectoryXc0" decimal,
  "HitTrajectoryXc1" decimal,
  "HitTrajectoryXc2" decimal,
  "HitTrajectoryXc3" decimal,
  "HitTrajectoryXc4" decimal,
  "HitTrajectoryXc5" decimal,
  "HitTrajectoryXc6" decimal,
  "HitTrajectoryXc7" decimal,
  "HitTrajectoryXc8" decimal,
  "HitTrajectoryYc0" decimal,
  "HitTrajectoryYc1" decimal,
  "HitTrajectoryYc2" decimal,
  "HitTrajectoryYc3" decimal,
  "HitTrajectoryYc4" decimal,
  "HitTrajectoryYc5" decimal,
  "HitTrajectoryYc6" decimal,
  "HitTrajectoryYc7" decimal,
  "HitTrajectoryYc8" decimal,
  "HitTrajectoryZc0" decimal,
  "HitTrajectoryZc1" decimal,
  "HitTrajectoryZc2" decimal,
  "HitTrajectoryZc3" decimal,
  "HitTrajectoryZc4" decimal,
  "HitTrajectoryZc5" decimal,
  "HitTrajectoryZc6" decimal,
  "HitTrajectoryZc7" decimal,
  "HitTrajectoryZc8" decimal,
  "HitLaunchConfidence" varchar,
  "HitLandingConfidence" varchar
);

-- practice_trackman_catcher table
CREATE TABLE IF NOT EXISTS "practice_trackman_catcher" (
  "PitchUID" uuid UNIQUE PRIMARY KEY DEFAULT (uuid_generate_v4()),
  "Catcher" varchar,
  "CatcherID" int,
  "CatcherThrows" varchar,
  "CatcherTeam" varchar,
  "ThrowSpeed" decimal,
  "PopTime" decimal,
  "ExchangeTime" decimal,
  "TimeToBase" decimal,
  "CatchPositionX" decimal,
  "CatchPositionY" decimal,
  "CatchPositionZ" decimal,
  "ThrowPositionX" decimal,
  "ThrowPositionY" decimal,
  "ThrowPositionZ" decimal,
  "BasePositionX" decimal,
  "BasePositionY" decimal,
  "BasePositionZ" decimal,
  "ThrowTrajectoryXc0" decimal,
  "ThrowTrajectoryXc1" decimal,
  "ThrowTrajectoryXc2" decimal,
  "ThrowTrajectoryYc0" decimal,
  "ThrowTrajectoryYc1" decimal,
  "ThrowTrajectoryYc2" decimal,
  "ThrowTrajectoryZc0" decimal,
  "ThrowTrajectoryZc1" decimal,
  "ThrowTrajectoryZc2" decimal,
  "CatcherThrowCatchConfidence" varchar,
  "CatcherThrowReleaseConfidence" varchar,
  "CatcherThrowLocationConfidence" varchar
);



CREATE TABLE IF NOT EXISTS "seasons" (
  "SeasonTitle" varchar UNIQUE PRIMARY KEY,
  "StartDate" date,
  "EndDate" date
);

-- Table for defensive shifting modeling team data
CREATE TABLE IF NOT EXISTS "defensive_shift_model_values" (
  "Pitcher" varchar,
  "PitcherTeam" varchar,
  "PitchType" varchar,
  "BatterSide" varchar,
  "ModelValues" decimal [],
  PRIMARY KEY ("Pitcher", "PitcherTeam", "PitchType", "BatterSide")
);

-- Table for heatmap modeling team data
CREATE TABLE IF NOT EXISTS "heatmap_model_values" (
  "Pitcher" varchar,
  "PitcherTeam" varchar,
  "PitchType" varchar,
  "AllPitches" decimal [],
  "SuccessfulPitches" decimal [],
  "PitchRatio" decimal [],
  PRIMARY KEY ("Pitcher", "PitcherTeam", "PitchType")
);

-- Table for batter scores
CREATE TABLE IF NOT EXISTS "batter_run_values" (
  "Batter" varchar,
  "BatterTeam" varchar,
  "BatterSide" varchar,
  "PitchType" varchar,
  "PitcherThrows" varchar,
  "NumPitches" int,
  "Score" decimal,
  PRIMARY KEY ("Batter", "BatterTeam", "PitchType", "PitcherThrows", "BatterSide")
);

-- Table for pitcher scores
CREATE TABLE IF NOT EXISTS "pitcher_run_values" (
  "Pitcher" varchar,
  "PitcherTeam" varchar,
  "PitchType" varchar,
  "Handedness" varchar,
  "Score" decimal,
  PRIMARY KEY ("Pitcher", "PitcherTeam", "PitchType")
);


-- Foreign key constraints for "practice_trackman_metadata"
ALTER TABLE "practice_trackman_metadata"
    ADD FOREIGN KEY ("HomeTeam") REFERENCES "teams" ("TeamName") ON DELETE CASCADE,
    ADD FOREIGN KEY ("AwayTeam") REFERENCES "teams" ("TeamName") ON DELETE CASCADE;

-- Foreign key constraints for "practice_trackman_pitcher"
ALTER TABLE "practice_trackman_pitcher"
    ADD FOREIGN KEY ("Pitcher", "PitcherTeam") REFERENCES "players" ("PlayerName", "TeamName") ON DELETE CASCADE,
    ADD FOREIGN KEY ("PitchUID") REFERENCES "practice_trackman_metadata" ("PitchUID") ON DELETE CASCADE;

-- Foreign key constraints for "practice_trackman_batter"
ALTER TABLE "practice_trackman_batter"
    ADD FOREIGN KEY ("Batter", "BatterTeam") REFERENCES "players" ("PlayerName", "TeamName") ON DELETE CASCADE,
    ADD FOREIGN KEY ("PitchUID") REFERENCES "practice_trackman_metadata" ("PitchUID") ON DELETE CASCADE;

-- Foreign key constraints for "practice_trackman_catcher"
ALTER TABLE "practice_trackman_catcher"
    ADD FOREIGN KEY ("Catcher", "CatcherTeam") REFERENCES "players" ("PlayerName", "TeamName") ON DELETE CASCADE,
    ADD FOREIGN KEY ("PitchUID") REFERENCES "practice_trackman_metadata" ("PitchUID") ON DELETE CASCADE;

-- Foreign key constraints for "trackman_pitcher"
ALTER TABLE "trackman_pitcher"
    ADD FOREIGN KEY ("PitchUID") REFERENCES "trackman_metadata" ("PitchUID") ON DELETE CASCADE,
    ADD FOREIGN KEY ("Pitcher", "PitcherTeam") REFERENCES "players" ("PlayerName", "TeamName") ON DELETE CASCADE;

-- Foreign key constraints for "trackman_batter"
ALTER TABLE "trackman_batter"
    ADD FOREIGN KEY ("PitchUID") REFERENCES "trackman_metadata" ("PitchUID") ON DELETE CASCADE,
    ADD FOREIGN KEY ("Batter", "BatterTeam") REFERENCES "players" ("PlayerName", "TeamName") ON DELETE CASCADE;

-- Foreign key constraints for "trackman_catcher"
ALTER TABLE "trackman_catcher"
    ADD FOREIGN KEY ("PitchUID") REFERENCES "trackman_metadata" ("PitchUID") ON DELETE CASCADE,
    ADD FOREIGN KEY ("Catcher", "CatcherTeam") REFERENCES "players" ("PlayerName", "TeamName") ON DELETE CASCADE;

-- Foreign key constraints for "trackman_metadata"
ALTER TABLE "trackman_metadata"
    ADD FOREIGN KEY ("HomeTeam") REFERENCES "teams" ("TeamName") ON DELETE CASCADE,
    ADD FOREIGN KEY ("AwayTeam") REFERENCES "teams" ("TeamName") ON DELETE CASCADE;

-- Foreign key constraints for "players"
ALTER TABLE "players"
    ADD FOREIGN KEY ("TeamName") REFERENCES "teams" ("TeamName") ON DELETE CASCADE;

-- Foreign key constraints for "teams"
ALTER TABLE "teams"
    ADD FOREIGN KEY ("Conference") REFERENCES "conferences" ("ConferenceName") ON DELETE CASCADE;

-- Foreign key constraints for "defensive_shift_model_values"
ALTER TABLE "defensive_shift_model_values"
    ADD FOREIGN KEY ("Pitcher", "PitcherTeam") REFERENCES "players" ("PlayerName", "TeamName") ON DELETE CASCADE;

-- Foreign key constraints for "heatmap_model_values"
ALTER TABLE "heatmap_model_values"
    ADD FOREIGN KEY ("Pitcher", "PitcherTeam") REFERENCES "players" ("PlayerName", "TeamName") ON DELETE CASCADE;

-- Foreign key constraints for "batter_run_values"
ALTER TABLE "batter_run_values"
    ADD FOREIGN KEY ("Batter", "BatterTeam") REFERENCES "players" ("PlayerName", "TeamName") ON DELETE CASCADE;

-- Foreign key constraints for "pitcher_run_values"
ALTER TABLE "pitcher_run_values"
    ADD FOREIGN KEY ("Pitcher", "PitcherTeam") REFERENCES "players" ("PlayerName", "TeamName") ON DELETE CASCADE;

-- Custom constraint for "players"
ALTER TABLE "players"
    ADD CONSTRAINT "no_commas_allowed" CHECK ("PlayerName" NOT LIKE '%,%');
