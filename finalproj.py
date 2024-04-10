#%%
import pandas as pd
import numpy as np
import time
from tqdm import tqdm

#### ACLED FUNCTION ####

selected_dates = set()
# **ACLED** EVENT DIFFERENCE
def eventdiff(acled_df, target, country, region, event_date, days):
    event_date = pd.to_datetime(event_date)  # Ensure event_date is a pandas Timestamp
    start_date = event_date - pd.Timedelta(days=days)
    end_date = event_date + pd.Timedelta(days=days)

    # FILTER / COUNT EVENTS BEFORE STRIKE EX. EVENT DATE
    events_before1 = len(acled_df[(acled_df['actor1'] == target) &
                                 (acled_df['country'] == country) &
                                 (acled_df['event_date'] > start_date) &
                                 (acled_df['admin1'] == region) &
                                 (acled_df['event_date'] < event_date)])

    events_before2 = len(acled_df[(acled_df['actor2'] == target) &
                                 (acled_df['country'] == country) &
                                 (acled_df['event_date'] > start_date) &
                                 (acled_df['admin1'] == region) &
                                 (acled_df['event_date'] < event_date)])

    events_before3 = len(acled_df[(acled_df['assoc_actor_1'] == target) &
                                 (acled_df['country'] == country) &
                                 (acled_df['event_date'] > start_date) &
                                 (acled_df['admin1'] == region) &
                                 (acled_df['event_date'] < event_date)])

    events_before4 = len(acled_df[(acled_df['assoc_actor_2'] == target) &
                                 (acled_df['country'] == country) &
                                 (acled_df['event_date'] > start_date) &
                                 (acled_df['admin1'] == region) &
                                 (acled_df['event_date'] < event_date)])

    # FILTER / COUNT EVENTS AFTER STRIKE EX. EVENT DATE
    events_after1 = len(acled_df[(acled_df['actor1'] == target) &
                                 (acled_df['country'] == country) &
                                 (acled_df['event_date'] < end_date) &
                                 (acled_df['admin1'] == region) &
                                 (acled_df['event_date'] > event_date)])

    events_after2 = len(acled_df[(acled_df['actor2'] == target) &
                                 (acled_df['country'] == country) &
                                 (acled_df['event_date'] < end_date) &
                                 (acled_df['admin1'] == region) &
                                 (acled_df['event_date'] > event_date)])

    events_after3 = len(acled_df[(acled_df['assoc_actor_1'] == target) &
                                 (acled_df['country'] == country) &
                                 (acled_df['event_date'] < end_date) &
                                 (acled_df['admin1'] == region) &
                                 (acled_df['event_date'] > event_date)])

    events_after4 = len(acled_df[(acled_df['assoc_actor_2'] == target) &
                                 (acled_df['country'] == country) &
                                 (acled_df['event_date'] < end_date) &
                                 (acled_df['admin1'] == region) &
                                 (acled_df['event_date'] > event_date)])

    return (events_after1 + events_after2 + events_after3 + events_after4) - (events_before1 + events_before2 + events_before3 + events_before4)

print("ACLED FUNCTION LOADED.")

# **ACLED** RANDOM DATE FUNCTION (NO STRIKE 1MO | FULL DATASET | REGION TEMPORAL)
def acled_randdate(drone_strike_df, acled_df, treatment_date, target_region, target_group):
    # DATE TO DATETIME
    drone_strike_df['Date'] = pd.to_datetime(drone_strike_df['Date'], errors='coerce')
    if 'event_date' in acled_df.columns:
        acled_df.loc[:, 'event_date'] = pd.to_datetime(acled_df['event_date'], errors='coerce')

    # 3YR WINDOW
    three_years_before = pd.to_datetime(treatment_date) - pd.DateOffset(years=2)
    three_years_after = pd.to_datetime(treatment_date) + pd.DateOffset(years=2)

    # FILTER STRIKES BY OTHER STRIKES IN REGION
    region_strikes = drone_strike_df[(drone_strike_df['Region'] == target_region) &
                                     (drone_strike_df['Date'] >= three_years_before) &
                                     (drone_strike_df['Date'] <= three_years_after)]
    strike_dates = region_strikes['Date'].dropna().unique()

    # EXCLUDE DATES W/ STRIKE W/IN 1 MO BUFFER
    excluded_dates = []
    for strike_date in strike_dates:
        excluded_dates.extend(pd.date_range(start=strike_date - pd.DateOffset(days=30),
                                            end=strike_date + pd.DateOffset(days=30)))

    # GENERATE POTENTIAL DATES W/O EXCLUSIONS
    potential_dates = pd.date_range(start=three_years_before, end=three_years_after)
    potential_dates = [date for date in potential_dates if date not in excluded_dates]

    # FILTER OUT EVENTS WHERE NO ACLED EVENTS W/IN 1MO
    valid_dates = []
    for date in potential_dates:
        one_month_before = date - pd.DateOffset(days=30)
        one_month_after = date + pd.DateOffset(days=30)

        events_in_range = acled_df[
            (acled_df['event_date'] >= one_month_before) &
            (acled_df['event_date'] <= one_month_after) &
            (acled_df['admin1'] == target_region) &
            ((acled_df['actor1'] == target_group) |
             (acled_df['actor2'] == target_group) |
             (acled_df['assoc_actor_1'].str.contains(target_group, na=False)) |
             (acled_df['assoc_actor_2'].str.contains(target_group, na=False)))
            ]

        if not events_in_range.empty:
            valid_dates.append(date)

    # Randomly select a valid date, checking for previously returned dates
    valid_date_found = False
    while valid_dates and not valid_date_found:
        random_date = np.random.choice(valid_dates)
        # Ensure the random date has not been previously selected
        if random_date not in selected_dates:
            valid_date_found = True
            # Add the newly selected date to the set of selected dates
            selected_dates.add(random_date)
        else:
            # Remove the date from potential dates and try again
            valid_dates.remove(random_date)
    if not valid_date_found:
        return None

    return random_date

# ACLED PREFILTERING FUNCTION
def filter_acled_by_targets(acled_df, target_groups):

    # Convert target_groups to a set for efficient lookups
    target_groups_set = set(target_groups)

    # Define the columns to check for target groups
    columns_to_check = ['actor1', 'actor2', 'assoc_actor_1', 'assoc_actor_2']

    # Function to check if any target group is present in any of the specified columns for a row
    def contains_target_group(row):
        return any(row[col] in target_groups_set for col in columns_to_check)

    # Filter the ACLED DataFrame based on the presence of target groups
    filtered_acled_df = acled_df[acled_df.apply(contains_target_group, axis=1)]

    return filtered_acled_df

# **ACLED** LOAD IN DATASETS
print("Loading ACLED...")
unf_acled_df = pd.read_excel('/Users/noahs/Desktop/DroneProj/ACLED2.xlsx')  # CHANGE TO ACLED INPUT
print("Loading ACLED Input...")
drone_df = pd.read_excel('/Users/noahs/Desktop/DroneProj/ACLEDinput.xlsx')  # CHANGE TO DRONE STRIKE DATASET INPUT
print("ACLED Loaded.")
print("Filtering ACLED Dataset.")

# List of target groups from the drone strike dataset
target_groups = ["Al-Qaeda", "ISIS", "Taliban", "Al Shabaab"]

# Filter the ACLED dataset
acled_df = filter_acled_by_targets(unf_acled_df, target_groups)

print("ACLED Pre-Filtering Completed Successfully.")

# **ACLED** TOTAL NUMBER OF STRIKES
total_strikes = len(drone_df)

print("Starting processing of drone strikes for ACLED countries.")

## **ACLED** ITERATIVE LOOP -> DRONE STRIKE PROCESSING FUNCTION
results = []

# Initialize the progress bar with the total number of iterations
pbar = tqdm(total=total_strikes, unit='strikes', desc="Processing ACLED Data", leave=True)
start_time = time.time()

for index, row in drone_df.iterrows():
    target = row['Target']
    country = row['ISO-3']
    region = row['Region']
    strike_date = pd.to_datetime(row['Date'])

    # CONTROL VARIABLES INITIALIZE AS NP.NAN
    c3d_diff = c1w_diff = c2w_diff = c1m_diff = np.nan

   # CONTROL & TREATMENT
    random_event_date = acled_randdate(drone_df, acled_df, strike_date, region, target)
    if random_event_date is not None:
        c3d_diff = eventdiff(acled_df, target, country, region, random_event_date, 3)
        c1w_diff = eventdiff(acled_df, target, country, region, random_event_date, 7)
        c2w_diff = eventdiff(acled_df, target, country, region, random_event_date, 14)
        c1m_diff = eventdiff(acled_df, target, country, region, random_event_date, 30)
        # TREATMENT
        t3d_diff = eventdiff(acled_df, target, country, region, strike_date, 3)
        t1w_diff = eventdiff(acled_df, target, country, region, strike_date, 7)
        t2w_diff = eventdiff(acled_df, target, country, region, strike_date, 14)
        t1m_diff = eventdiff(acled_df, target, country, region, strike_date, 30)
        if t3d_diff != 0 or t1w_diff != 0 or t2w_diff != 0 or t1m_diff != 0:
            # CIV / DECAP TREATMENT GROUPS
            civilian_casualties = row['CivKill']
            leader_casualties = row['LeadKill']
            strike_id = row['StrikeID']
            # APPEND
            results.append([strike_id, country, region, target, civilian_casualties, leader_casualties, random_event_date,
                 t3d_diff, t1w_diff, t2w_diff, t1m_diff, c3d_diff, c1w_diff, c2w_diff, c1m_diff])

    # Progress Bar
    pbar.update(1)

    # Optionally, you can calculate and display estimated time remaining
    elapsed_time = time.time() - start_time  # Time elapsed since start
    strikes_processed = index + 1  # Number of strikes processed so far
    if strikes_processed < total_strikes:
        estimated_total_time = (elapsed_time / strikes_processed) * total_strikes
        estimated_time_remaining = estimated_total_time - elapsed_time
        hours, remainder = divmod(estimated_time_remaining, 3600)
        minutes, _ = divmod(remainder, 60)
        pbar.set_postfix_str(f"Remaining: {int(hours)}h {int(minutes)}m")
    else:
        pbar.set_postfix_str("Processing complete.")

# **ACLED** RESULT NOTIFICATION
pbar.close()
print("ACLED PROCESSING COMPLETE.")
print("APPRENDING ACLED RESULTS.")

# **ACLED** STORE RESULTS
results_df = pd.DataFrame(results, columns=['StrikeID', 'Country', 'Region', 'Target', 'CivKill', 'LeadKill', 'RandDate',
                                            'T_3D_Diff','T_1W_Diff','T_2W_Diff','T_1M_Diff',
                                            'C_3D_Diff','C_1W_Diff','C_2W_Diff','C_1M_Diff'])

# **ACLED** WRITE TO EXCEL
results_df.to_excel('/Users/noahs/Desktop/DroneProj/ACLEDoutput.xlsx', index=False) # CHANGE TO PATH TO WRITE FILE

print("ACLED RESULTS WRITTEN TO EXCEL.")