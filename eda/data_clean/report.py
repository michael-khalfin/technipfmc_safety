import pandas as pd 
import math 
from utils import find_common_cols

action = pd.read_csv('data/DIM_CONSOLIDATED_ACTION_PLAN.csv', low_memory=False)




def check_files_with_loss_potential_record():
    # Main Loss File 
    loss = pd.read_csv('data/DIM_CONSOLIDATED_LOSS_POTENTIAL.csv', low_memory=False)

    # Load files
    acc = pd.read_csv('data/DIM_CONSOLIDATED_ACCIDENTS.csv', low_memory=False)
    haz = pd.read_csv('data/DIM_CONSOLIDATED_HAZARD_OBSERVATIONS.csv', low_memory=False)
    nm = pd.read_csv('data/DIM_CONSOLIDATED_NEAR_MISSES.csv', low_memory=False)
    print(f"\nFILE SIZES")
    print(f"\tLOSS_POTENTIAL: {len(loss):,} rows, {len(loss.columns)} columns")
    print(f"\tACCIDENTS: {len(acc):,} rows, {len(acc.columns)} columns")
    print(f"\tHAZARD_OBSERVATIONS: {len(haz):,} rows, {len(haz.columns)} columns")
    print(f"\tNEAR_MISSES: {len(nm):,} rows, {len(nm.columns)} columns")
    print(f"\tACTION_PLAN: {len(action):,} rows, {len(action.columns)} columns")


    # Count how many LOSS_POTENTIAL records link to each file
    loss_ids = set(loss['RECORD_NO_LOSS_POTENTIAL'].dropna().astype(str))
    acc_ids = set(acc['RECORD_NO_LOSS_POTENTIAL'].dropna().astype(str))
    haz_ids = set(haz['RECORD_NO_LOSS_POTENTIAL'].dropna().astype(str))
    nm_ids = set(nm['RECORD_NO_LOSS_POTENTIAL'].dropna().astype(str))


    print(f"\nUnique RECORD_NO_LOSS_POTENTIAL values:")
    print(f"\tLOSS_POTENTIAL: {len(loss_ids):,}")
    print(f"\tACCIDENTS: {len(acc_ids):,}")
    print(f"\tHAZARD_OBSERVATIONS: {len(haz_ids):,}")
    print(f"\tNEAR_MISSES: {len(nm_ids):,}")

    # Check overlaps with Main Loss File 
    acc_in_loss = len(acc_ids & loss_ids)
    haz_in_loss = len(haz_ids & loss_ids)
    nm_in_loss = len(nm_ids & loss_ids)


    print(f"\nLOSS POTENTIAL LINKAGE")
    acc_loss_percentage = acc_in_loss/len(acc_ids)*100
    haz_loss_percentage = haz_in_loss/len(haz_ids)*100
    nm_loss_percentage = nm_in_loss/len(nm_ids)*100
    acc_haz_overlap = len(acc_ids & haz_ids)
    acc_nm_overlap = len(acc_ids & nm_ids)
    haz_nm_overlap = len(haz_ids & nm_ids)
    print(f"   ACCIDENTS -> LOSS_POTENTIAL: {acc_in_loss:,} / {len(acc_ids):,} ({acc_loss_percentage:.1f}%)")
    print(f"   HAZARD -> LOSS_POTENTIAL: {haz_in_loss:,} / {len(haz_ids):,} ({haz_loss_percentage:.1f}%)")
    print(f"   NEAR_MISSES -> LOSS_POTENTIAL: {nm_in_loss:,} / {len(nm_ids):,} ({nm_loss_percentage:.1f}%)")
    print(f"   ACCIDENTS overlap with HAZARD: {acc_haz_overlap:,}")
    print(f"   ACCIDENTS overlap with NEAR_MISSES: {acc_nm_overlap:,}")
    print(f"   HAZARD overlap with NEAR_MISSES: {haz_nm_overlap:,}")


    print(f"\nCOMMON COLUMNS FOR ACCIDENTS & HAZARDS")
    find_common_cols(acc, haz)
    print(f"\nCOMMON COLUMNS FOR ACCIDENTS & NEAR_MISSES")
    find_common_cols(acc, nm)
    print(f"\nCOMMON COLUMNS FOR HAZARDS & NEAR_MISSES")
    find_common_cols(haz, nm)


    if (
    all(math.ceil(p) == 100 for p in {acc_loss_percentage, haz_loss_percentage, nm_loss_percentage})
    and all(int(o) == 0 for o in {acc_haz_overlap, acc_nm_overlap, haz_nm_overlap})
    ):
        print("\n[REPORT ARTIFACT] ACCIDENTS, HAZARDS, and NEAR MISSES ARE MUTUALLY EXCLUSIVE AND ALL EXIST WITHIN LOSS POTENTIAL")


    return


check_files_with_loss_potential_record()








