# General
import numpy as np
import pandas as pd
import json
import pickle
import time
import os

from physrisk.kernel import  Asset
from physrisk.utils.lazy import lazy_import
from physrisk.utils.helpers import get_iterable # From impact.py
from physrisk.kernel.assets import Asset
from physrisk.kernel.hazard_model import HazardDataRequest

# For hazards
from physrisk.kernel.hazards import (
    RiverineInundation,
    CoastalInundation,
    Wind,
    Fire,
    WaterStress,
    Landslide,
    Subsidence
)

from physrisk.data.pregenerated_hazard_model import ZarrHazardModel
from physrisk.hazard_models.core_hazards import get_default_source_paths
from physrisk.data.inventory import EmbeddedInventory

pd = lazy_import("pandas")


def read_assets():

    # Read assets from database
    # Add path to the file "global_power_plant_database.csv"
    # In this case, it is in the same directory as this tutorial so we will just load it

    # db_file_name = "src\global_power_plant_database.csv"
    db_file_name = "global_power_plant_database.csv"
    exp_df = pd.read_csv(db_file_name, low_memory=False)

    # Clean data to take only Spanish ones and remove outliers
    exp_df = exp_df[exp_df.country == "ESP"]
    exp_df = exp_df[exp_df.latitude > 34]
    asset_names = exp_df.name.tolist()
    asset_subtype = exp_df.primary_fuel.tolist()
    ids_ = exp_df.gppd_idnr.tolist()

    # Extract data from the dataframes to build the Asset objects
    longitudes = np.array(exp_df["longitude"])
    latitudes = np.array(exp_df["latitude"])


    ##############################################################################
    # CUSTOM LOCATION: COMMENT
    ##############################################################################

    # latitudes[0] = 36
    # longitudes[0] = -0.1

    ##############################################################################
    # CUSTOM LOCATION: COMMENT
    ##############################################################################


    # primary_fuel = np.array(exp_df["primary_fuel"])
    # generation = np.array(exp_df["estimated_generation_gwh_2017"])

    # Generate list of assets of type PowerGeneratingAsset
    # The only mandatory arguments are latitude and longitude. You can add the ones you want.
    # In this case, we have added the generation and the primary_fuel but you can easily pass the ones you want like:
    # PowerGeneratingAsset(lat, lon, food = "pasta", team = "Barsa", goals = 33)

    assets = [Asset(lat, lon, **{'id':id_, 'name':name, 'subtype':subtype}) for lon, lat, id_, name, subtype in zip(longitudes, latitudes, ids_, asset_names, asset_subtype)]

    return assets


def find_s3_zarr_source_paths():
    # Get paths to S3 bucket

    source_paths = get_default_source_paths(EmbeddedInventory())
    zarr_source_paths = {
        RiverineInundation: source_paths[RiverineInundation],
        CoastalInundation: source_paths[CoastalInundation],
        Wind: source_paths[Wind],
        Fire: source_paths[Fire],
        WaterStress: source_paths[WaterStress],
        Landslide: source_paths[Landslide],
        Subsidence: source_paths[Subsidence],
    }

    return zarr_source_paths


def create_hazard_model():

    zarr_source_paths = find_s3_zarr_source_paths()
    hazard_model = ZarrHazardModel(source_paths=zarr_source_paths)

    return hazard_model


def read_inventory():
    
    f = open(os.path.join('physrisk/data/static/hazard', 'inventory.json'), 'r')
    inventory = json.load(f)['resources'] 
    f.close()

    return inventory


def define_hazards_to_download():

    hazards_to_download ={
        'RiverineInundation': RiverineInundation,
        'CoastalInundation': CoastalInundation,
        'Wind': Wind,
        'Fire': Fire,
        'WaterStress': WaterStress,
        'Landslide': Landslide,
        'Subsidence': Subsidence,
    }

    return hazards_to_download


def get_inventory_metadata(inventory, i, full=False):

    metadata = inventory[i]
    hazard_type = metadata['hazard_type']
    group_id = metadata['group_id']
    indicator_id = metadata['indicator_id']
    model_id = metadata['indicator_model_gcm']
    scenarios = metadata['scenarios']
    description = metadata['description']
    display_name = metadata['display_name']

    if not full:
        return hazard_type, group_id, indicator_id, model_id, scenarios
    else:
        return hazard_type, group_id, indicator_id, model_id, scenarios, description, display_name

def create_dict_key(asset_hazard_data, hazard_type, list_=False):

    if hazard_type not in asset_hazard_data.keys():
        if not list_:
            asset_hazard_data[hazard_type] = dict()
        else:
            asset_hazard_data[hazard_type] = []


def extract_scenarios(scenarios, j):

    scenario = scenarios[j]['id']
    years = scenarios[j]['years']

    return scenario, years


def create_flattened_requests(inventory, assets, hazards_to_download):


    def create_HazardDataRequest(ht, asset, scenario, year, indicator_id):

        histo = HazardDataRequest(
            ht,
            asset.longitude,
            asset.latitude,
            scenario=scenario,
            year=year,
            indicator_id=indicator_id,
        )

        return histo

    def create_response_key(asset_hazard_data_scenario_asset, asset_name):

        if 'response' not in asset_hazard_data_scenario_asset[asset_name].keys():
            asset_hazard_data_scenario_asset[asset_name]['request'] = []
            asset_hazard_data_scenario_asset[asset_name]['response'] = []


    counter = 0
    flattened_requests = []
    asset_hazard_data = dict()

    for i in range(len(inventory)):
        hazard_type, group_id, indicator_id, model_id, scenarios = get_inventory_metadata(inventory, i)
        if hazard_type not in hazards_to_download: continue
        hazard_type_group_id = hazard_type + '_' + group_id
        create_dict_key(asset_hazard_data, hazard_type_group_id)
        
        for j in range(len(scenarios)):
            scenario, years = extract_scenarios(scenarios, j)

            for year in years:
                scenario_year_key = indicator_id + '_' + scenario + '_' + str(year) + '_' + model_id + '_' + str(i)
                asset_hazard_data_scenario = asset_hazard_data[hazard_type_group_id]
                create_dict_key(asset_hazard_data_scenario, scenario_year_key)

                for asset in assets:
                    histo = create_HazardDataRequest(hazards_to_download[hazard_type], 
                                                     asset, scenario, year, indicator_id)

                    asset_name = asset.name +'_'+ asset.subtype + '_code'+str(counter)
                    asset_hazard_data_scenario_asset = asset_hazard_data[hazard_type_group_id][scenario_year_key]
                    create_dict_key(asset_hazard_data_scenario_asset, asset_name)
                    create_response_key(asset_hazard_data_scenario_asset, asset_name)

                    asset_hazard_data_scenario_asset[asset_name]['request'].append(histo)
                    asset_hazard_data_scenario_asset[asset_name]['response'].append(counter)

                    flattened_requests.append(histo)

                    counter += 1

    return flattened_requests, asset_hazard_data


def add_responses_to_dict(resp_list, inventory, assets, hazards_to_download, pickle_name, save = True):

    counter = 0
    for i in range(len(inventory)):
        hazard_type, group_id, indicator_id, model_id, scenarios = get_inventory_metadata(inventory, i)
        if hazard_type not in hazards_to_download: continue
        hazard_type_group_id = hazard_type + '_' + group_id

        for j in range(len(scenarios)):
            scenario, years = extract_scenarios(scenarios, j)

            for year in years:
                scenario_year_key = indicator_id + '_' + scenario + '_' + str(year) + '_' + model_id + '_' + str(i)

                for asset in assets:
                    asset_name = asset.name +'_'+ asset.subtype + '_code'+str(counter)
                    counter += 1
                    aux = []
                    for count_ in asset_hazard_data[hazard_type_group_id][scenario_year_key][asset_name]['response']:
                        aux.append(resp_list[count_])
                    asset_hazard_data[hazard_type_group_id][scenario_year_key][asset_name]['response'] = aux

    if save:
        f = open(pickle_name, 'wb')
        pickle.dump(asset_hazard_data, f)
        f.close()

    return asset_hazard_data


def create_excel_from_responses(inventory, asset_hazard_data, excel_name):

    def define_excel_columns():
        cols = ['asset_name', 'latitude', 'longitude', 'hazard_type', 'indicator_id', 
                'scenario', 'year', 'model', 'description', 'display_name']
        return cols
    
    def define_is_rp_data():

        is_rp_data = {
            'RiverineInundation_river_tudelft':True,
            'CoastalInundation_coastal_tudelft':True,
            'Wind_ecb':True,
            'Fire_ecb_fwi':False,
            'WaterStress_wri':False,
            'Landslide_jrc':False,
            'Subsidence_jrc':False,
        }
        return is_rp_data

    dfs_dict = dict()
    counter = 0
    is_rp_data = define_is_rp_data()
    for i in range(len(inventory)):
        rows = []
        cols = define_excel_columns()

        hazard_type, group_id, indicator_id, model_id, scenarios, description, display_name = get_inventory_metadata(inventory, i, full=True)
        if hazard_type not in hazards_to_download: continue
        hazard_type_group_id = hazard_type + '_' + group_id

        for j in range(len(scenarios)):
            scenario, years = extract_scenarios(scenarios, j)

            for year in years:
                scenario_year_key = indicator_id + '_' + scenario + '_' + str(year) + '_' + model_id + '_' + str(i)

                for asset in assets:
                    asset_name = asset.name +'_'+ asset.subtype + '_code'+str(counter)
                    request = asset_hazard_data[hazard_type_group_id][scenario_year_key][asset_name]['request'][0]
                    response = asset_hazard_data[hazard_type_group_id][scenario_year_key][asset_name]['response'][0] 

                    counter += 1

                    lat = request.latitude
                    lon = request.longitude

                    intensities = response.intensities.tolist()

                    new_row = [asset_name, lat, lon, hazard_type_group_id, indicator_id, scenario, 
                               year, model_id, description, display_name]
                    new_row.extend(intensities)

                    rows.append(new_row)

        create_dict_key(dfs_dict, hazard_type_group_id, list_=True)

        df = pd.DataFrame(rows)
        if is_rp_data[hazard_type_group_id]:
            intensity_columns = response.return_periods
            intensity_columns = [str(ax) for ax in intensity_columns]
        else:
            intensity_columns = ['parameter']
        cols.extend(intensity_columns)
        df.columns = cols

        dfs_dict[hazard_type_group_id].append(df)


    writer = pd.ExcelWriter(excel_name)
    df_param = pd.DataFrame()
    for hazard_type_group_id, is_rp in is_rp_data.items():
        sheetName = hazard_type_group_id.split('_')[0]
        if is_rp:
            df_rp = pd.DataFrame()
            dfs = dfs_dict[hazard_type_group_id]
            for df in dfs:
                if df_rp.empty:
                    df_rp = df
                else:
                    df_rp = pd.concat([df_rp, df], axis = 0)

            asset_name_split = pd.DataFrame(df_rp.asset_name.apply(lambda x: x.split('_')).tolist())
            df_rp['asset_name'] = asset_name_split.iloc[:,0]
            df_rp['asset_subtype'] = asset_name_split.iloc[:,1]
            df_rp.to_excel(writer, sheet_name=sheetName, index=False)

        else:
            dfs = dfs_dict[hazard_type_group_id]
            for df in dfs:
                if df_param.empty:
                    df_param = df
                else:
                    df_param = pd.concat([df_param, df], axis = 0)

    asset_name_split = pd.DataFrame(df_param.asset_name.apply(lambda x: x.split('_')).tolist())
    df_param['asset_name'] = asset_name_split.iloc[:,0]
    df_param['asset_subtype'] = asset_name_split.iloc[:,1]
    df_param.to_excel(writer, sheet_name='Parameters', index=False)

    writer._save()


def add_risk_metrics(excel_name):


    hazard_type_dict = {
        'RiverineInundation':'RiverineInundation_river_tudelft', 
        'CoastalInundation':'CoastalInundation_coastal_tudelft',
        'Wind':'Wind_ecb',
        'Fire':'Fire_ecb_fwi',
        'WaterStress':'WaterStress_wri',
        'Landslide':'Landslide_jrc',
        'Subsidence':'Subsidence_jrc',
    }

    hazard_type_year = {
        'RiverineInundation':1971, 
        'CoastalInundation':1971,
        'Wind':1980,
        'Fire':1971,
        'Landslide':1980,
        'Subsidence':1980,
    }

    with open('risk_mv.pickle', 'rb') as handle:
        asset_impacts = pickle.load(handle)

    df_river = pd.read_excel(excel_name, sheet_name='RiverineInundation')
    df_coast = pd.read_excel(excel_name, sheet_name='CoastalInundation')
    df_wind = pd.read_excel(excel_name, sheet_name='Wind')
    df_param = pd.read_excel(excel_name, sheet_name='Parameters')

    hazard_type_df = {
        'RiverineInundation': df_river, 
        'CoastalInundation': df_coast,
        'Wind': df_wind,
        'Params': df_param,
    }

    new_cols = ['impact_mean', 'impact_distr_bin_edges', 'impact_distr_p', 'impact_exc_exceed_p', 'impact_exc_values']
    for new_col in new_cols:
        for df_ in hazard_type_df.values():
            df_[new_col] = ""

    for asset, asset_impact in asset_impacts.items():

        lat = asset.latitude
        lon = asset.longitude

        for impact in asset_impact:
            hazard_type = impact.hazard_type
            hazard_type_id = hazard_type_dict[hazard_type]
            scenario = impact.key.scenario_id
            
            year = impact.key.year
            try:
                year = int(year)
            except:
                year = hazard_type_year[hazard_type]
            
            impact_mean = impact.impact_mean
            impact_distr_bin_edges = impact.impact_distribution.bin_edges
            impact_distr_probabilities = impact.impact_distribution.probabilities
            impact_exc_exceed_probabilities = impact.impact_exceedance.exceed_probabilities
            impact_exc_values = impact.impact_exceedance.values


            new_row = [impact_mean,
                       ';'.join(impact_distr_bin_edges.astype(str)),
                       ';'.join(impact_distr_probabilities.astype(str)),
                       ';'.join(impact_exc_exceed_probabilities.astype(str)),
                       ';'.join(impact_exc_values.astype(str))]
            
            try:
                df_ = hazard_type_df[hazard_type]
            except:
                df_ = hazard_type_df['Params']

            df_.loc[(df_.hazard_type == hazard_type_id) & 
                    (df_.scenario == scenario) & 
                    (df_.year == year) &
                    (df_.latitude == lat) &
                    (df_.longitude == lon), new_cols] = new_row
        
    return hazard_type_df
    






if __name__ == '__main__':

    pickle_name = 'data_mv.pkl'
    excel_name = pickle_name.replace('pkl', 'xlsx')

    assets = read_assets()
    hazard_model = create_hazard_model()

    hazards_to_download = define_hazards_to_download()
    inventory = read_inventory()

    if pickle_name in os.listdir(os.getcwd()):
        f = open(pickle_name, 'rb')
        asset_hazard_data = pickle.load(f)
        f.close()

    else:

        t = time.time()
        flattened_requests, asset_hazard_data = create_flattened_requests(inventory, assets, hazards_to_download)

        responses = hazard_model.get_hazard_events(get_iterable(flattened_requests))

        resp_list = []
        for req in flattened_requests: resp_list.append(responses[req])

        elapsed = time.time() - t
        print(elapsed)

        asset_hazard_data = add_responses_to_dict(resp_list, inventory, assets, hazards_to_download, pickle_name)

    create_excel_from_responses(inventory, asset_hazard_data, excel_name)
    hazard_type_df = add_risk_metrics(excel_name)


    writer = pd.ExcelWriter(excel_name)
    for key_, df_ in hazard_type_df.items():
        df_.to_excel(writer, sheet_name=key_, index=False)
    writer._save()

        