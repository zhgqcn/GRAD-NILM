import numpy as np

def getUkdaleAppliancesAttribution():
    '''
    The attributions for each appliance in UKDALE Dataset.
    Ref:
        - https://github.com/sambaiga/UNETNiLM/blob/HEAD/src/data/load_data.py#L24
        - https://github.com/Yueeeeeeee/BERT4NILM/blob/HEAD/utils.py#L61-L67
    '''
    ukdale_appliance_data = {
        "kettle": {
            "mean": 700,
            "std": 1000,
            'cutoff':3100,
            'on_power_threshold': 2000,
            'max_on_power': 3998,
            'min_on': 2,
            'min_off': 0
        },
        "fridge": {
            "mean": 200,
            "std": 400,
            "cutoff":300,
            'on_power_threshold': 50, 
            'max_on_power': 3323,
            'min_on': 10,
            'min_off': 2    
        },
        "dishwasher": {
            "mean": 700,
            "std": 700,
            "cutoff":2500,
            'on_power_threshold': 10,
            'max_on_power': 3964,
            'min_on': 300,
            'min_off': 300
        },
        
        "washingmachine": {
            "mean": 400,
            "std": 700,
            "cutoff":2500,
            'on_power_threshold': 20,
            'max_on_power': 3999,
            'min_on': 300,
            'min_off': 300
        },
        "microwave": {
            "mean": 500,
            "std": 800,
            "cutoff":3000,
            'on_power_threshold': 200,
            'max_on_power': 3969,
            'min_on': 2,
            'min_off': 5
        },
    }
    return ukdale_appliance_data

def getUkdaleAggregateAttribution():
    '''
    return:
        aggregate_mean = 522
        aggregate_std = 814
        aggregate_cutoff = 6000
    '''
    aggregate_mean = 522
    aggregate_std = 814
    aggregate_cutoff = 6000
    return aggregate_mean, aggregate_std, aggregate_cutoff

def getMeanStd():
    ukdale_appliance_data = getUkdaleAppliancesAttribution()
    aggregate_mean, aggregate_std, aggregate_cutoff = getUkdaleAggregateAttribution()

    mean = np.array([ukdale_appliance_data['washingmachine']['mean'], 
                     ukdale_appliance_data['dishwasher']['mean'],
                     ukdale_appliance_data['kettle']['mean'],
                     ukdale_appliance_data['fridge']['mean'],
                     ukdale_appliance_data['microwave']['mean'],
                     aggregate_mean])
                        
    std  = np.array([ukdale_appliance_data['washingmachine']['std'], 
                     ukdale_appliance_data['dishwasher']['std'],
                     ukdale_appliance_data['kettle']['std'],
                     ukdale_appliance_data['fridge']['std'],
                     ukdale_appliance_data['microwave']['std'],
                     aggregate_std])

    return mean, std


if __name__ == '__main__':
    mean, std = getMeanStd()
    print(mean, std)