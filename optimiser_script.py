from base_module import *
import pandas as pd
import shelve

# Live file folder path.

live_folder = 'Live Files//'


# Live data files are imported.

live_inventory = pd.read_csv(live_folder + 'live_inventory.csv', header=0)
live_inventory.columns = ['entity', 'ISIN', 'currency', 'depot_notional', 'custodian_notional']

live_fx = pd.read_csv(live_folder + 'live_fx.csv', header=0)
live_fx.columns = ['currency', 'USD_FX_rate']

live_prices = pd.read_csv(live_folder + 'live_prices.csv', header=0)
live_prices.columns = ['ISIN', 'closing_price']

live_cost_factors = pd.read_csv(live_folder + 'live_cost_factors.csv', header=0)
live_cost_factors.columns = ['ticker', 'cost_factor']

live_priority_securities = pd.read_csv(live_folder + 'live_priority_securities.csv', header=0)
live_priority_securities.columns = ['ISIN']
live_priority_securities['priority_status'] = 'Yes'

live_margin = pd.read_csv(live_folder + 'live_margin.csv', header=0)
live_margin.columns = ['entity', 'counterparty_ID', 'RQV', 'margin_currency']


# User inputs comma-separated list of holiday currencies.
# This will limit the available notional to the value held at Custodian for all ISINs of the listed currencies.

holiday_list = [str(x).upper() for x in input('\n' + 'Enter list of holiday currencies (separated by comma):' + '\n').split(', ')]


# Security objects are created.

security_database = shelve.open('security_profile_database')

security_list = [Security(live_inventory.loc[i], live_cost_factors, live_priority_securities, holiday_list, live_fx, live_prices, security_database) \
                 for i in range(len(live_inventory))]

security_database.close()


# Security objects are split by entity and loaded into separate Inventory objects.

inventory_entities = live_inventory.entity.unique()
index_securities_by_entity = []

for i in range(len(inventory_entities)):
    index_securities_by_entity.append([j for j, x in enumerate(security_list) if x.entity == inventory_entities[i]])

inventory_list = [Inventory(inventory_entities[i], [security_list[j] for j in index_securities_by_entity[i]]) \
                  for i in range(len(inventory_entities))]


# Margin objects are created.

margin_list = [Margin(live_margin.loc[i], live_fx) \
               for i in range(len(live_margin))]


# Margin objects are split by entity and loaded into separate Margin_Set objects.

margin_entities = live_margin.entity.unique()
index_counterparties_by_entity = []

for i in range(len(margin_entities)):
    index_counterparties_by_entity.append([j for j, x in enumerate(margin_list) if x.entity == margin_entities[i]])

margin_set_list = [Margin_Set(margin_entities[i], [margin_list[j] for j in index_counterparties_by_entity[i]]) \
                   for i in range(len(margin_entities))]


# Desired longbox excesses are entered by user.

for i in range(len(margin_set_list)):
    margin_set_list[i].Add_Longbox(int(input('\n' + 'Please enter desired minimum ' + margin_set_list[i].entity + ' longbox excess (USD):'+ '\n')))


# Entity Inventory and Margin_Set objects are packaged into Funding_Set objects:

funding_set_list = [Funding_Set(margin_entities[i], inventory_list[i], margin_set_list[i]) \
                   for i in range(len(margin_entities))]


# Matrices depending on shelved counterparty profile data are generated.
# Concentration limit schedules are retrieved.

counterparty_database = shelve.open('counterparty_profile_database')

for i in range(len(funding_set_list)):
    funding_set_list[i].Generate_Haircut_Matrix(counterparty_database)
    funding_set_list[i].Generate_Weight_Matrices()
    
    for j in range(funding_set_list[i].margin_set.size):
            funding_set_list[i].margin_set.margins[j].Retrieve_Concentration_Limit_Schedule(counterparty_database)

counterparty_database.close()


# Funding routine is run for each Funding_Set.

security_database = shelve.open('security_profile_database')

for i in range(len(funding_set_list)):

    Run_Funding_Routine(Funding_Set=funding_set_list[i], FX_data=live_fx, security_database=security_database, folder=live_folder)

security_database.close()

print('\n' + 'Funding routines complete for all entities.')