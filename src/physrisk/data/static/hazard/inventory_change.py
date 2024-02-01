
import json

f = open('inventory.json', 'r')
inv = json.load(f)
f.close()

for data_ in inv['resources']:
    data_['hazard_type'] = data_.pop('type')
    data_['indicator_id'] = data_.pop('id')
    data_['path'] = data_.pop('array_name')
    data_['map']['path'] = data_['map'].pop('array_name')
    data_['indicator_model_gcm'] = 'NA'

f = open('inventory.json', 'w')
json.dump(inv, f, sort_keys=True, indent=4)
f.close()
