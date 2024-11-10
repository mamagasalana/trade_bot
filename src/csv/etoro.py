# URL = 'https://api.etorostatic.com/sapi/instrumentsmetadata/V1.1/instruments/bulk?bulkNumber=1&cv=68fce28ea7acbb8a6e3c47b243cd9d77_925a60096663c53ba1f879cf6075b245&totalBulks=1'

import json
js = json.load(open('bulk.json'))

out = {}
for j in js['InstrumentDisplayDatas']:
    out[j['SymbolFull']] = j['InstrumentID']

with open('INSTRUMENT_MAP.json', 'w') as ofile:
    ofile.write(json.dumps(out))