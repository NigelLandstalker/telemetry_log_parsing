import os
import yaml

YAML_PATH = os.path.join('..','..','software','cangen','packets') #MAY NEED TO CHANGE YAML PATH IF THE FILE STRUCTURE CHANGES

c_lengths = {
	'uint8_t' : 1,
	'int8_t' : 1,
	'uint16_t' : 2,
	'int16_t' : 2,
	'uint32_t' : 4,
	'int32_t' : 4,
	'uint64_t' : 8,
	'int64_t' : 8,
	'bitfield' : 1,
	'float' : 4
}
	
def loadAsList(expandRepeatPackets=False):
	allpackets = {}
	allpackets['packets'] = []
	for file in os.listdir(YAML_PATH):
		if file.endswith('.ignore.yaml') or not file.endswith('.yaml'):
			continue
		f = open(os.path.join(YAML_PATH, file), 'r')
		yaml_file = yaml.load(f)
		for packet in yaml_file['packets']:
			packet['file'] = file
			if expandRepeatPackets and 'repeat' in packet:
				baseId = packet['id']
				baseName = packet['name']
				if 'offset' in packet:
					offset = packet['offset']
				else:
					offset = 1
				for i in range(packet['repeat']):
					newPacket = packet.copy()
					newPacket['id'] = baseId+i*offset
					newPacket['name'] = baseName + "__" + str(i)
					allpackets['packets'].append(newPacket)
					del newPacket['repeat']
			else:
				try:
					for i in range(len(packet['name'])):
						newpacket = packet.copy()
						newpacket['name'] = packet['name'][i]
						newpacket['id'] = packet['id'][i]
						allpackets['packets'].append(newpacket)
				except:
					allpackets['packets'].append(packet)
	return allpackets
