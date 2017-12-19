#TASKS/ISSUES
#Add a key error counter for packet ids. I've disabled the print for invalid ids for now since it was annoying.
#It's a pain to have 0x precede every hex ID.
#Reverse polish notation argument mathematics for graph generation.
#FFS Implement signed integer parsing. 
#implement multithreading

#NON NATIVE IMPORTS (CHECK THESE ARE INSTALLED IF THE PROGRAM DOESN'T WORK!)
import yaml
import numpy as np

#Native imports
import argparse #For CLI interface.
import csv
import sys
import struct
import binascii
import os
import ast
import math
import time
import datetime

WSC_2017_TZ_OFFSET = 52200

#Global Variables
DEFAULT_MAX_GRAPH_POINTS = 10000
TIME_RESOLUTION_SECONDS  = 0
TZ_OFFSET = WSC_2017_TZ_OFFSET

#Graphing Variables
GRAPH_FONT_SIZE = 15

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

#UTILITY FUNCTIONS: 
def timestamp_to_datetime(timestamp):
	return datetime.datetime.fromtimestamp(float(timestamp) + TZ_OFFSET) 
	
def ts_to_mm_dd_hh_mm(timestamp):
	dt = timestamp_to_datetime(timestamp)
	return "{0}-{1}-{2}-{3}".format(dt.month, dt.day, dt.hour, dt.minute)
	
def yyyy_mm_dd_hh_to_ts(yyyy_mm_dd_hh):
	try:
		time_array = [int(x) for x in yyyy_mm_dd_hh.split('-')]
		d = datetime.date(time_array[0], time_array[1], time_array[2])
		t = datetime.time(time_array[3], 0) #Keep minutes at 0 for this instance
		dt = datetime.datetime.combine(d, t)
		return time.mktime(dt.timetuple())
	except:
		print("Error generating timestamps from that time range. Try checking the time range formatting")
	
def diff(a, b):
	return abs(a - b)
	
def floor(a): #A not stupid floor function that actually returns an int.
	return int(math.floor(a))
	
def loadAsList(expandRepeatPackets=False):
	allpackets = {}
	allpackets['packets'] = []
	for file in os.listdir('packets/'):
		if file.endswith('.ignore.yaml') or not file.endswith('.yaml'):
			 continue
		f = open('packets/' + file, 'r')
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
	
class Packet():
	def __init__(self, timestamp, data):
		self.data = data
		self.timestamp = timestamp
		self.f_ts = float(timestamp)
		
	def get_var(self, var):
		if var in self.data:
			return float(self.data[var])
		else:
			raise KeyError("Packet variable keyword {0} not found in packet.".format(var))

class Packet_List():
	def __init__(self, id):
		self.id   = id
		self.list = []
		self.length = len(self.list)

	def get_timestamp_placement(self, timestamp): #Returns the index in the list where the entry should be added, should one exist. Otherwise, returns None
		if self.length == 0 or float(timestamp) < self.list[0].f_ts:
			return 0
		elif float(timestamp) - self.list[self.length - 1].f_ts > TIME_RESOLUTION_SECONDS: #Check if the new element needs to be sorted at all.
			return self.length #Return the index of the list where the object will be placed.
		else:
			#Perform a binary search to decide where to place the new timestamp
			def bin_search_range(sorted_list, left, right, value):
				mid = floor((left + right) / 2) 
				if abs(right - left) <= 1: #If we have narrowed it down to between two adjacent indices. Here we know mid is on one of the two bounds for where value fits.
					if diff(value, sorted_list[right].f_ts) > TIME_RESOLUTION_SECONDS and diff(value, sorted_list[left].f_ts) > TIME_RESOLUTION_SECONDS:
						return right
					else:
						return None #If the value doesn't fit, skip it. 
				if value > sorted_list[mid].f_ts: 
					return bin_search_range(sorted_list, mid, right, value)
				elif value < sorted_list[mid].f_ts:
					return bin_search_range(sorted_list, left, mid, value) 
				else:
					return None
			return bin_search_range(self.list, 0, self.length - 1, float(timestamp))
			
	def add(self, packet, index):
		if isinstance(packet, Packet):
			if index == None:
				self.list.append(packet)
				self.length += 1
			else:
				if index >= 0 and index <= self.length:
					self.list.insert(index, packet)
					self.length += 1
				else:
					raise ValueError('Attempted to add packet to packetlist at an invalid index')
		else:
			raise TypeError
			
	def get_var(self, id):
		return [packet.get_var(var) for packet in self.list]
	
	def get_array(self, index, readable=False):
		if readable:
			return [timestamp_to_datetime(self.list[index].timestamp).isoformat(), self.list[index].data]
		else:
			return [self.list[index].timestamp, self.list[index].data]
	
	def timestamps(self):
		return [packet.f_ts for packet in self.list]
	
	def readable_timestamps(self):
		return [timestamp_to_datetime(packet.f_ts).isoformat() for packet in self.list]
	
	def data(self):
		return [packet.data for packet in self.list]
		
	#Returns a set of a specified number of data points from the list evenly spaced within the time range.
	def get_even_timerange(self, max_points, start_time, stop_time):
		if (start_time == None or stop_time == None):
			start_time = min(self.timestamps())
			stop_time  = max(self.timestamps())
			time_cropped_list = self.list
		elif stop_time >= start_time:
			time_cropped_list = list(filter(lambda x: x.timestamp >= start_time and x.timestamp <= stop_time, self.list))		else:
			raise ValueError("Cannot filter data by a negative time range.")
		if max_points == None or int(max_points) > len(time_cropped_list):
			return time_cropped_list #In case maximum points is too large, we don't need to remove any data points for our range. 
		filtered_points = []
		time_between_points = (stop_time - start_time) / int(max_points)
		time = start_time
		while time_cropped_list != []:
			filtered_points.append(time_cropped_list[0])
			time += time_between_points
			time_cropped_list = list(filter(lambda x: x.timestamp >= time, time_cropped_list))
		return filtered_points
		
#Bit of a kludge, but it just needs to work at this point. 
#Takes the logged packet and synchronizes them to a common timestamp.
#Returns a new dictionary  
def synchronize_packets(logged_packets): 
	#Synchronize to the packet id with the lowest resolution. 
	def id_with_min_length(logged_packets): #Get the ID with the lowest resolution
		ids = logged_packets.keys()
		min_id = ids[0]
		min_length = logged_packets[min_id].length
		for id in ids:
			current_length = logged_packets[id].length
			if current_length < min_length:
				min_id = id
				min_length = current_length
		return min_id	
		
	min_len_id = id_with_min_length(logged_packets) #Min len id is the lowest resolution packet id of all the logged packets. Use its timestamps to synchronize the rest. 
	sync_timestamps = logged_packets[min_len_id].timestamps() #This is SORTED
	
	new_logged_packets = {}
	for id in logged_packets: #Cycle through all the ids.
		print("Synchronizing ID: {0}".format(id))
		packet_list = logged_packets[id].list #Sorted
		new_packet_list = Packet_List(id)
		packet_list_index = 0
		skipped_entries   = 0
		last_entry        = None
		for s_ts in sync_timestamps: #Each synchronized timestamp will have one piece of data accompanying it per id.
			try: 
				while packet_list[packet_list_index].f_ts < s_ts:
					skipped_entries   += 1
					packet_list_index += 1
			except IndexError:
				new_packet_list.add(packet_list[len(packet_list) - 1], None)
				break
				
			matched_packet = packet_list[packet_list_index]
			packet_list_index += 1
			matched_packet.timestamp = s_ts
			
			assert last_entry is not matched_packet
			last_entry = matched_packet
			 
			new_packet_list.add(matched_packet, None)
		print("Percentage skipped: {0}\n".format(100 * (float(skipped_entries) / float(packet_list_index))))
		new_logged_packets[id] = new_packet_list
		assert new_packet_list.list[0].timestamp == sync_timestamps[0]
	return new_logged_packets
	
def do_math(logged_packets, math_operators, packet_defs):
	r_polish = math_operators.split(" ")
	valid_ops = ['+', '-', '/', '*']
	def eval_polish(op, left, right):
		if op not in valid_ops:
			raise OpError 
	try: 
		eval_polish()
	except: 
		pass
	#Math string should look like this: + 0x201:soc 0x201Ah
	

#Swap endianness for little endian packet data. 
#If little endian, then reverse the order of the packet BYTES.
def check_endianness(data_splice__, packet_def__):
	if 'endian' in packet_def__ and packet_def__['endian'] == 'little':
		swapped = ""
		for index in range(len(data_splice__) - 1, 0, -2):
			swapped += data_splice__[index - 1] + data_splice__[index]
		return swapped
	else:
		return data_splice__

def write_parsed_to_csv(logged_packets, csv_title, packet_defs):
	with open(csv_title + ".csv", "wb") as csvfile:
		titles = ['timestamp',]
		for id in logged_packets.keys():
			packet_def = packet_defs["0x{0}".format(id.lower())]
			for var in packet_def['data']:
				titles.append('{0}_{1}:{2}'.format(id, packet_def['name'], var['name']))
				
		writer = csv.DictWriter(csvfile, fieldnames=titles)
		writer.writeheader()
		
		to_write  = {}
		timestamps = logged_packets[logged_packets.keys()[0]].readable_timestamps()
		for index, ts in enumerate(timestamps):
			to_write['timestamp'] = ts
			for id in logged_packets.keys():
				packet_def = packet_defs["0x{0}".format(id.lower())]
				for var in packet_def['data']:
					try:
						to_write['{0}_{1}:{2}'.format(id, packet_def['name'], var['name'])] = logged_packets[id].list[index].get_var(var['name'])
					except IndexError:
						pass #Just ignore any index errors for now. They aren't too big of a deal when there are millions of entries. 
						
			writer.writerow(to_write)
		csvfile.close()
	print("CSV file {0} sucessfully written".format(args.c + ".csv"))
	
def read_csv(csv_title, logged_packets):
	with open(csv_title + ".csv", 'r') as csvfile:
		reader = csv.reader(csvfile)
		headers = reader.next()[1:]
		header_ids = [header.split(':')[0][:3] for header in headers]
		header_vars = [header.split(':')[1] for header in headers]
		id_var_dict = {}
		for header_id in header_ids:
			if header_id in id_var_dict:
				id_var_dict[header_id].append(header.split(':')[1])
			else:
				id_var_dict[header_id] = [header.split(':')[1],]

		for row in reader:
			if '.' in row[0]:
				dt = datetime.datetime.strptime(row[0], "%Y-%m-%dT%H:%M:%S.%f")
			else:
				dt = datetime.datetime.strptime(row[0], "%Y-%m-%dT%H:%M:%S")
			ts = time.mktime(dt.timetuple())
			row_packets = {id : Packet(ts, {var_name: None for var_name in id_var_dict[id]}) for id in header_ids}
			for index, data_ in enumerate(row[1:]):
				current_id = header_ids[index]
				current_var = header_vars[index]
				row_packets[current_id].data[current_var] = data_
			
			for id in row_packets:
				packet = row_packets[id]
				if id not in logged_packets.keys(): #Just in case the data was already parsed into logged_packets beforehand. Eventually replace this with sorting the data into existing lists. 
					new_packet_list = Packet_List(id)
					new_packet_list.add(packet, None)
					logged_packets[id] = new_packet_list
				else:
					logged_packets[id].add(packet, None)
		csvfile.close()
	return logged_packets
			
#Generates up to two pyplot graphs from parameters
def generate_pyplot_graphs(logged_packets, graphing_ids, graphing_vars, start_time, stop_time, max_points, packet_defs):
	import matplotlib.pyplot as plt
	import matplotlib

	graph_data = [logged_packets[id].get_even_timerange(max_points, start_time, stop_time) for id in graphing_ids]	
	x_values = map(timestamp_to_datetime, [packet.timestamp for packet in graph_data[0]]) #Doesn't matter which of the graph datasets we use here, since they are identical. 
	y_values = [[packet.get_var(graphing_vars[index]) for packet in graph_data[index]] for index in range(len(graphing_vars))] #Nested list comprehensions are fun
	dates    = matplotlib.dates.date2num(x_values)
	
	#Configure the graph
	matplotlib.rcParams.update({'font.size': GRAPH_FONT_SIZE})
	fig = plt.figure(figsize=(12, 6), dpi=100)
	host = fig.add_subplot(111)
	host.set_xlabel("Timestamp")
	
	plots = []
	axes_count = 0
	axes = host
	sides = ['left', 'right']
	for index in range(len(graphing_ids)):
		packet_def_vars = packet_defs['0x' + graphing_ids[index]]['data']
		for p_var in packet_def_vars:
			if p_var['name'] == graphing_vars[index]:
				try: 
					label_ = "{0}:{1} ({2})".format(graphing_ids[index], graphing_vars[index], p_var['units'])
				except KeyError:
					label_ = "{0}:{1}".format(graphing_ids[index], graphing_vars[index])
				finally: 
					break
		color_ = plt.cm.viridis(float(index) / float(len(graphing_ids)))
		p, = axes.plot_date(dates, y_values[index], color=color_, label = label_)
		plots.append(p)
		axes.set_ylabel(label_)
		axes.yaxis.label.set_color(color_)
		if index < len(graphing_ids) - 1:
			axes = host.twinx()
			axes_count += 1
			if axes_count > 1: 
				position = ('outward', 100 * (axes_count - 1))
				axes.spines['right'].set_position(position)
				
	axes.legend(handles=plots, loc='best')
	
	plt.title(' and '.join([var for var in graphing_vars]) + " vs time")
	plt.tight_layout()
	plt.show()
	
if __name__ == "__main__":
	packet_defs = {hex(packet["id"]) : packet for packet in loadAsList(expandRepeatPackets=True)['packets']} #BUILD THE PACKET DEFINITIONS FROM .yaml files

	#Basic command line interface.
	cli_parser = argparse.ArgumentParser(prog='parse_telemetry_log.py', description='Command-line interface for telemetry log parsing. Parses the files within the "logs" folder within the program\'s directory.')
	cli_parser.add_argument('-p', nargs='*', choices=packet_defs.keys().append('all'), help='A list of CAN packet IDs to include in the parsed csv file.')
	cli_parser.add_argument('-c', help='Enables parsing to csv file. Include the csv filename as the next argument. Don\'t add .csv')
	cli_parser.add_argument('-g', nargs='*', help='Enables Command-line interactive graphing of parsed Telemetry data via pyplot. \n Include a list of desired variable names. Variable names must be defined within the desired CAN packet IDs entered.')
	cli_parser.add_argument('-gt', nargs=2, help='Option to enter a timeframe for graph plotting. If left blank or undefined, the entire time range is used. Takes two arguments: Start, then stop. Format these strings in yyyy-mm-dd-hh.')
	cli_parser.add_argument('-gp', nargs=1, help='Option to enter a maximum number of data points for graph plotting. If left blank or undefined, a default value of {0} is used.'.format(DEFAULT_MAX_GRAPH_POINTS))
	cli_parser.add_argument('-m', nargs='*', help='Enables the user to perform mathematical operations on the data. Results are added to the csv file and/or graphed where desired.')
	cli_parser.add_argument('-r', help='Reads data from an existing .csv file into the program. Include the csv filename as the next argument. Don\'t add .csv')
	args = cli_parser.parse_args()

	#BEGIN PARSING CODE
	logged_packets  = {} #Dictionary of lists given by packet ID.
	if args.p != None:
		if args.p == []: 
			args.p = 'all'
		print("Parsing telemetry logs targeting IDs: {0}".format(args.p))
		
		#Setup data parsing
		skipped_entries, total_entries, error_entries = 0, 0, 0
		for logfilename in os.listdir('logs/'):
			if not logfilename.endswith('.txt'):
				continue
			with open('logs/' + logfilename, 'r') as f:
				lines = f.readlines()
				#First three hex values in the hex string will be the packet ID. Sort the packets by their ID
				file_entries = len(lines)
				total_entries += file_entries
				for logentry in lines:
				
					#Skip incomplete/obviously corrupt data entries.
					if len(logentry) < 2 or len(logentry.split()) < 2:
						error_entries += 1
						continue

					packet = logentry.split()[1]
					id = packet[:3].lstrip('0').lower() #Remove leading zeros from packet id.
					#Ensure that the packet id is valid.
					try:
						packet_def = packet_defs["0x{0}".format(id.lower())] #Get the correct packet def
					except KeyError:
						#print("{0} was read from file {1} but does not exist".format(id, logfilename))
						error_entries += 1
						continue
					#print("Want "+" 0x{0}".format(id.lower()))

					#Ensure that we want this packet.
					if args.p != 'all' and "0x{0}".format(id.lower()) not in args.p:
						skipped_entries += 1
						continue

						#Instantiate a packet list entry for this id if necessary
					if id not in logged_packets:
						logged_packets[id] = Packet_List(id)
					
					#Determine where to put the packet. Throw out packets that have timestamps which are too close to one another. 
					timestamp   = logentry.split()[0]
					packet_list_index = logged_packets[id].get_timestamp_placement(timestamp)
					if packet_list_index == None: 
						skipped_entries += 1
						continue

					#Parse the data
					parsed_data = {}
					hex_index   = 0
					data        = packet[3:]
					try:
						for entry in packet_def['data']:
							entry_type  = entry['type']
							hex_length  =  2 * c_lengths[entry_type]
							data_splice = data[hex_index : hex_index + hex_length]

							#Perform an additional check for corrupted data
							if len(data_splice) < hex_length:
								raise ValueError

							#Parse entry based on its data type:
							if(entry_type in {'uint8_t', 'uint16_t', 'uint32_t', 'uint64_t'}):
								data_splice = check_endianness(data_splice, packet_def)
								parsed_data[entry['name']] = int(data_splice, 16)
								if ('conversion' in entry):
									parsed_data[entry['name']] *= entry['conversion']
									
							elif(entry_type in {'int8_t', 'int16_t', 'int32_t', 'int64_t'}):
								data_splice = check_endianness(data_splice, packet_def)
								bitlength = int(''.join([s for s in entry_type if s.isdigit()]))
								data_splice = int(data_splice, 16)
								if data_splice >= 1 << (bitlength - 1): data_splice -= 1 << bitlength
								parsed_data[entry['name']] = data_splice
								if ('conversion' in entry):
									parsed_data[entry['name']] *= entry['conversion']
								
							elif(entry_type == 'bitfield'):
								parsed_data[entry['name']] = None #NOT IMPLEMENTED YET

							elif(entry_type == 'float'):
								parsed_data[entry['name']] = struct.unpack('<f', binascii.unhexlify(data_splice))[0]
								if ('conversion' in entry):
									parsed_data[entry['name']] *= entry['conversion']
							else:
								parsed_data[entry['name']] = None #Should actually be none.
							hex_index += hex_length
					except ValueError:
						error_entries += 1
						continue
						
					logged_packets[id].add(Packet(timestamp, parsed_data), packet_list_index)
			f.close()
			print("Telemetry logfile {0} successfully parsed".format(logfilename))
		print("\nFinished parsing telemetry data. \n Parsed entries: {3} \n Total entries: {0} \n Skipped entries: {1} \n Invalid entries: {2}".format(total_entries, skipped_entries, error_entries, total_entries - (skipped_entries + error_entries)))

	#END PARSING CODE
	
	if args.r != None:
		logged_packets = read_csv(args.r, logged_packets)

	print("\nSynchronizing timestamps.")
	logged_packets = synchronize_packets(logged_packets)	
		
	#Perform any desired computations specified in the CLI:
	if args.m != None:
		logged_packets = do_math(logged_packets, args.m, packet_defs)

	#Generate csv file if that option was selected in the CLI
	if args.c != None:
		print("Writing Telemetry data to CSV file")
		write_parsed_to_csv(logged_packets, args.c, packet_defs)

	#Generate graphs if that option was selected in the CLI and the arguments were valid.
	if args.g != None:
		graphing_ids = [arg.split(':')[0][2:] if '0x' in arg.split(':')[0] else arg.split(':')[0] for arg in args.g]
		graphing_vars = [arg.split(':')[1] for arg in args.g]
		print("\nGenerating graphs with variables: {0}".format(args.g))
		if args.gt != None:
			generate_pyplot_graphs(logged_packets, graphing_ids, graphing_vars, yyyy_mm_dd_hh_to_ts(args.gt[0]), yyyy_mm_dd_hh_to_ts(args.gt[1]), args.gp, packet_defs)
		else:
			generate_pyplot_graphs(logged_packets, graphing_ids, graphing_vars, None, None, args.gp, packet_defs)
	else:
		print("No graphing option selected or invalid graph parameters. Will not generate graphs")
		
#cd D:\SVN\elrond\telemetry\telemetry_log_parsing
#python parse_telemetry_log.py 0x201 -g soc