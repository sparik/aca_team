#!/usr/bin/python3
import argparse
import sys
import os

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-s', '--separator', default=',',
    	help="Separator to use in fields argument")
    parser.add_argument('-o', '--output_file', default=None,
    	help="Output file. If not specified, output will be written to stdout")
    parser.add_argument('-f', '--fields', default=None,
    	help="Used to specify fields to cut (include in the output)")
    parser.add_argument('file', nargs='?', 	default=None, 
    	help="Input file. If not specified, input will be read from stdin")
    parser.add_argument('-c', '--complement', action='store_true',
    	help="If this flag is set, all fields will be shown except the ones specified by --fields")
    parser.add_argument('-u', '--unique', action='store_true',
    	help="If this flag is set, no field will be shown more than once")

    args = parser.parse_args()
    return args


def process_lines(instream, outstream, fields_idx, needed_fields):
	"""
	read from instream line by line, and write to outstream
	the fields with indices in fields_idx. Names of the fields
	are in needed_fields
	"""
	print(','.join(needed_fields), file=outstream)

	for line in instream:
		features = line.replace('\n', '').split(',')
		needed_features = [features[i] for i in fields_idx]
		print(','.join(needed_features), file=outstream)

def unique_in_order(a):
	"""
	removes repeating elements of a, leaves only unique ones,
	and preserves order.
	"""
	d = {}
	res = []
	for x in a:
		if not x in d:
			d[x] = 1
			res.append(x)

	return res


def cut(instream, args):
	"""
	perform the cut specified by args, reading the file from instream
	initialize fields of the dataset
	initialize the fields to cut (needed_fields)
	process_lines
	"""

	fields = instream.readline()
	fields = fields.replace('\n', '')
	fields = fields.split(',')

	needed_fields = parse_needed_fields(fields, args.fields, args.separator)


	complement = args.complement
	unique = args.unique

	if unique is True:
		needed_fields = unique_in_order(needed_fields)

	if complement is True:
		needed_fields = [f for f in fields if f not in needed_fields]

	fields_idx = [fields.index(f) for f in needed_fields]

	if not args.output_file is None:
		with open(args.output_file, 'w') as f:
			process_lines(instream, f, fields_idx, needed_fields)
	else:
		process_lines(instream, sys.stdout, fields_idx, needed_fields)

	return 0

def parse_needed_fields(fields, fields_arg, separator):
	"""
    :param fields: names of fields of the dataset
    :param fields_arg: the argument fields: a string 
    specifying which fields to leave
    :param separator: the argument separator: a string specifying 
    the separator between fields to cut
	:return: list of field names of the resulting cut
	"""

	res = []

	if not fields_arg is None:
		split_fields = fields_arg.split(separator)
	else:
		return res

	for x in split_fields:
		if x.find("-") != -1:
			interval = x.split("-")
			if len(interval) > 2:
				raise Exception("bad value for fields argument. Wrong interval '%s'" % x)
			else:
				start_f = interval[0]
				end_f = interval[1]
				if not start_f:
					start_ind = 0
				else:
					if not start_f in fields:
						raise Exception("bad value for fields argument. Wrong field '%s'" % start_f)
					start_ind = fields.index(start_f)

				if not end_f:
					end_ind = len(fields) - 1
				else:
					if not end_f in fields:
						raise Exception("bad value for fields argument. Wrong field '%s'" % end_f)
					end_ind = fields.index(end_f)

			for i in range(start_ind, end_ind + 1):
				res.append(fields[i])
		else:
			if not x in fields:
				raise Exception("bad value for fields argument. Wrong field '%s'" % x)
			res.append(x)
				


	return res


def main():
	args = parse_args()

	if not args.file is None:
		if not os.path.isfile(args.file):
			print("%s: specified input file does not exist" % (os.path.basename(__file__)))
			sys.exit(1)
		with open(args.file, 'r') as f:
			try:
				cut(f, args)
			except Exception as ex:
				print("%s: %s" % (os.path.basename(__file__), str(ex)), file=sys.stderr)
				sys.exit(1)
	else:
		try:
			cut(sys.stdin, args)
		except Exception as ex:
			print("%s: %s" % (os.path.basename(__file__), str(ex)), file=sys.stderr)
			sys.exit(1)




if __name__ == '__main__':
	main()