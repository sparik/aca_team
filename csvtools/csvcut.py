#!/usr/bin/python3

"""
this script can be used to select specific columns from a csv file
"""

import argparse
import sys
import os

def parse_args():
    """
    parse arguments to the script and return them
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('-s', '--separator', default=',',
                        help="Separator of the input file. Also used in the output file")
    parser.add_argument('-o', '--output_file', default=None,
                        help="Output file. If not specified, output will be written to stdout")
    parser.add_argument('-f', '--fields', default=None,
                        help="Used to specify fields to cut (include in the output)")
    parser.add_argument('file', nargs='?', default=None,
                        help="Input file. If not specified, input will be read from stdin")
    parser.add_argument('-c', '--complement', action='store_true',
                        help="If this flag is set, all fields will be shown except the ones specified by --fields")
    parser.add_argument('-u', '--unique', action='store_true',
                        help="If this flag is set, no field will be shown more than once")

    args = parser.parse_args()
    return args


def process_lines(instream, outstream, fields_idx, needed_fields, separator):
    """
    read from instream line by line, and write to outstream
    the fields with indices in fields_idx. Names of the fields
    are in needed_fields
    :param instream: input stream to read the file from
    :param outstream: output stream to write the output file to
    :param fields_idx: a list of indexes of fields to select
    :param needed_fields: names of the fields to select
    :param separator: separator of the input file. Also used in the output file
    """
    print(separator.join(needed_fields), file=outstream)

    for line in instream:
        features = line.strip().split(separator)
        needed_features = [features[i] for i in fields_idx]
        print(separator.join(needed_features), file=outstream)

def unique_in_order(lst):
    """
    removes repeating elements of a, leaves only unique ones,
    and preserves order.
    :param lst: a list (is not changed)
    :return: a new list created from the provided list's unique elements
    """
    d = {}
    res = []
    for elem in lst:
        if not elem in d:
            d[elem] = 1
            res.append(elem)

    return res


def cut(instream, args):
    """
    perform the cut specified by args, reading the file from instream
    initialize fields of the dataset
    initialize the fields to cut (needed_fields)
    process_lines
    :param instream: stream to read the csv file from
    :param args: arguments of the cut
    """

    fields = instream.readline()
    fields = fields.strip()
    fields = fields.split(args.separator)

    needed_fields = parse_needed_fields(fields, args.fields, ",")

    complement = args.complement
    unique = args.unique

    if unique is True:
        needed_fields = unique_in_order(needed_fields)

    if complement is True:
        needed_fields = [f for f in fields if f not in needed_fields]

    fields_idx = [fields.index(f) for f in needed_fields]

    if not args.output_file is None:
        with open(args.output_file, 'w') as f:
            process_lines(instream, f, fields_idx, needed_fields, args.separator)
    else:
        process_lines(instream, sys.stdout, fields_idx, needed_fields, args.separator)

def parse_needed_fields(fields, fields_arg, separator):
    """
    :param fields: names of fields of the dataset
    :param fields_arg: the argument fields: a string
    specifying which fields to leave
    :param separator: separator of the fields argument. "," by default
    :return: list of field names of the resulting cut
    """

    res = []

    if not fields_arg is None:
        needed_intervals = fields_arg.split(separator)
    else:
        return res

    # needed_intervals - list of needed fields and ranges
    # e.g. ["a", "c", "b-d", 5-10, 3]

    for interval in needed_intervals:
        if "-" in interval:
            try:
                start_f, end_f = interval.split("-")
            except:
                raise Exception("bad value for fields argument. Wrong interval '%s'" % interval)
            if not start_f:
                start_ind = 0
            else:
                if start_f.isdigit():
                    start_ind = int(start_f) - 1
                elif not start_f in fields:
                    raise Exception("bad value for fields argument. Wrong field '%s'" % start_f)
                else:
                    start_ind = fields.index(start_f)

            if not end_f:
                end_ind = len(fields) - 1
            else:
                if end_f.isdigit():
                    end_ind = int(end_f) - 1
                elif not end_f in fields:
                    raise Exception("bad value for fields argument. Wrong field '%s'" % end_f)
                else:
                    end_ind = fields.index(end_f)

            start_ind = max(start_ind, 0)
            end_ind = min(end_ind + 1, len(fields))

            for i in range(start_ind, end_ind):
                res.append(fields[i])
        else:
            needed_field = interval
            if not needed_field in fields:
                raise Exception("bad value for fields argument. Wrong field '%s'" % needed_field)
            res.append(needed_field)

    return res


def main():
    args = parse_args()

    try:
        if not args.file is None:
            if not os.path.isfile(args.file):
                print("%s: specified input file does not exist" % (os.path.basename(__file__)))
                sys.exit(1)
            with open(args.file, 'r') as file:
                cut(file, args)
        else:
            cut(sys.stdin, args)
    except BrokenPipeError:
        pass
    except Exception as ex:
        print("%s: %s" % (os.path.basename(__file__), str(ex)), file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
