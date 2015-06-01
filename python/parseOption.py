#!/usr/bin python

from optparse import OptionParser as OP
import sys

def parseOption(argv) :
	parser = OP()
	
	parser.add_option("-f", "--file", dest = "filename", action = "store", default = "text",\
			metavar = "FILE", help = "write to file")

	parser.add_option("-n", type = "int", dest = "num", metavar = "number",\
			help = "a integer", default = 12)

	parser.add_option("-v", action = "store_false", dest = "verbose", metavar = "bool")

	(options, args) = parser.parse_args()
	
	print options.filename
	print options.num
	print options.verbose

if __name__ == "__main__" :
	parseOption(sys.argv)
	
