#!/usr/bin/env python

# This module is used to test whether an identifier is valid

import string
import keyword

def idcheck(identifier) :
	"This function is used to test whether an identifier is valid"

	alpha = string.join(string.ascii_letters,'_');
	number = string.digits;
	alphaNum = string.join(alpha,number);
	keyWord = keyword.kwlist

# whether identifier is a keyword
	if identifier in keyWord :
		print "ERROR: '%s' is a keyword " % identifier
		return False

# when length is 1
	if len(identifier) == 1 and identifier not in alpha :
		print "ERROR: '%s' is invalid identifier" % identifier
		return False;

# when length larger than 1
	if identifier[0] not in alpha :
		print "ERROR: '%s' is invalid identifier" % identifier
	else :
		for x in identifier[1:] :
			if x not in alphaNum :
				print "EEROR: '%s' is invalid identifier" % identifier
				return False;

	return True;

# test function 
def main() :
	id = raw_input("please input an identifier : ")
	if idcheck(id) :
		print "valid id"

if __name__ == "__main__" :
	main();

	