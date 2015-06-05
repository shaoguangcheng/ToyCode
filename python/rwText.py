#!/usr/bin/env python

# This module is used to create or read text file

import os

def writeText():
	"This function is used to create a text file"

	ls = os.linesep # make the program more suitable in different platform

	#get file name
	while True:
		fname = raw_input("please input file name : ")
		if os.path.exists(fname) :
			print "ERROR : '%s' already exists" % fname
		else :
			break

	#get file content
	all = []
	print "\nEnter lines('.' to quit)\n"

	#loop utill user terminates input
	while True :
		entry = raw_input(">")
		if entry == '.' :
			break
		else :
			all.append(entry)

	#write lines to file
	fobj = open(fname,"w")
	fobj.writelines(["%s%s" % (x,ls) for x in all])
	fobj.close()
	print "DONE"

def readText() :
	"This function is used to read a text file"

	#get file name
	fname = raw_input("please input file name : ")
	print 

	#read file
	try :
		fobj = open(fname,"r")
	except IOError, e :
		print "*** file open error : ", e
	else :
		for eachline in fobj :
			print eachline,
		fobj.close()
		
	#	fobj.seek(0)

	#	all = fobj.readlines()
	#	fobj.close()
	#	for x in all :
	#		print x,


# The main function is used to test if the function is right
def main() :
	"test functions"
	print "w : create an empty text file and write some thing in it"
	print 
	print "r : read content from an existed text file"

	choice = raw_input("please input your choice : ")

	if choice == "w" :
		writeText()
	elif choice == "r" :
		readText()
	else :
		print "input error"
		quit()


#  This part canbe executed no matter which way you choose 
# jundge this module is recall by which way
if __name__ == "main" :
	main()

