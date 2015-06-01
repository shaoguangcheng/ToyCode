#!/usr/bin/env python

def createFile():
	while True :
		filename = raw_input("input file name : ")
		if os.path.exists(filename) :
			print "file has exsited already,please input another"
			continue
		else :
			break
	try :
		fd = open(filename,"w")
	except EOFError :
		return
	print "now you can input content to this file ... "
	while True :
		line = raw_input(">")
		if line.strip() :
			fd.write(line + os.linesep)
		else :
			break
	fd.close()

def dispFile() :
	while True :
		filename = raw_input("input file name : ")
		if not os.path.exists(filename) :
			print "file do not exsits"
			break
		else :
			break
	print "*"*20
	for line in open(filename) :
		print line.strip(os.linesep)

def editFile() :
	while True :
		filename = raw_input("input file name : ")
		if not os.path.exists(filename) :
			print "file do not exsits"
			break
		else :
			break
	fd = open(filename,"r")
	alllines = fd.readlines()
	fd.close()
	while True :
		lineNumberStr = raw_input("enter line you want to edit : ")
		if lineNumberStr == 'q' :
			break
		lineNumber = int(lineNumberStr)
		content = raw_input("enter new content : \n>")
		content += os.linesep
		if lineNumber > len(alllines) :
			alllines.append(content)
		else :
			alllines[lineNumber-1] = content
	fd = open(filename,"w")
	fd.truncate()
	fd.writelines(alllines)
	fd.close()

def saveFile() :
	pass

def showMenu() :
	prompt = '''
*********************
(1) create file
(2) display file
(3) edit file
(4) save file
(5) quit
*********************
enter your choice : '''
	while True :
		choice = raw_input(prompt)
		if choice == 'q' :
			break
		try :
			choiceInt = int(choice)
		except ValueError :
			print "Error : have not this choice, input another ... "
			continue
		if choiceInt == 1 :
			createFile()
		elif choiceInt == 2 :
			dispFile()
		elif choiceInt == 3 :
			editFile()
		elif choiceint == 4 :
			pass


if __name__ == "__main__" :
	showMenu()
