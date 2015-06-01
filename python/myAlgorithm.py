#!/usr/bin/env python

# in this module, I try to implement some algorithms that commonly use

def bubbleSort(a) :
	"bubble sort" 
	flag = True
	f = 0
	while flag:
		flag = False
		f += 1
		for x in range(1,len(a)-f) :
			if a[x-1] > a[x] :
				a[x-1],a[x] = a[x],a[x-1]
				flag = True


def bubbleSort_1(a) :
	k = len(a)
	flag = True
	while flag :
		flag = False
		for x in range(1,k) :
			if a[x-1] > a[x] :
				a[x-1],a[x] = a[x],a[x-1]
				flag = True
				k = x

def randn(n,begin,end) :
	a = []
	for x in range(n) :
		a.append(random.randint(begin,end))
	return a

def insertSort(a) :
	for x in range(1,len(a)) :
		f = False
		for y in range(x) :
			if a[x] < a[y] :
				f = True
				break;
		p = a[x]
		if f :
			for z in range(x-1,y-1,-1) :
				a[z+1] = a[z]
			a[y] = p

def insertSort(a) :
	for x in range(1,len(a)) :
		if a[x] < a[x-1] :
			p = a[x]
			for y in range(x-1,-1,-1) :
				if p < a[y]:
					a[y+1] = a[y]
				else :
					break
			if y == x-1:
				a[y] = p
			else :
				a[y+1] = p

def calNoneWhiteCharacter(fd) :
	"compute the total none white charaters in file fd"
	"fd is file descriptor"
	fd.seek(0)
	return sum((len(word) for line in fd for word in line.split()))

def getfactors(n) :
	"return all the factors of n, including 1 and n"
	return [x for x in range(1,n+1) if n%x == 0]

def stepMul(N) :
	"compute N!"
	return reduce(lambda x,y:x*y,range(1,N+1))

def listfile(path,sep = "..") :
	if not os.path.exists(path) :
		print "none exsiting path"
	items = os.listdir(path)
	for item in items :
		if item.startswith('.') :
			continue
		name = path + os.sep + item 
		if os.path.isdir(name) :
			listfile(name,sep+"..")
		print "%s%s" % (sep,name)

def dispFileN() :
	"display front N lines of the file "
	N = raw_input("input N : ")
	fileName = raw_input("input file name : ")
	x = 0
	try :
		fd = open(fileName,"r")
	except EOFError :
		return
	while x < int(N) :
		line = fd.readline().strip(os.linesep)
		if line :
			print line
		else :
			print "total lines in this file are less than %d" % N
			break
		x += 1

def countLine(fileName) :
	"compute total lines of the specified file"
	return len([line for line in open(fileName)])

def readFilebyPage(filename) :
	if not os.path.isfile(filename) :
		print "none exsiting file"
		return
	fd = open(filename,"r")
	nlines = 25
	line = 0
	for s in fd  :
		print s.strip(os.linesep)
		line += 1
		if line >= nlines :
			ch = raw_input("press any key to continue...")
			line = 0


def calTime(fun) :
	"calculate the time consumed of function 'fun' using decorator"
	def _fun(*arg) :
		begin = time.clock()
		result = fun(*arg)
		end = time.clock()
		print "*"*20
		print "time consumed : %f" % (end-begin)
		print "*"*20
		return result
	return _fun

def testFun(fun,*noneKeyVar,**keyVar) :
	'''This fuction is used to test whether function 'fun' is executed cprrectly
if succeed, return (True, funResult). otherwise, return (False, reason to fail)'''
	try :
		result = fun(*noneKeyVar,**keyVar)
		return (True, result)
	except Exception, msg:
		return (False,msg)


def makeFilter(string) :
	"This closure is used to find a specified string in a given file"
	def filterDoc(fileName) :
		fd = open(fileName,"r")
		lines = fd.readlines()
		fd.close()
		return [(index,line.strip(os.linesep).strip('\t')) for index,line in enumerate(lines) if string in line]
	return filterDoc
