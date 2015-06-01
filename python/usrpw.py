#!/usr/bin/env python

"Implement iteract UI in this script"

database = {}
def loadData() :
    try :
        fobj = open('./_database_','r')
    except (EOFError):
        print "EOFError"

    item = fobj.readline().strip()
    while item != "" :
        item = item.split(' ')
        database.setdefault(item[0],item[1])
        item = fobj.readline().strip()
    fobj.close()

def newuser() :
    prompt = "login desired :"
    while True :
        name = raw_input(prompt)
        if name in database :
            print "This name has existed, please choose another"
            continue
        else :
            break
    
    while True :
        pwd_1 = raw_input("passwd : ")
        pwd_2 = raw_input("please input the password again : ")
        if pwd_1 == pwd_2 :
            database[name] = pwd_1
            saveuser(name,pwd_1)
            break 
        else :
            print "your input is wrong"


def olduser() :
    name = raw_input("login : ")
    pwd  = raw_input("password : ")
    password = database.get(name)
    if pwd == password :
        print "\nwelcome back"
    else :
        print "\nlogin incorrectly"

def saveuser(name,pwd) :
    try :
        fobj = open('./_database_','a')
    except (EOFError) :
        print "EOFError"

    fobj.writelines(name+" "+pwd+"\n")
    fobj.close()
      
CMDs = {'n' : newuser, 'e' : olduser}
def showMenu() :
    loadData()
    prompt = """
(N)ew user login
(E)xisting user login
(Q)uit

Enter choice :"""
    while True :
        while True :
            try:
                choice = raw_input(prompt).strip()[0].lower()
            except (EOFError,KeyboardInterrupt) :
                choice = 'q'
            if choice not in "neq" :
                print "\nwrong input, please input again"
                continue
            else :
                break
        print "\nyour choice is %s" % choice
        if choice == "q" :
            break
        else :
            CMDs[choice]()

if __name__ == "__main__" :
  showMenu()
