#!/usr/bin/env python
"test stack"

stack = []

def push() :
    "push"
    stack.append(raw_input("Enter new string : ").strip())

def pop() :
    "pop"
    if len(stack) <= 0:
        print "can not pop an element from an empty stack"
    else :
        print "removed [",repr(stack.pop()),"]"

def view() :
    "show all element in stack"
    print repr(stack)

CMDs = {'u':push,'o':pop,'v':view}

def showMenu() :
    pr = """
p(U)sh
p(O)p
(V)iew
(Q)uit

Enter choice : """


    while True :
        while  True :
            try :
                choice = raw_input(pr).strip().lower()
            except (EOFError,KeyboardInterrupt,IndexError) :
                choice = 'q'
            
                print "\n You picked : [%s]" % choice
            if choice not in "uovq" :
                print "Invalid option, try again"
            else :
                break

        if choice == 'q' :
            break

        CMDs[choice]()

if __name__ == '__main__' :
    showMenu()
