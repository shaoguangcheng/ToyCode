#!/usr/bin/env python

"I try to write some functions for future use"

import string

def strip(s) :
    "This function is analogous with string.strip()"
    start = 0;
    while start < len(s) :
        if s[start] != " " :
            break
        start -= 1
        
    end = len(s)-1
    while end >= start :
        if s[end] != " ":
            break
        end -= 1

    return s[start : end]+s[end]

def transNum(num) :
    "convert a integrate number to english.for example : 12->one-two"
    digits = ["zero","one","two","three","four","five","six","seven","eight","nine"]
    
    l = []
    while num/10 > 0 :
        l.append(num%10)
        num /= 10
    l.append(num%10)

    s = ""
    i = len(l)-1
    while i >= 0 :
        s += digits[l[i]]
        if i > 0 :
            s += "-"
        i -= 1

    return s

def changeUL(s) :
    "change uppercase to lowercase and change lowercase to uppercase in s dtring,for example: input 'Linux code' , output 'lINUX CODE' "
    alphaUpper = string.uppercase
    alphaLower = string.lowercase
    s_ = ""

    for x in range(len(s)):
        p = s[x]
        if p in alphaUpper :
            p = string.lower(p)
        elif p in alphaLower :
            p = string.upper(p)
        s_ += p
    return s_

def findchr(s,ch) :
    "findchr()要在字符串 string 中查找字符 ch,找到就返回该值的索引,否则返回-1"
    for x in range(len(s)) :
        if s[x] == ch :
            break
    else :
        return -1
    return x

def rfindchr(s,ch) :
    for x in range(len(s)-1,-1,-1) :
        if s[x] == ch :
            break
    else :
	return -1
    return x

def Rochambeau() :
    hint = "please input a choice : \n A : paper\n B : rock\n C : sissors\n\n"
    n = 0
    challengerWin = 0
    computerWin = 0
    
    choiceItems = "AaBbCcQq"
    data = {'A':"paper",'B':"rock",'C':"sissors",'a':"paper",'b':"rock",'c':"sissors"}
    
    while True :
        print "%sThe %d tie%s" % ("*"*9,n,"*"*9)
        choice = raw_input(hint)
        while True :
            if choice in choiceItems :
                break
            else :
                print "Error input,please choose again"
                choice = raw_input(hint)

        if choice == "Q" or choice == "q" :
            break

        index = random.randint(0,2)
        error = ord(choice) - ord(choiceItems[index])
        
        print "your choice : %s" % data[choice]
        print "computer choice : %s" % data[choiceItems[index]]

        if error == 0:
            print "Draw"
        elif error == -1 or error == 2:
            challengerWin += 1
            print "challenger win"
        else :
            computerWin += 1
            print "computer win"

        n += 1
        print "challenger win : %d, computer win : %d" %(challengerWin,computerWin)
        print "*"*20
