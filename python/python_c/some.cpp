#include <stdio.h>

extern "C"{
extern void printm(const char* s)
	{
		printf("%s\n",s);
	}

extern	bool hello()
	{
		printf("hello\n");
		return true;
	}
}
