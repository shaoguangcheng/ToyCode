#ifndef FILE_IO_H
#define FILE_IO_H

#include <stdio.h>
#include <unistd.h>
#include <malloc.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>

/////test funtion mkstmp and tmpfile//////////////////
typedef int file_handle;

file_handle writeTempFile(char* buffer, size_t len);
char* readTempFile(file_handle fd,size_t *len);



#endif 
