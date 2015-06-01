#include "file_IO.h"

//////////////////part 1/////////////////////////
file_handle writeTempFile(char* buffer, size_t len)
{
	char tempFileName[] = "./tmpfile.XXXXXX";
	file_handle fd = mkstemp(tempFileName);
	if(fd == -1){
		fprintf(stderr, "%s\n", strerror(errno));
		exit(1);
	}

	unlink(tempFileName);
	size_t size = write(fd,&len,sizeof(size_t));
	if(size != sizeof(size_t)){
		fprintf(stderr, "%s\n", strerror(errno));
		exit(1);
	}

	size = write(fd,buffer,len);
		if(size != len){
		fprintf(stderr, "%s\n", strerror(errno));
		exit(1);
	}

	return fd;
}

char* readTempFile(file_handle fd,size_t *len)
{
	char* buffer;
	lseek(fd,0,SEEK_SET);

	int size = read(fd,len,sizeof(size_t));
	if(size != sizeof(size_t)){
		fprintf(stderr, "%s\n", strerror(errno));
		exit(1);
	}

	buffer = (char*)malloc((*len)*sizeof(char));
	if(buffer == NULL){
		fprintf(stderr, "%s\n",strerror(errno) );
		exit(1);
	}

	size = read(fd,buffer,*len);
	if(size != *len){
		fprintf(stderr, "%s\n", strerror(errno));
		exit(1);
	}

	close(fd);
	return buffer;
}
/////////////////////////////////////////////////////////////