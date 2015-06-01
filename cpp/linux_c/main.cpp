#include "file_IO.h"
#include <sys/wait.h>

//////////////////////////
//////test mkstmp/////////////
void test1()
{
	char buffer[] = "linux code";
	char* buffer_;
	size_t len = sizeof(buffer)/sizeof(buffer[0]);

	file_handle fd = writeTempFile(buffer,len);
	buffer_ = readTempFile(fd,&len);
	fprintf(stdout, "%s\n",buffer_);

	free(buffer_);
}
//////////////////////////

///////////test fork //////////////
void testFork()
{
    pid_t pid;
    char* message;
    int n;
    int exit_code;

    printf("fork programming start...\n");
    pid = fork();
    switch(pid){
    case -1 :
        fprintf(stderr,"%s\n",strerror(errno));
        exit(-1);
    case 0 :
        message = "this is child process\n";
        n = 3;
        exit_code = 37;
        break;
    default :
        message = "this is parent process\n";
        n = 5;
        exit_code = 0;
    }

    while((n--)>0){
        printf("%s\n",message);
        sleep(1);
    }

    if(pid != 0){
        int statusVal;
        pid_t childPID;

        childPID = wait(&statusVal);
        if(WIFEXITED(statusVal) != 0)
            printf("child process exit wpth code %d\n",WEXITSTATUS(statusVal));
        else
            printf("child process exit abnormally\n");
    }

    exit(exit_code);
}

/////////////////test thread///////////////
#include <pthread.h>
#include <semaphore.h>

int flag = 0;

pthread_mutex_t mutex;
sem_t sem;

void* threadFunc(void* arg)
{
    printf("thread function's argumrnt is %s\n",(char*)arg);
    sleep(2);
    printf("thread finished\n");

    sem_post(&sem);
    for(int i=0;i<3;i++){
        pthread_mutex_lock(&mutex);
        flag++;
        printf("this is new thread,flag = %d\n",flag);
        pthread_mutex_unlock(&mutex);
        sleep(1);
    }

    void* result = (void*)("Thank you");
    pthread_exit(result);
}

void testThread()
{
    pthread_t thread;
    pthread_attr_t threadAttr;

    int res;
    char message[] = "hello world\n";
    void* result;

    pthread_mutex_init(&mutex,NULL);

    res = pthread_attr_init(&threadAttr);
    if(res != 0){
        perror("init thread attribute error!\n");
        exit(-1);
    }

    res = pthread_attr_setdetachstate(&threadAttr,PTHREAD_CREATE_DETACHED);
    if(res != 0){
        perror("setting thread attribute failed!\n");
        exit(-1);
    }

    res = sem_init(&sem,0,0);
    if(res != 0){
        perror("init sem_t failed\n");
        exit(-1);
    }

    res = pthread_create(&thread,NULL,threadFunc,message);
    if(res != 0){
        perror("create new thread failed\n");
        exit(-1);
    }

    pthread_attr_destroy(&threadAttr);

    sem_wait(&sem);
    for(int i=0;i<3;i++){
        pthread_mutex_lock(&mutex);
        flag++;
        printf("this is main thread,flag = %d\n",flag);
        pthread_mutex_unlock(&mutex);
        sleep(1);
    }

    pthread_join(thread,&result);

    printf("thread result : %s\n",(char*)result);
    printf("main thread finished\n");
}


int main()
{
	test1();
    testFork();
    testThread();


	return 0;
}
