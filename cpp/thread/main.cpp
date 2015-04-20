#include "Thread.h"

#include <iostream>
#include <unistd.h>

class testThread : public Thread
{
    virtual void threadEntryFunc(){
        std::cout << "In test thread" << std::endl;
    }
};

int main()
{
    Thread* t = new testThread;

    t->startThread();

    sleep(2);

    std::cout << "Test thread joining..." <<std::endl;
    t->waitThreadToExit();
    std::cout << "Test thread finished" << std::endl;

    delete t;

    return 0;
}
