//#include "Thread.h"

#include <thread>

/**
 * class _Thread_
 */
template <typename Func>
_Thread_::_Thread_(Func func)
    : _thread_(new std::thread(func))
{}

template <typename Func, typename Args>
_Thread_::_Thread_(Func func, Args arg)
    : _thread_(new std::thread(func, arg))
{}

void _Thread_::join()
{
    static_cast<std::thread*>(_thread_)->join();
}

bool _Thread_::joinable() const
{
    return static_cast<std::thread*>(_thread_)->joinable();
}

/**
 * class Thread
 */

Thread::Thread()
    : thread_(NULL)
{}

Thread::~Thread()
{
    waitThreadToExit();
    if(NULL != thread_)
        delete thread_;
}

bool Thread::isAnyThreadRunning() const
{
    return thread_ != NULL && thread_->joinable();
}

bool Thread::waitThreadToExit()
{
    if(isAnyThreadRunning()){
        try{
            thread_->join();
        }
        catch(...){
            return false;
        }
    }

    return true;
}

bool Thread::startThread()
{
    if(!waitThreadToExit())
        return false;
    else{
        try{
            thread_ = new _Thread_([&](){this->threadEntryFunc();}); // be careful here
        }
        catch(...){
            return false;
        }
    }

    return true;
}

