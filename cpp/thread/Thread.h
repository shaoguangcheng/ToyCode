#ifndef _THREAD_H_
#define _THREAD_H_

/*
 * A simple wrap for standard thread
 */
class _Thread_
{
public:
    template <typename Func>
    _Thread_(Func func);

    template <typename Func, typename Args>
    _Thread_(Func func, Args arg);

    void join();
    bool joinable() const;

private:
    void* _thread_;
};

/*
 * For any derived class, virtual function threadEntryFunc should be overrided
 */
class Thread
{
public:
    Thread();
    virtual ~Thread();

    // return true if starting correctly
    bool startThread(); // Be careful the implementation

    //block util thread exits
    bool waitThreadToExit();

    bool isAnyThreadRunning() const;

protected:
    virtual void threadEntryFunc(){}

    _Thread_* thread_;
};

#include "Thread.hpp"

#endif
