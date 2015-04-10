#ifndef CQUEUE_H
#define CQUEUE_H

#include <stdio.h>
#include <stdlib.h>

// circular queue
template <typename T>
class cQueue{
private :
	T *array;
	int nElem;
	int realElem;
	int front;
	int rear;

public :
	cQueue(int n) : array(new T[n]), nElem(n), realElem(0), front(0), rear(0) {}
	~cQueue(){ if (array != NULL){ delete[] array; } }

	void push(const T &x);
	T pop();

	int size() const;
};

template <typename T>
void cQueue<T>::push(const T &x)
{
	if (realElem == nElem){
		printf("queue is full\n");
		return;
	}

	array[rear] = x;
	++rear;
	if (rear == nElem)
		rear = 0;
	++realElem;
}

template <typename T>
T cQueue<T>::pop()
{
	if (realElem == 0){
		printf("Queue is empty\n");
		exit(-1);
	}

	T tmp = array[front];
	++front;
	if (front == nElem)
		front = 0;
	--realElem;

	return tmp;
}

template <typename T>
int cQueue<T>::size() const
{
	return realElem;
}
#endif
