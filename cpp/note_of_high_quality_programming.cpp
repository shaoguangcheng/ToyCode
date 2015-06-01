#include <stdio.h>
#include <malloc.h>
#include <string.h>
#include <stdlib.h>

#pragma warning(once:53)

void testStruct()
{
	struct sys{
//		char *name;
		int  ver;
		char ff;
		int ker[];
	};
	
	struct sys *p;
	printf("%ld\n",sizeof(*p));//c语言中的内存对齐，结构体所占存储空间为其最长成员变量的整数倍

	p = (sys*)malloc(sizeof(sys)+100*sizeof(int));
	p->ker[0] = 1210;

	free(p);
}

void testUnion()
{
	union pp{
		int i;
		char a[2];
	};
	pp p;
	pp *u;
	u = &p;
	u->a[0] = 0x39;
	u->a[1] = 0x38;
	printf("%d\n",u->i);	
}

int checkSys()
{
	union check{
		int i;
		char ch;
	};

	check u;
	u.i = 1;
	return u.ch == 1; //if u.ch = 0,大端模式（高字节在低位），否则小端模式（高字节在低位）
}

int testArray()
{
	int a[5] = {1,2,3,4,5};
	int *p1 = (int*)(&a+1);
	int *p2 = (int*)((int*)a+1);
	printf("%d %d\n",p1[-1],*p2);//5 2
}

int testDefine()
{
#define SQR(x) ((x)*(x))

	printf("%d\n",SQR(4)*4);
	printf("SQR(4)");

#define X 3
#define Y ((X)*2)
#undef X
#define X 2

	printf("%d\n",Y);

#define MAX(x,y) ((x)>(y)?(x):(y))
	printf("%d\n",MAX(1,2));
}

void testTypedef()
{
	typedef int a[10];
	printf("%ld\n",sizeof(a));
	
	typedef int* b[10];	//指针数组
	b p;
	int m = 2;
	p[0] = &m;
	printf("%d\n",*(p[0]));

	typedef int (*c)[10];
	int n[10] = {1};
	c q = &n;
	printf("%d\n",(*q)[7]);
}


void testPointer()
{
	int a[3] = {1,2,3};
	printf("%x %x %x\n",(void*)a,(void*)&a,(void*)&a[0]);
	
	printf("sizeof(a[5]) = %ld\n",(sizeof(a[5])));
}

void testArrayPointer()
{
	char a[5] = {'A','B','C','D','E'};
	char(*p1)[5] = &a;
//	char(*p2)[5] = a;
	printf("%x %x\n",p1+1,p1);	
	printf("%c %x\n",(*p1)[1],*p1+1);

	int b[3] = {1,2,3};
	int* ptr = (int*)(b+1);
	printf("%x\n",*ptr);

	int c = (4,3);
	printf("c = %d\n",c);
}
/////////////////////////////////////
char* getMemory1(char* p, int num)
{
	p = (char*)malloc(num*sizeof(char));
	return p;
}

void getMemory2(char** p,int num)
{
	*p = (char*)malloc(num*sizeof(char));
}

void testMemory()
{
	char* s = NULL;
	s = getMemory1(s,10);
	printf("%ld\n",sizeof(s));

	char* ss = NULL;
	getMemory2(&ss,10);
	printf("%ld\n",sizeof(ss));

	free(s);
	free(ss);
}

////////////////////////////////////////////////
void testStructPointer()
{
	typedef struct student{
		char* name;
		int score;
	}stu,*pstu;

	stu s;
	s.name = (char*)malloc(sizeof(char)*20);
	strcpy(s.name,"LInux");
	printf("%s\n",s.name);
	free(s.name);

	pstu p;
	p = (pstu)malloc(sizeof(stu));
	p->name = (char*)malloc(sizeof(char)*20);
	strcpy(p->name,"linux");
	p->score = 90;
	free(p->name);
	free(p);
}

void testMem()
{
	char s[] = "abcd";
//	char* ss = (char*)malloc(sizeof(s));
	char* ss = (char*)malloc(sizeof(char)*strlen(s)+1*sizeof(char));
	strcpy(ss,s);
	printf("%s\n",ss);
}

typedef struct node_
{
	char ch;
	node_* pNext;
}node,*pNode;

void createList(pNode pHead)
{
	char ch = 's';

	pNode n_ = pHead;
	while(ch != 'p'){
		pNode n = (pNode)malloc(sizeof(node));
		if(n == NULL)
			return;
		scanf("%c",&ch);
		getchar();
		n->ch = ch;
		n->pNext = NULL;
		n_->pNext= n;
		n_ = n_->pNext;
		printf("---%c\n",n->ch);
	}

	n_ = pHead->pNext;
	while(n_ != NULL){
		printf("%c ",n_->ch);
		n_ = n_->pNext;
	}
	printf("\n");
}

void delList(pNode pHead)
{	
	pNode n_ = pHead->pNext;
	while(n_ != NULL){
		pNode temp = n_->pNext;
		free(n_);
		n_ = temp;
	}
	free(pHead);
}

void testSimList()
{
	pNode pHead = (pNode)malloc(sizeof(node));
	pHead->pNext = NULL;
	createList(pHead);
	delList(pHead);
}

#include <unistd.h>
#include <sys/types.h>
#include <signal.h>
void testFork()
{
	int pid;
	if((pid = fork()) < 0){
		perror("fork error");
		return;
	}
	else if(pid == 0){
		execl("/home/cheng/workShop/pythonEX/idcheck.py","ss");
//		FILE* fp = popen("/home/cheng/workShop/pythonEX/test.py","r");
//		char buffer[1024];
//		fgets(buffer,1024,fp);
//		fprintf(stdout,"%s\n",buffer);
//		pclose(fp);
//		kill(getpid(),0);
		return;
	}
//	printf("hello");
	return;
}

template <class T>
void insertSort(T* ptr,int n)
{
	for(int i=1;i<n;i++){
		if(ptr[i] < ptr[i-1]){
			int temp = ptr[i];
			int j;
			for(j=i-1;i>=0;j--)
				if(temp < ptr[j])
					ptr[j+1] = ptr[j];
				else 
					break;
			ptr[j+1] = temp;
		}
	}
}

template <class T>
void shellSort(T* ptr,int n)
{
	for(int step=n/2;step>=1;step/=2)
		for(int i=step;i<n;i+=step)
			if(ptr[i] < ptr[i-step]){
				int temp = ptr[i];
				int j;
				for(j=i-step;j>=0;j-=step)
					if(ptr[j] > temp)
						ptr[j+step] = ptr[j];
					else
						break;

				ptr[j+step] = temp;
			}
}

template <class T>
void quickSort(T* ptr,int left,int right)
{
	if(left < right){
		int base = ptr[left];
		int i = left,j = right;
		while(i<j){
			while(i<j){
				if(ptr[j] < base){
					ptr[i] = ptr[j];
					break;
				}
				else
					j--;
			}

			while(i<j){
				if(ptr[i] > base){
					ptr[j] = ptr[i];
					break;
				}
				else
					i++;
			}
		}

		ptr[i] = base;
		quickSort(ptr,left,i-1);
		quickSort(ptr,i+1,right);
	}
}

//efficient implement 
long int pow_(int x,int n)
{
	long int r = 1;
	while(n){
		if(n&1 != 0)
			r *= x;
		x *= x;
		n = n>>1;
	}
	return r;
}
//////////////////////////////////////
// test how to use auto_ptr
#include <iostream>
#include <memory>
template <class T>
std::ostream& operator << (std::ostream &out,const std::auto_ptr<T> &p)
{
	if(p.get() == NULL)
		out << "NULL";
	else
		out << *p;

	return out;
}

std::auto_ptr<int> add(const std::auto_ptr<int> &p)
{
	std::auto_ptr<int> p_(new int(2));
	*p_ += *p;
	return p_;
}

class p{
private :
	int val;
public :
	p(int a) : val(a){}
	int operator ()(int elem){
		return elem + val;
	}
};

void testauto_ptr()
{
	std::auto_ptr<int> p1(new int(32));
	std::cout << p1 << std::endl;

	std::auto_ptr<int> p2;
	p2 = p1;
	std::cout << p1 << std::endl;
	std::cout << p2 << std::endl; 

	const std::auto_ptr<int> p3(p2);
	std::cout << p2 << std::endl;
	std::cout << p3 << std::endl;

	std::cout << add(p3) << std::endl;
}
//////////////////////////////////////////

////////////////////////////////////////
// test stdout and stderr
void testOutStream()
{
	int i = 0; 
	while(i++ < 10){
		fprintf(stdout,"*");
		fflush(stdout);
		sleep(1);
	}
	
	fprintf(stdout,"\n");

	i = 0;
	while(i++ < 10){
		fprintf(stderr,"*");
		sleep(1);
	}
}

class person
{
public :
	person(){}
	person(char* name_, int age_) :age(age_){
		name = new char [20];
		strcpy(name,name_);
	}
	virtual ~person(){printf("here person\n");delete [] name;}
	char* getName(){return name;}

private :
	char *name;
	int age;
};

class man : public person
{
public :
	man(int p_ = 0,char* name_ = "me", int age_ = 0) : p(p_), person(name_,age_){}
	~man(){printf("here man\n");}
private :
	int p;
};

void testClass2()
{
//	person p("cheng",20);
	person * p = new man();
//	man m(1);
	printf("%s\n",p->getName());
	delete p;
}

int main()
{
	const char* cmd = "clear";
	system(cmd);
//	testStruct();
//	testUnion();
//	printf("%d\n",checkSys());
//	testArray();
//	testDefine();
//	testTypedef();
//#pragma message("I am here\n")
//	testPointer();
//	testArrayPointer();
//	testMemory();
//	testStructPointer();
//	testMem();
//	testSimList();
//	testOutStream();
//	testClass2();
	testFork();
}
