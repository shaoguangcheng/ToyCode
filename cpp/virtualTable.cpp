#include <iostream>

using namespace std;

class base
{
public:
  virtual void f()
  {
    cout << "base::f()" << endl;
  }

  virtual void g()
  {
    cout << "base::g()" << endl;
  }

  virtual void h()
  {
    cout << "base::h()" << endl;
  }

  void k()
  {
    cout << "base::k()" << endl;
  }
};

class derive : public base 
{
public :
  void f()
  {
    cout << "derive::f()" << endl;
  }
};

int main()
{
  typedef void (*pFun)(void);
  typedef long long int64; 

  pFun fun = NULL;
  
  cout << "Base class instance :" << endl;
  base b;
  int64* pVTable = (int64*)*(int64*)(&b); // the address of virtual table

  fun = (pFun)pVTable[0];
  fun();

  fun = (pFun)pVTable[1];
  fun();
  
  fun = (pFun)pVTable[2];
  fun();

  (&b)->k(); // non-static member function (&b = this pointer)
  
  cout << "Derived class instance :" << endl;
  derive d;	
  pVTable = (int64*)*(int64*)(&d); // the address of virtual table
  
  fun = (pFun)pVTable[0];
  fun();

  fun = (pFun)pVTable[1];
  fun();
  
  fun = (pFun)pVTable[2];
  fun();

  (&d)->k();

  return 0;
}
