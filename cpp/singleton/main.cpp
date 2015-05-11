#include "singleton.h"


int main()
{
  singleton::get();
  cout << singleton::getCount() << endl;
  singleton::getCount()++;

  singleton::get();
  cout << singleton::getCount() << endl;
 	
	singleton* s = singleton::get();
	singleton*ss = s->get();

  return 0;
}
