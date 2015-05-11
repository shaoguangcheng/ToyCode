#ifndef SINGLETON_H
#define SINGLETON_H

#include <iostream>
#include <memory>

using std::shared_ptr;
using std::cout;
using std::endl;

class singleton
{
 private:
  int count;
  singleton() {
		cout << "construct singleton" << endl;
		count = 0;
  }

	singleton(const singleton& );
	singleton& operator=(const singleton&);
  
 public:
  static shared_ptr<singleton> singleton_;

  static singleton* get(){
	if(!singleton_.get()){
	  singleton_.reset(new singleton());
	}
	return singleton_.get();
  }
  
  static int&  getCount(){
	return get()->count;
  }
};

shared_ptr<singleton> singleton::singleton_;

#endif
