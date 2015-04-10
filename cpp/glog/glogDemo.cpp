#include "glog_config.h"


void fun()
{
	int *x;
	*x = 1;
	delete x; // special intention here
	*x += 1;
}

int main(int argc, char* argv[])
{
	GlogHelper glog_(argv[0]);

	LOG(INFO) << "hello, glog!";

	int n = 8;
	double x = 9.0;
	std::string name("Lisa");
	LOG_IF(ERROR, n > 5) << "n is larger than 5";

	CHECK(n == 8) << ", n is not equal to 10";
	CHECK_EQ(n, 8) << ", n is not equal to 8";
	CHECK_GT(n, 5);
	CHECK_LT(n, 10);
	CHECK_NOTNULL(&n);
	CHECK_STREQ("Lisa", name.c_str());
	CHECK_DOUBLE_EQ(9.0, x);

	for(int i = 0; i < 100; ++i){
		LOG_IF(INFO,i==100) << "LOG_IF(INFO,i==100)  google::COUNTER=" << google::COUNTER << "  i=" << i; 
		LOG_EVERY_N(INFO,10) << "LOG_EVERY_N(INFO,10)  google::COUNTER=" << google::COUNTER <<"  i=" << i;
		LOG_IF_EVERY_N(WARNING,(i>50),10) << "LOG_IF_EVERY_N(INFO,(i>50),10)  google::COUNTER=" << google::COUNTER<<"  i=" << i;
		LOG_FIRST_N(ERROR,5) << "LOG_FIRST_N(INFO,5)  google::COUNTER=" << google::COUNTER << "  i=" << i;
	}

	fun();

	return 0;
}