#include <string>
#include <iostream>
#include <fcntl.h>

#include "simple_proto/simple.pb.h"
#include "protoIO.h"

using std::string;
using std::cout;
using std::endl;

int main(int argc, char* argv[])
{
  CHECK_EQ(argc, 2) << "Must have two args";

  string filename(argv[1]);

  simple::array simple_array;

  // read text proto file
  CHECK_EQ(readProtoFromTextFile(filename.c_str(), &simple_array), true) << "Read array config failed";

  cout << "name  : " << simple_array.name() << endl;
  cout << "array size : " << simple_array._array__size() << endl;
  cout << "array[0] width : " << simple_array._array_(0).width() << endl;
  int shapeSize = simple_array._array_(0).shape().dim_size();
  simple::array_* _array_ = simple_array.mutable__array_(0);
  simple::arrayShape* shape = _array_->mutable_shape();

  for(int i = 0; i < shapeSize; ++i){
	cout << "dim :" << shape->dim(i) << endl;
  }

  shape->add_dim(4);
  shape->add_dim(2);

  int data[] = {1,2,3,4,5,5,6,7,8,9};
  for(int i = 0; i < 10; ++i)
	_array_->add_data(data[i]);

  string newFilename = filename + "_";
  writeProtoToTextFile(simple_array, newFilename.c_str());

  newFilename = filename + "binary";
  writeProtoToBinaryFile(simple_array, newFilename.c_str());

  readProtoFromBinaryFile(newFilename.c_str(), &simple_array);

  return 0;
}
