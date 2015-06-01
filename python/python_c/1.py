import ctypes as ct

def test() :
    lib = ct.cdll.LoadLibrary("/home/cheng/libsome.so")
    lib. hello()

if __name__ == "__main__" :
    test()
