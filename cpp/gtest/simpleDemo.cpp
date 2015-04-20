#include "gtest/gtest.h"

#include <cmath>

int gcd(int a, int b)
{
	if(0 == a)
		return b;
	if(0 == b)
		return a;

	return gcd(b, a%b);
}

bool isPrime(int x)
{
	if(x < 2)
		return false;

	int end = std::sqrt(x);
	for(int i = 2; i <= end; ++i)
		if(x%i == 0)
			return false;

	return true;
}

TEST(gcdTest, HandleZeroInput)
{
	ASSERT_EQ(3, gcd(3, 0));
	ASSERT_EQ(3, gcd(3, 0));
}

TEST(gcdTest, HandleNoneZeroInput)
{
	EXPECT_EQ(2, gcd(4, 10));
	EXPECT_EQ(6, gcd(30, 18));
}

TEST(isPrimeTest, NonLegalInput)
{
	EXPECT_FALSE(isPrime(-1));
	EXPECT_FALSE(isPrime(0));
	EXPECT_FALSE(isPrime(1));	
}

TEST(isPrimeTest, LegalInput)
{
	EXPECT_TRUE(isPrime(2));
	EXPECT_TRUE(isPrime(109));
	EXPECT_FALSE(isPrime(10000));
}

///////////////////////////////////////////////////////////////////

// self-defined global event
class testEnvironment : public testing::Environment
{
public :
	virtual void SetUp(){
		std::cout << "Do something before all test cases to execute" << std::endl;
	}

	virtual void TearDown(){
		std::cout << "Do something after all test cases executed" << std::endl;
	}
};
///////////////////////////////////////////////////////////////////

// Value Parameterized Test
class isPrimeParamTest : public testing::TestWithParam<int> // TestWithParam is derived from Test
{
protected:
	int n;
	isPrimeParamTest(){
		n = GetParam();
	}
};

TEST_P(isPrimeParamTest, handleTrueReturn)
{
	EXPECT_TRUE(isPrime(n));
}

INSTANTIATE_TEST_CASE_P(TrueReturn,
						isPrimeParamTest,
						testing::Values(3, 5, 11, 23, 17));

/*
TEST_P(isPrimeParamTest, handleTrueReturn)
{
	int n = GetParam();
	EXPECT_TRUE(isPrime(n));
}

INSTANTIATE_TEST_CASE_P(TrueReturn,
						isPrimeParamTest,
						testing::Values(3, 5, 11, 23, 17));
*/

/*
int main(int argc, char* argv[])
{
	testing::AddGlobalTestEnvironment(new testEnvironment);
	testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
*/