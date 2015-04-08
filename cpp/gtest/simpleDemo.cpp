#include "gtest/gtest.h"

int gcd(int a, int b)
{
	if(0 == a)
		return b;
	if(0 == b)
		return a;

	return gcd(b, a%b);
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

