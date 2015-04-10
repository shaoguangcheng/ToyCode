#include "gtest/gtest.h"

#include "cQueue.h"

#include <time.h>

/////////////////////////////////////////////////////////
// show how to test a class
/////////////////////////////////////////////////////////
class quickTest : public testing::Test
{
protected:
	virtual void SetUp(){
		start = time(NULL);
	}

	virtual void TearDown(){
		const time_t end = time(NULL);
		EXPECT_TRUE(end - start <= 1) << "The test took too long";
	}

	time_t start;
};

class cQueueTest : public quickTest
{
protected:
	cQueue<int>* cq;

	virtual void SetUp(){
		quickTest::SetUp();
		cq = new cQueue<int>(3);
	}

	virtual void TearDown(){
		quickTest::TearDown();
	}
};


TEST_F(cQueueTest, Constructor)
{
	EXPECT_EQ(0, cq->size());
}

TEST_F(cQueueTest, Push)
{
	cq->push(0);
	EXPECT_EQ(1, cq->size());
	cq->push(1);
	EXPECT_EQ(2, cq->size());
	cq->push(2);
	EXPECT_EQ(3, cq->size());
	cq->push(3);
	EXPECT_EQ(3, cq->size());			
}

TEST_F(cQueueTest, Pop)
{
	cq->pop();
	EXPECT_EQ(2, cq->size());
	cq->pop();
	EXPECT_EQ(1, cq->size());
	cq->pop();
	EXPECT_EQ(0, cq->size());		
}