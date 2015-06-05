#include <iostream>
#include <cstdlib>
#include <ctime>
#include <vector>

using namespace std;

const int nState  = 6;
const int nAction = 6;
const float R[nState][nAction] = {
			 {-1, -1, -1, -1, 0, -1},
			 {-1, -1, -1, 0, -1, 100},
			 {-1, -1, -1, 0, -1, -1},
			 {-1, 0, 0, -1, 0, -1},
			 {0, -1, -1, 0, -1, 100},
			 {-1, 0, -1, -1, 0, 100}};

//generate a random integer in [begin, end)
int rangeRand(int begin, int end)
{
	return (begin == end) ? begin : rand()%(end-begin) + begin;
}

float maxVec(vector<float>& v)
{
	int n = v.size();
	float ret = v[0];
	for(int i = 1; i < n; ++i)
		if(ret < v[i])
			ret = v[i];

	return ret;
}

int maxVecIndex(vector<float>& v)
{
	int n = v.size();
	int ret = 0;
	for(int i = 1; i < n; ++i)
		if(v[ret] < v[i])
			ret = i;

	return ret;
}

template <typename Dtype>
void printMat(const vector<vector<Dtype> >& M)
{
	int m = M.size();
	for(int i = 0; i < m; ++i){
		int n = M[i].size();
		for(int j = 0; j < n; ++j)
			cout << M[i][j] << " ";
		cout << endl;
	}	
}

void init(vector<vector<float> >& Q, vector<vector<int> >& N)
{
	Q.resize(nState, vector<float>(nAction, 0.0));	

	N.resize(nState);
	for(int i = 0; i < nState; ++i)
		for(int j = 0; j < nAction; ++j)
			if(R[i][j] >= 0)
				N[i].push_back(j);

	srand((unsigned)time(NULL));
}

void learningQ(vector<vector<float> >& Q, int nEpisode, float gamma, int target)
{
	vector<vector<int> > N;

	init(Q, N);
	
	int episode = 0;
	while(episode < nEpisode){
		++episode;
		int state = rangeRand(0, nState);

		do{
			int n = N[state].size();
			int nextState = N[state][rangeRand(0, n)]; 
			int action = nextState; 
			Q[state][action] = R[state][action] + gamma*maxVec(Q[nextState]); 
			state = nextState;		
		}while(state != target);
	}
}

void run(vector<vector<float> >& Q, vector<vector<int> >& path, int target)
{
	path.resize(nState);
	
	for(int begin = 0; begin < nState; ++begin){
		int i = begin;
		while(i != target){
			path[begin].push_back(i);
			i = maxVecIndex(Q[i]);
		}
		path[begin].push_back(target);
	}
}

int main(int argc, char* argv[])
{
	vector<vector<float> > Q;
	vector<vector<int> > path;

	learningQ(Q, 1000, 0.8, 5);
	run(Q, path, 5);

	printMat(path);

	return 0;
}
