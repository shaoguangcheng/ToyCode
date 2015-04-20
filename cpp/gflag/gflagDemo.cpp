#include "flag_declare.h"


int main(int argc, char* argv[])
{
	parseCommandLine(argc, argv);

	google::ShutDownCommandLineFlags();

	return 0;
}
