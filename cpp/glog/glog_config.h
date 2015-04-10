#ifndef GLOG_CONFIG_H
#define GLOG_CONFIG_H

#include <glog/logging.h>

#include <stdlib.h>

void signalHandle(const char* data, int size);

void signalHandle(const char* data, int size)
{
	LOG(ERROR) /*<< __func__ << ", " 
			   << __FILE__ << ": Error ... " */
			   << std::string(data, size);
}

#define LOGDIR "log"
#define MKDIR "mkdir -p "LOGDIR

class GlogHelper
{
public:
	GlogHelper(const char* program){
		system(MKDIR);

		// some common used configuration for glog
		google::InitGoogleLogging(program);
		google::SetStderrLogging(google::INFO);
		FLAGS_colorlogtostderr = true;
		google::SetLogDestination(google::INFO, LOGDIR"/INFO_");
		google::SetLogDestination(google::WARNING,LOGDIR"/WARNING_");
		google::SetLogDestination(google::ERROR,LOGDIR"/ERROR_");
		google::SetLogFilenameExtension(".log");
		google::InstallFailureSignalHandler();
		google::InstallFailureWriter(&signalHandle);
	}

	~GlogHelper(){
		google::ShutdownGoogleLogging();
	}
};

#endif
