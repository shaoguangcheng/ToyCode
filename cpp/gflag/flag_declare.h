#ifndef FLAG_DECLARE_H
#define FLAG_DECLARE_H

#include <gflags/gflags.h>

#include <string>

#include <sys/stat.h>
#include <stdio.h>

using std::string;
using google::int32;

DECLARE_int32(age);
DECLARE_string(name);
DECLARE_bool(isBoy);
DECLARE_double(weight);
DECLARE_string(filename); // file that describe the information of child detailly

DEFINE_int32(age, 12, "Use integer to denote the age of child");
DEFINE_string(name, "Li Ming, Zhang hua", "Children's name");
DEFINE_bool(isBoy, true, "Whether the child is a boy");
DEFINE_double(weight, 40.0, "Use double to denote the weight of the child");
DEFINE_string(filename, "gflag.pro", "");

static bool ValidateAge(const char* flagname, int32 value)
{
	if(value > 6 && value < 15)
		return true;
	printf("Invalid value for --%s: %d\n", flagname, int(value));
	return false;
}

static bool ValidateName(const char* flagname, const string& name)
{
	return true;
}

static bool ValidateIsBoy(const char* flagname, bool value)
{
	return true;
}

static bool ValidateWeight(const char* flagname, double value)
{
	if(value > 0 && value <100)
		return true;
	printf("Invalid value for --%s: %d\n", flagname, int(value));
	return false;	
}

static bool ValidateFilename(const char* flagname, const string& value)
{
	if(value.c_str() == NULL)
		return false;

	struct stat statbuf;
	lstat(value.c_str(), &statbuf);

	int ret = S_ISREG(statbuf.st_mode);

	return bool(ret);
}

static bool flagsValidator()
{
	bool ret = true;

	ret &= google::RegisterFlagValidator(&FLAGS_age, ValidateAge);
	ret &= google::RegisterFlagValidator(&FLAGS_name, ValidateName);
	ret &= google::RegisterFlagValidator(&FLAGS_isBoy, ValidateIsBoy);
	ret &= google::RegisterFlagValidator(&FLAGS_weight, ValidateWeight);
	ret &= google::RegisterFlagValidator(&FLAGS_filename, ValidateFilename);

	return ret;
}

void parseCommandLine(int argc, char* argv[])
{
    flagsValidator();

    string usage("This program shows a demo that how to use gflags.\n");
    google::SetUsageMessage(usage);

    google::ParseCommandLineFlags(&argc, &argv, true);

    if(argc == 1)
        google::ShowUsageWithFlagsRestrict(argv[0], "gflagDemo");
}


#endif
