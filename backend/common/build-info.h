#pragma once

#include <cstdio>

int fDgsBuildNumber(void);

const char * fDgsCommit(void);
const char * fDgsCompiler(void);

const char * fDgsBuildTarget(void);
const char * fDgsBuildInfo(void);

void fDgsPrintBuildInfo(const char *, FILE * = stderr);
