#pragma once

#include <cstdlib>
#include <string>

using namespace std;

extern int N;
extern int T;
extern bool V, S, W;

extern char param_fname[100];
extern char input_fname[100];
extern char answer_fname[100];
extern char output_fname[100];

void parse_args(int argc, char **argv);
void print_help();
void *read_binary(const char *fname, size_t *size);
void write_binary(int *output, const char *filename, int size_);
double get_time();

