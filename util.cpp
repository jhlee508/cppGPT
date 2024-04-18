#include "util.h"

#include <cstdlib>
#include <cstring>
#include <unistd.h>

using namespace std;

int N = 1;
int T = 5;
bool V, S, W;

char input_fname[] = "./data/input.bin";
char param_fname[] = "./assets/model_file.bin";
char answer_fname[] = "./data/answer.bin";
char output_fname[] = "./data/output.bin";

void parse_args(int argc, char **argv) {
    int args;
    while ((args = getopt(argc, argv, "i:o:a:p:n:t:vswh")) != -1) {
        switch (args) {
        case 'i':
            strcpy(input_fname, optarg);
            break;
        case 'o':
            strcpy(output_fname, optarg);
            break;
        case 'a':
            strcpy(answer_fname, optarg);
            break;
        case 'p':
            strcpy(param_fname, optarg);
            break;
        case 'n':
            N = atoi(optarg);
            break;
        case 't':
            T = atoi(optarg);
            break;
        case 'v':
            V = true;
            break;
        case 's':
            S = true;
            break;
        case 'w':
            W = true;
            break;
        case 'h':
            print_help();
            exit(0);
            break;
        default:
            print_help();
            exit(0);
            break;
        }
    }

    fprintf(stdout, "\n=============================================\n");
    fprintf(stdout, " Model: GPT-2 (12 layers)\n");
    fprintf(stdout, "---------------------------------------------\n");
    fprintf(stdout, " Warmup: %s\n", W ? "ON" : "OFF");
    fprintf(stdout, " Validation: %s\n", V ? "ON" : "OFF");
    fprintf(stdout, " Save output: %s\n", S ? "ON" : "OFF");
    fprintf(stdout, " Number of Prompts: %d\n", N);
    fprintf(stdout, " Number of Tokens to generate: %d\n", T);
    fprintf(stdout, "=============================================\n\n");
}

void print_help() {
    fprintf(stdout,
        " Usage: ./main [-i 'pth'] [-p 'pth'] [-o 'pth'] [-a 'pth'] [-t 'tokens'] [-n 'prompts'] [-v] [-s] [-w] [-h]\n");
    fprintf(stdout, " Options:\n");
    fprintf(stdout, "  -i: Input binary path (default: data/input.bin)\n");
    fprintf(stdout, "  -p: Model parameter path (default: assets/model_file.bin)\n");
    fprintf(stdout, "  -o: Output binary path (default: output.bin)\n");
    fprintf(stdout, "  -a: Answer binary path (default: data/answer.bin)\n");
    fprintf(stdout, "  -n: Number of prompts (default: 1)\n");
    fprintf(stdout, "  -t: Number of tokens to generate (default: 5)\n");
    fprintf(stdout, "  -v: Enable validation (default: OFF)\n");
    fprintf(stdout, "  -s: Enable saving output tensor (default: OFF)\n");
    fprintf(stdout, "  -w: Enable warmup (default: OFF)\n");
    fprintf(stdout, "  -h: Print manual and options (default: OFF)\n");
}

void* read_binary(const char *fname, size_t *size) {
    FILE *f = fopen(fname, "rb");
    if (f == NULL) {
        fprintf(stderr, "[ERROR] Cannot open file \'%s\'\n", fname);
        exit(-1);
    }

    fseek(f, 0, SEEK_END);
    size_t size_ = ftell(f);
    rewind(f);

    void *buf = malloc(size_);
    size_t ret = fread(buf, 1, size_, f);
    if (ret == 0) {
        fprintf(stderr, "[ERROR] Cannot read file \'%s\'\n", fname);
        exit(-1);
    }
    fclose(f);

    if (size != NULL)
        *size = (size_t)(size_ / 4); // 4 bytes per float or int

    return buf;
}

void write_binary(int *output, const char *filename, int size_) {
    FILE *f = (FILE *)fopen(filename, "w");
    fwrite(output, sizeof(int), size_, f);
    fclose(f);
}

double get_time() {
  struct timespec tv;
  clock_gettime(CLOCK_MONOTONIC, &tv);

  return tv.tv_sec + tv.tv_nsec * 1e-9;
}

int check_validation(int* output, int* answer, int size_) {

    int diff = -1;
    for (int i = 0; i < size_; i++) {
        if (output[i] != answer[i]) {
            diff = i;
            break;
        }
    }

    return diff;
}