#include "util.h"

#include <cstdlib>
#include <cstring>
#include <unistd.h>

using namespace std;

int T = 1;
bool V, S, W;

char param_fname[] = "./assets/model_file.bin";
char input_fname[] = "./assets/input1N.bin";
char answer_fname[] = "./assets/answer.bin";
char output_fname[] = "./output.bin";

void parse_args(int argc, char **argv) {
    int args;
    while ((args = getopt(argc, argv, "i:o:a:p:t:vswh")) != -1) {
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

    fprintf(stderr, "[LOG] Running...\n");
    
    fprintf(stdout, "\n=============================================\n");
    fprintf(stdout, " Model: GPT-2 (12 layers)\n");
    fprintf(stdout, "---------------------------------------------\n");
    fprintf(stdout, " Warmup: %s\n", W ? "ON" : "OFF");
    fprintf(stdout, " Validation: %s\n", V ? "ON" : "OFF");
    fprintf(stdout, " Save output: %s\n", S ? "ON" : "OFF");
    fprintf(stdout, "=============================================\n\n");
}

void print_help() {
    fprintf(stdout,
        " Usage: ./main [-i 'pth'] [-p 'pth'] [-o 'pth'] [-a 'pth'] [-t 'tokens'] [-v] [-s] [-w] [-h]\n");
    fprintf(stdout, " Options:\n");
    fprintf(stdout, "  -i: Input binary path (default: assets/input1N.bin)\n");
    fprintf(stdout, "  -p: Model parameter path (default: assets/model_file.bin)\n");
    fprintf(stdout, "  -o: Output binary path (default: output.bin)\n");
    fprintf(stdout, "  -a: Answer binary path (default: assets/answer.bin)\n");
    fprintf(stdout, "  -t: Number of tokens to generate (default: 1)\n");
    fprintf(stdout, "  -v: Enable validation (default: OFF)\n");
    fprintf(stdout, "  -s: Enable saving output tensor (default: OFF)\n");
    fprintf(stdout, "  -w: Enable warmup (default: OFF)\n");
    fprintf(stdout, "  -h: Print manual and options (default: OFF)\n");
}

void* read_binary(const char *fname, size_t *size) {
    fprintf(stderr, "[LOG] Reading binary... \n");
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
        *size = (size_t)(size_ / 4); // float

    return buf;
}

double get_time() {
  struct timespec tv;
  clock_gettime(CLOCK_MONOTONIC, &tv);

  return tv.tv_sec + tv.tv_nsec * 1e-9;
}