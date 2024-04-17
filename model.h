#pragma once

#include "tensor.h"


/* Input Sequence Length */
static int N_SEQ = 10;

/* Constants */
#define MATH_PI 3.1415926535f

/* Model Hyperparameter Configurations */
#define N_VOCAB 50257
#define N_CTX 1024
#define N_EMBD 768
#define N_HEAD 12
#define N_LAYER 12

/* Model Parameter Offsets */
#define OFFSET1     (3*N_EMBD)          // 2304
#define OFFSET2     N_EMBD*(3*N_EMBD)   // 768*2304
#define OFFSET3     N_EMBD              // 768
#define OFFSET4     N_EMBD*N_EMBD       // 768*768
#define OFFSET5     4*N_EMBD            // 3072
#define OFFSET6     N_EMBD*(4*N_EMBD)   // 768*3072 
#define OFFSET7     N_CTX*N_EMBD        // 1024*768
#define OFFSET8     N_VOCAB*N_EMBD      // 50257*768


void initialize_parameters(const char* param_fname);
void initialize_activations();
void generate_tokens(int* input, int n_token);
void finalize_parameters();
void finalize_activations();