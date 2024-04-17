#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cassert>
#include <vector>

#include "util.h"
#include "model.h"


int main(int argc, char **argv) {
    int *input, *output;
    size_t input_size;
    double st = 0.0, et = 0.0;
    parse_args(argc, argv);

    ////////////////////////////////////////////////////////////////////
    // INITIALIZATION                                                 //
    ////////////////////////////////////////////////////////////////////
    /* Initialize input and output */
    fprintf(stderr, "[LOG] Reading input from %s\n", input_fname);
    input = (int *)read_binary(input_fname, &input_size);
    output = (int *)malloc(N * T * sizeof(int));
    
    /* Initialize parameters and activations */
    fprintf(stderr, "[LOG] Initializing... \n");
    initialize_parameters(param_fname);
    initialize_activations();

    /* Cannot surpass the max_seq_len of the model */
    assert(N_SEQ + T <= N_CTX);

    /* Warm-up */
    if (W) {
        fprintf(stdout, " Warming up... \n");
        for (int i = 0; i < 3; i++)
            generate_tokens(input, output, 1, 1);
    }

    ////////////////////////////////////////////////////////////////////
    // MODEL COMPUTATION                                              //
    ////////////////////////////////////////////////////////////////////
    st = get_time();

    /* Text Generation */
    fprintf(stdout, " Generating tokens... \n");
    generate_tokens(input, output, N, T);

    et = get_time();

    /* Print the result */
    fprintf(stdout, " Done!\n");
    fprintf(stdout, " Elapsed time: %lf (sec)\n", et - st);
    fprintf(stdout, " Throughput: %lf (tokens/sec)\n", N*T / (et - st));
    
    ////////////////////////////////////////////////////////////////////
    // FINALIZATION                                                   //
    ////////////////////////////////////////////////////////////////////    
    /* Finalize parameters and activations */
    fprintf(stderr, "[LOG] Finalizing... \n");
    finalize_parameters();
    finalize_activations();

    /* Save output */
    if (S) {
        fprintf(stdout, " Saving output... \n");
        write_binary(output, output_fname, N*T);
    }

    /* Validation */
    if (V) {
        fprintf(stdout, " Validation... \n");

        // TODO: Validation
    }

    return 0;
}