#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cassert>
#include <vector>

#include "util.h"
#include "model.h"


int main(int argc, char **argv) {
    // float *input
    // size_t input_size;
    double st = 0.0, et = 0.0;
    parse_args(argc, argv);

    ////////////////////////////////////////////////////////////////////
    // INITIALIZATION                                                 //
    ////////////////////////////////////////////////////////////////////
    fprintf(stderr, "[LOG] Reading input... \n");

    // TODO: Read input
    // float* input = (float *)read_binary(input_fname, &input_size);

    // vector type example_input
    vector<int> example_input = {
        // Alan Turing theorized that computers would one day become
        36235, 39141, 18765, 1143, 326, 9061, 561, 530, 1110, 1716 
    };

    // int example_input[] = {
    //     // Alan Turing theorized that computers would one day become
    //     36235, 39141, 18765, 1143, 326, 9061, 561, 530, 1110, 1716 
    // }; 
    // float example_answer[] = {
    //     // the
    //     262 
    // };
    
    fprintf(stderr, "[LOG] Initializing... \n");
    initialize_parameters(param_fname);
    initialize_activations();

    if (W) {
        fprintf(stdout, " Warming up... \n");

        // TODO: Warm-up
    }

    ////////////////////////////////////////////////////////////////////
    // MODEL COMPUTATION                                              //
    ////////////////////////////////////////////////////////////////////
    st = get_time();

    N_SEQ = example_input.size();

    /* Cannot surpass the max_seq_len of the model */
    assert(N_SEQ + T <= N_CTX);

    /* Text Generation */
    fprintf(stdout, " Generating tokens... \n");
    generate_tokens(example_input, T);

    et = get_time();
    fprintf(stdout, " Done!\n");
    fprintf(stdout, " Elapsed time: %lf (sec)\n", et - st);
    fprintf(stdout, " Throughput: %lf (tokens/sec)\n", T / (et - st));
    
    ////////////////////////////////////////////////////////////////////
    // FINALIZATION                                                   //
    ////////////////////////////////////////////////////////////////////    
    fprintf(stderr, "[LOG] Finalizing... \n");
    finalize_parameters();
    finalize_activations();

    if (S) {
        fprintf(stdout, " Saving output... \n");

        // TODO: Save output
    }

    if (V) {
        fprintf(stdout, " Validation... \n");

        // TODO: Validation
    }

    return 0;
}