#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cassert>
#include <vector>
#include <mpi.h>

#include "util.h"
#include "model.h"


int main(int argc, char **argv) {

    /* MPI Initialization */
    int mpi_rank, mpi_size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    
    /* Parse arguments */
    if (mpi_rank == 0) {
        parse_args(argc, argv);
    }

    ////////////////////////////////////////////////////////////////////
    // INITIALIZATION                                                 //
    ////////////////////////////////////////////////////////////////////
    
    int *input, *output;

    if (mpi_rank == 0) {
        /* Load input (size: N x N_SEQ) from file  */
        fprintf(stderr, "[LOG] Reading input from %s\n", input_fname);
        size_t input_size;
        input = (int *)read_binary(input_fname, &input_size);
        
        if (input_size % N_SEQ != 0) {
            fprintf(stderr, "[ERROR] Invalid input size\n");
            exit(1);
        }
    }

    /* Allocate output (size: N x T) */
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
    
    double st = 0.0, et = 0.0;

    if (mpi_rank == 0) {
        fprintf(stdout, " Start computation... \n");
        st = get_time();
    }

    /* Text Generation */
    MPI_Barrier(MPI_COMM_WORLD);
    generate_tokens(input, output, N, T);
    MPI_Barrier(MPI_COMM_WORLD);

    if (mpi_rank == 0) {
        et = get_time();

        /* Print the result */
        fprintf(stdout, " Done!\n");
        fprintf(stdout, " Elapsed time: %lf (sec)\n", et - st);
        fprintf(stdout, " Throughput: %lf (tokens/sec)\n", 
            N*T / (et - st));
    } 
    
    ////////////////////////////////////////////////////////////////////
    // FINALIZATION                                                   //
    ////////////////////////////////////////////////////////////////////    
    
    /* Finalize parameters and activations */
    fprintf(stderr, "[LOG] Finalizing... \n");
    finalize_parameters();
    finalize_activations();
    
    if (mpi_rank == 0) {
        /* Save output */
        if (S) {
            fprintf(stdout, " Saving output... \n");
            write_binary(output, output_fname, N*T);
        }

        /* Validation */
        if (V) {
            fprintf(stdout, " Validation... \n");

            int *answer = (int *)read_binary(answer_fname, NULL);
            int ret = check_validation(output, answer, N*T);
            if (ret == -1) {
                fprintf(stdout, " Validation passed!\n");
            } else {
                fprintf(stdout, " Validation failed: First mismatch "
                    "at prompt[#%d], token_ID[#%d] (output[%d]=%d <-> "
                    "answer[%d]=%d)\n", ret / T, ret % T, ret, 
                    output[ret], ret, answer[ret]);
            }
        }
    }

    /* MPI Finalization */
    MPI_Finalize();

    return 0;
}