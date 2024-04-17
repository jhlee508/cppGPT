#include <cstdio>
#include <cmath>

#include "model.h"
#include "util.h"


/* [Parameters] */
/* 12 Transformer Blocks */
Tensor *attn_b[N_LAYER], *attn_w[N_LAYER];
Tensor *proj_b[N_LAYER], *proj_w[N_LAYER];
Tensor *ln_1_b[N_LAYER], *ln_1_g[N_LAYER];
Tensor *ln_2_b[N_LAYER], *ln_2_g[N_LAYER];
Tensor *mlp1_b[N_LAYER], *mlp1_w[N_LAYER];
Tensor *mlp2_b[N_LAYER], *mlp2_w[N_LAYER];
/* Final Layer Normalization */
Tensor *ln_f_b, *ln_f_g;
/* Word Positional and Token Embedding */
Tensor *wpe, *wte;

/* [Activations] */
Tensor *embd;
Tensor *ffn_proj_output;
Tensor *mha_qkv_output;
Tensor *mha_output;
Tensor *mha_qkv_split_tmp;
Tensor *mha_qkv_head_tmp;
Tensor *mha_mask_tmp;
Tensor *mha_merge_head_tmp;
Tensor *mha_q_tmp;
Tensor *mha_k_tmp;
Tensor *mha_v_tmp;
Tensor *mha_attn_output;
Tensor *mha_concat_head_tmp;
Tensor *attn_score_output;
Tensor *zero_seq_tmp;
Tensor *k_transposed_tmp;
Tensor *zero_dv_tmp;
Tensor *wte_transposed_tmp;
Tensor *zero_vocab_tmp;
Tensor *residual_tmp;
Tensor *logit_output;
Tensor *transformer_block_output;


/* [Model Initialization] */
void initialize_parameters(const char* param_fname) {
    size_t param_size;
    fprintf(stderr, "[LOG] Loading param from %s\n", param_fname);
    float* param = (float*)read_binary(param_fname, &param_size);
    fprintf(stderr, "[LOG] Total param size: %zu\n", param_size);

    /* Loading parameters */
    size_t pos = 0;
    /* (The stored order of OpenAI's GPT2-small parameter checkpoints) */
    int order[] = {0, 1, 10, 11, 2, 3, 4, 5, 6, 7, 8, 9, };
    for (int i = 0; i < N_LAYER; i++) {
        attn_b[order[i]] = new Tensor({3*N_EMBD}, param + pos); pos += OFFSET1;
        attn_w[order[i]] = new Tensor({N_EMBD, 3*N_EMBD}, param + pos); pos += OFFSET2;
        proj_b[order[i]] = new Tensor({N_EMBD}, param + pos); pos += OFFSET3;
        proj_w[order[i]] = new Tensor({N_EMBD, N_EMBD}, param + pos); pos += OFFSET4;
        ln_1_b[order[i]] = new Tensor({N_EMBD}, param + pos); pos += OFFSET3;
        ln_1_g[order[i]] = new Tensor({N_EMBD}, param + pos); pos += OFFSET3;
        ln_2_b[order[i]] = new Tensor({N_EMBD}, param + pos); pos += OFFSET3;
        ln_2_g[order[i]] = new Tensor({N_EMBD}, param + pos); pos += OFFSET3;
        mlp1_b[order[i]] = new Tensor({4*N_EMBD}, param + pos); pos += OFFSET5;
        mlp1_w[order[i]] = new Tensor({N_EMBD, 4*N_EMBD}, param + pos); pos += OFFSET6;
        mlp2_b[order[i]] = new Tensor({N_EMBD}, param + pos); pos += OFFSET3;
        mlp2_w[order[i]] = new Tensor({4*N_EMBD, N_EMBD}, param + pos); pos += OFFSET6;
    }
    ln_f_b = new Tensor({N_EMBD}, param + pos); pos += OFFSET3;
    ln_f_g = new Tensor({N_EMBD}, param + pos); pos += OFFSET3;
    wpe = new Tensor({N_CTX, N_EMBD}, param + pos); pos += OFFSET7;
    wte = new Tensor({N_VOCAB, N_EMBD}, param + pos); pos += OFFSET8;
    
    if (pos != param_size) {
        fprintf(stderr, "[ERROR] Loading param failed: %zu != %zu\n", pos, param_size);
        exit(1);
    }
}

void initialize_activations() {
    // Embedding
    embd = new Tensor({N_SEQ, N_EMBD});

    // FFN
    ffn_proj_output = new Tensor({N_SEQ, 4*N_EMBD});

    // MHA
    mha_qkv_output = new Tensor({N_SEQ, 3*N_EMBD});
    mha_output = new Tensor({N_SEQ, N_EMBD});
    mha_qkv_split_tmp = new Tensor({3, N_SEQ, N_EMBD});
    mha_qkv_head_tmp = new Tensor({3, N_HEAD, N_SEQ, N_EMBD/N_HEAD});
    mha_mask_tmp = new Tensor({N_SEQ, N_SEQ});
    mha_merge_head_tmp = new Tensor({N_HEAD, N_SEQ, N_EMBD/N_HEAD});
    mha_q_tmp = new Tensor({N_SEQ, N_EMBD/N_HEAD});
    mha_k_tmp = new Tensor({N_SEQ, N_EMBD/N_HEAD});
    mha_v_tmp = new Tensor({N_SEQ, N_EMBD/N_HEAD});
    mha_attn_output = new Tensor({N_SEQ, N_EMBD/N_HEAD});
    mha_concat_head_tmp = new Tensor({N_SEQ, N_EMBD});

    // Attention
    attn_score_output = new Tensor({N_SEQ, N_SEQ});
    zero_seq_tmp = new Tensor({N_SEQ});
    k_transposed_tmp = new Tensor({N_EMBD/N_HEAD, N_SEQ});
    zero_dv_tmp = new Tensor({N_EMBD/N_HEAD});

    // Projection to vocab dimension 
    wte_transposed_tmp = new Tensor({N_EMBD, N_VOCAB});
    zero_vocab_tmp = new Tensor({N_VOCAB});

    // Transformer block
    residual_tmp = new Tensor({N_SEQ, N_EMBD});
    logit_output = new Tensor({N_SEQ, N_VOCAB}); 
    transformer_block_output = new Tensor({N_SEQ, N_EMBD});
}


/* Token + Positional Embedding
 * @param [x] input: [N_SEQ]
 * @param [wte] input: [N_VOCAB, N_EMBD]
 * @param [wpe] input: [N_CTX, N_EMBD]
 * @param [x_out] output: [N_SEQ, N_EMBD]
*/
void token_pos_embedding(vector<int> x, 
                         Tensor* wte, Tensor* wpe, 
                         Tensor* x_out) {

    for (int i = 0; i < N_SEQ; i++) {
        for (int j = 0; j < N_EMBD; j++) {
            x_out->buf[i*N_EMBD + j] = 
                wte->buf[x[i]*N_EMBD + j] + 
                wpe->buf[i*N_EMBD + j];
        }
    }
}

/* GELU 
 * @param [x] input: [N_SEQ, 4*N_EMBD]
 * @param [x] output: [N_SEQ, 4*N_EMBD]
 */
void gelu(Tensor* x) {

    for (int i = 0; i < N_SEQ; i++) {
        for (int j = 0; j < 4*N_EMBD; j++) {
            x->buf[i*4*N_EMBD + j] = 0.5 * x->buf[i*4*N_EMBD + j] * 
                (1.f + tanh(sqrt(2.f / MATH_PI) * (x->buf[i*4*N_EMBD + j] + 
                0.044715f * x->buf[i*4*N_EMBD + j] * x->buf[i*4*N_EMBD + j] * 
                x->buf[i*4*N_EMBD + j])));
        }
    }
}

/* Softmax (w/ Max Trick)
 * @param [x] input: [N, D]
 * @param [x] output: [N, D]
 */
void softmax(Tensor* x) {

    int N = x->shape[0];
    int D = x->shape[1];

    for (int i = 0; i < N; i++) {
        float max_val = x->buf[i*D];
        for (int j = 1; j < D; j++) {
            if (x->buf[i*D + j] > max_val) {
                max_val = x->buf[i*D + j];
            }
        }
        float sum = 0;
        for (int j = 0; j < D; j++) {
            x->buf[i*D + j] = exp(x->buf[i*D + j] - max_val);
            sum += x->buf[i*D + j];
        }
        for (int j = 0; j < D; j++) {
            x->buf[i*D + j] /= sum;
        }
    }
}

/* Layer Normalization
 * @param [x] input: [N_SEQ, N_EMBD]
 * @param [gamma] input: [N_EMBD]
 * @param [beta] input: [N_EMBD]
 * @param [x] output: [N_SEQ, N_EMBD]
 */
void layer_norm(Tensor* x, 
                Tensor* gamma, Tensor* beta) {

    float eps = 1e-5;
    for (int i = 0; i < N_SEQ; i++) {
        float mean = 0;
        float var = 0;
        for (int j = 0; j < N_EMBD; j++) {
            mean += x->buf[i*N_EMBD + j];
            var += x->buf[i*N_EMBD + j] * x->buf[i*N_EMBD + j];
        }
        mean /= N_EMBD;
        var = var / N_EMBD - mean * mean;
        for (int j = 0; j < N_EMBD; j++) {
            x->buf[i*N_EMBD + j] = (x->buf[i*N_EMBD + j] - mean) * 
            (1.0 / sqrt(var + eps)) * gamma->buf[j] + beta->buf[j];
        }
    }
}

/* Linear 
 * @param [x] input: [M, IN]
 * @param [w] input: [IN, OUT]
 * @param [b] input: [OUT]
 * @param [x_out] output: [M, OUT]
 */
void linear(Tensor* x, 
            Tensor* w, Tensor* b, 
            Tensor* x_out) {

    int M = x->shape[0];
    int IN = x->shape[1];
    int OUT = w->shape[1]; 

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < OUT; j++) {
            x_out->buf[i*OUT + j] = 0;
            for (int k = 0; k < IN; k++) {
                x_out->buf[i*OUT + j] += x->buf[i*IN + k] * w->buf[k*OUT + j];
            }
            x_out->buf[i*OUT + j] += b->buf[j];
        }
    }
}

/* Transpose
 * @param [x] input: [M, N]
 * @param [x_out] output: [N, M]
 */
void transpose(Tensor* x, 
               Tensor* x_out) {

    int M = x->shape[0];
    int N = x->shape[1];

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            x_out->buf[j*M + i] = x->buf[i*N + j];
        }
    }
}

/* (Elem-wise) Scale
 * @param [x] input: [N_SEQ, N_SEQ]
 * @param [scale] input: [1]
 */
void scaling(Tensor* x, float scale) {

    for (int i = 0; i < N_SEQ; i++) {
        for (int j = 0; j < N_SEQ; j++) {
            x->buf[i*N_SEQ + j] *= scale;
        }
    }
}

/* (Elem-wise) Masking
 * @param [x] input: [N_SEQ, N_SEQ]
 * @param [mask] input: [N_SEQ, N_SEQ]
 * @param [x] output: [N_SEQ, N_SEQ]
 */
void masking(Tensor* x, Tensor* mask) {

    for (int i = 0; i < N_SEQ; i++) {
        for (int j = 0; j < N_SEQ; j++) {
            x->buf[i*N_SEQ + j] += mask->buf[i*N_SEQ + j];
        }
    }
}

/* (Elem-wise) Copy
 * @param [x] input: [N_SEQ, N_EMBD]
 * @param [x_out] output: [N_SEQ, N_EMBD] 
 */
void copy(Tensor* x, Tensor* x_out) {

    for (int i = 0; i < N_SEQ; i++) {
        for (int j = 0; j < N_EMBD; j++) {
            x_out->buf[i*N_EMBD + j] = x->buf[i*N_EMBD + j];
        }
    }
}

/* (Elem-wise) Addition
 * @param [x] input: [N_SEQ, N_EMBD]
 * @param [add] input: [N_SEQ, N_EMBD]
 * @param [x] output: [N_SEQ, N_EMBD]
 */
void addition(Tensor* x, Tensor* add) {

    for (int i = 0; i < N_SEQ; i++) {
        for (int j = 0; j < N_EMBD; j++) {
            x->buf[i*N_EMBD + j] += add->buf[i*N_EMBD + j];
        }
    }
}

/* Greedy Sampling
 * @param [x] input: [N_VOCAB]
 * @return [max_idx] output: [1]
 */
int greedy_sampling(Tensor* x) {

    int max_idx = 0;
    float max_val = -INFINITY;
    for (int i = 0; i < N_VOCAB; i++) {
        if (x->buf[(N_SEQ-1)*N_VOCAB + i] > max_val) {
            max_val = x->buf[(N_SEQ-1)*N_VOCAB + i];
            max_idx = i;
        }
    }
    
    return max_idx;
}

/* (Position-wise) Feed-Forward Network
 * @param [x] input: [N_SEQ, N_EMBD]
 * @param [mlp1_w] input: [N_EMBD, 4*N_EMBD]
 * @param [mlp1_b] input: [4*N_EMBD]
 * @param [mlp2_w] input: [4*N_EMBD, N_EMBD]
 * @param [mlp2_b] input: [N_EMBD]
 * @param [x_out] output: [N_SEQ, N_EMBD]
 */
void ffn(Tensor* x, 
         Tensor* mlp1_w, Tensor* mlp1_b, 
         Tensor* mlp2_w, Tensor* mlp2_b, 
         Tensor* x_out) {
    
    /* Projection Up [N_SEQ, N_EMBD] -> [N_SEQ, 4*N_EMBD] */
    linear(x, mlp1_w, mlp1_b, ffn_proj_output);

    /* GELU */
    gelu(ffn_proj_output);

    /* Projection Down [N_SEQ, 4*N_EMBD] -> [N_SEQ, N_EMBD] */
    linear(ffn_proj_output, mlp2_w, mlp2_b, x_out);
}

/* Attention
 * @param [q] input: [N_Q, D_K]
 * @param [k] input: [N_K, D_K]
 * @param [v] input: [N_K, D_V]
 * @param [mask] input: [N_Q, N_K]
 * @param [out] output: [N_Q, D_V]
 */
void attention(Tensor* q, Tensor* k, Tensor* v, 
               Tensor* mask, 
               Tensor* out) {

    int N_Q = q->shape[0]; 
    int N_K = k->shape[0]; 
    int D_K = k->shape[1];

    /* q @ k.T */
    transpose(k, k_transposed_tmp);
    linear(q, k_transposed_tmp, zero_seq_tmp, attn_score_output);

    /* Elem-wise scaling */    
    // for (int i = 0; i < N_Q; i++) {
    //     for (int j = 0; j < N_K; j++) {
    //         attn_score_output->buf[i*N_K + j] /= sqrt(D_K);
    //     }
    // }
    scaling(attn_score_output, 1.0 / sqrt(D_K));
    

    /* Apply mask */
    // for (int i = 0; i < N_Q; i++) {
    //     for (int j = 0; j < N_K; j++) {
    //         attn_score_output->buf[i*N_K + j] += 
    //             mask->buf[i*N_K + j];
    //     }
    // }
    masking(attn_score_output, mask);

    /* softmax */
    softmax(attn_score_output);

    /* attn_score_output @ v */
    linear(attn_score_output, v, zero_dv_tmp, out);
} 

/* (Masked) Multi-Head Self Attention 
 * @param [x] input: [N_SEQ, N_EMBD]
 * @param [attn_b] input: [3*N_EMBD]
 * @param [attn_w] input: [N_EMBD, 3*N_EMBD]
 * @param [proj_b] input: [N_EMBD]
 * @param [proj_w] input: [N_EMBD, N_EMBD]
 * @param [x_out] output: [N_SEQ, N_EMBD]
 */
void mha(Tensor* x, 
    Tensor* attn_b, Tensor* attn_w, 
    Tensor* proj_b, Tensor* proj_w, 
    Tensor* x_out) {

    /* QKV projection: [n_seq, n_embd] -> [n_seq, 3*n_embd]) */
    linear(x, attn_w, attn_b, mha_qkv_output);

    /* Split into qkv: [n_seq, 3*n_embd] -> [3, n_seq, n_embd] */
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < N_SEQ; j++) {
            for (int k = 0; k < N_EMBD; k++) {
                mha_qkv_split_tmp->buf[i*N_SEQ*N_EMBD + j*N_EMBD + k] = 
                    mha_qkv_output->buf[j*3*N_EMBD + i*N_EMBD + k];
            }
        }
    }

    /* Split into heads: [3, n_seq, n_embd] -> [3, n_head, n_seq, n_embd/n_head] */
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < N_HEAD; j++) {
            for (int k = 0; k < N_SEQ; k++) {
                for (int l = 0; l < N_EMBD/N_HEAD; l++) {
                    mha_qkv_head_tmp->buf[i*N_HEAD*N_SEQ*N_EMBD/N_HEAD + j*N_SEQ*N_EMBD/N_HEAD + k*N_EMBD/N_HEAD + l] = 
                        mha_qkv_split_tmp->buf[i*N_SEQ*N_EMBD + k*N_EMBD + j*N_EMBD/N_HEAD + l];
                }
            }
        }
    }

    /* Generate mask to hide future inputs */
    for (int i = 0; i < N_SEQ; i++) {
        for (int j = 0; j < N_SEQ; j++) {
            if (i >= j) {
                mha_mask_tmp->buf[i*N_SEQ + j] = 0;
            } else {
                mha_mask_tmp->buf[i*N_SEQ + j] = -1e10;
            }
        }
    }
    
    /* Perform Attention over each head [n_head, 3, n_seq, n_embd/n_head] -> [n_head, n_seq, n_embd/n_head] */
    for (int i = 0; i < N_HEAD; i++) {

        /* Extract q, k, v from qkv_head */
        for (int j = 0; j < N_SEQ; j++) {
            for (int k = 0; k < N_EMBD/N_HEAD; k++) {
                mha_q_tmp->buf[j*N_EMBD/N_HEAD + k] = 
                    mha_qkv_head_tmp->buf[0*N_HEAD*N_SEQ*N_EMBD/N_HEAD + i*N_SEQ*N_EMBD/N_HEAD + j*N_EMBD/N_HEAD + k];
                mha_k_tmp->buf[j*N_EMBD/N_HEAD + k] = 
                    mha_qkv_head_tmp->buf[1*N_HEAD*N_SEQ*N_EMBD/N_HEAD + i*N_SEQ*N_EMBD/N_HEAD + j*N_EMBD/N_HEAD + k];
                mha_v_tmp->buf[j*N_EMBD/N_HEAD + k] = 
                    mha_qkv_head_tmp->buf[2*N_HEAD*N_SEQ*N_EMBD/N_HEAD + i*N_SEQ*N_EMBD/N_HEAD + j*N_EMBD/N_HEAD + k];
            }
        }

        /* Attention */
        attention(mha_q_tmp, mha_k_tmp, mha_v_tmp, mha_mask_tmp, mha_attn_output);

        /* Merge each head's attn output [n_seq, n_embd/n_head] -> [n_head, n_seq, n_embd/n_head] */
        for (int j = 0; j < N_SEQ; j++) {
            for (int k = 0; k < N_EMBD/N_HEAD; k++) {
                mha_merge_head_tmp->buf[i*N_SEQ*N_EMBD/N_HEAD + j*N_EMBD/N_HEAD + k] = 
                    mha_attn_output->buf[j*N_EMBD/N_HEAD + k];
            }
        }
    }

    /* Concat each heads [n_head, n_seq, n_embd/n_head] -> [n_seq, n_embd] */
    for (int i = 0; i < N_SEQ; i++) {
        for (int j = 0; j < N_HEAD; j++) {
            for (int k = 0; k < N_EMBD/N_HEAD; k++) {
                mha_concat_head_tmp->buf[i*N_EMBD + j*N_EMBD/N_HEAD + k] = 
                    mha_merge_head_tmp->buf[j*N_SEQ*N_EMBD/N_HEAD + i*N_EMBD/N_HEAD + k];
            }
        }
    }

    /* OUT projection [n_seq, n_embd] -> [n_seq, n_embd] */
    linear(mha_concat_head_tmp, proj_w, proj_b, x_out);
}

/* Transformer Block
 * @param [x] input: [N_SEQ, N_EMBD]
 * @param [attn_b] input: [3*N_EMBD]
 * @param [attn_w] input: [N_EMBD, 3*N_EMBD]
 * @param [proj_b] input: [N_EMBD]
 * @param [proj_w] input: [N_EMBD, N_EMBD]
 * @param [ln_1_b] input: [N_EMBD]
 * @param [ln_1_g] input: [N_EMBD]
 * @param [ln_2_b] input: [N_EMBD]
 * @param [ln_2_g] input: [N_EMBD]
 * @param [mlp1_b] input: [4*N_EMBD]
 * @param [mlp1_w] input: [N_EMBD, 4*N_EMBD]
 * @param [mlp2_b] input: [N_EMBD]
 * @param [mlp2_w] input: [4*N_EMBD, N_EMBD]
 * @param [x_out] output: [N_SEQ, N_EMBD]
 */
void transformer_block(Tensor* x, 
    Tensor* attn_b, Tensor* attn_w, 
    Tensor* proj_b, Tensor* proj_w, 
    Tensor* ln_1_b, Tensor* ln_1_g, 
    Tensor* ln_2_b, Tensor* ln_2_g, 
    Tensor* mlp1_b, Tensor* mlp1_w, 
    Tensor* mlp2_b, Tensor* mlp2_w,
    Tensor* x_out) {

    /* Copy Residual */
    // for (int i = 0; i < N_SEQ*N_EMBD; i++) {
    //     residual_tmp->buf[i] = x->buf[i];
    // }
    copy(x, residual_tmp);

    /* Layer Normalization */
    layer_norm(x, ln_1_g, ln_1_b);

    /* Masked Multi-Head Self Attention */
    mha(x, attn_b, attn_w, proj_b, proj_w, mha_output);

    /* Add Residual */
    // for (int i = 0; i < N_SEQ*N_EMBD; i++) {
    //     mha_output->buf[i] += residual_tmp->buf[i];
    // }   
    addition(mha_output, residual_tmp);

    /* Copy Residual */
    // for (int i = 0; i < N_SEQ*N_EMBD; i++) {
    //     residual_tmp->buf[i] = mha_output->buf[i];
    // }
    copy(mha_output, residual_tmp);

    /* Layer Normalization */
    layer_norm(mha_output, ln_2_g, ln_2_b);

    /* Position-wise Feed-Forward Network */
    ffn(mha_output, mlp1_w, mlp1_b, mlp2_w, mlp2_b, x_out);

    /* Add Residual */
    // for (int i = 0; i < N_SEQ*N_EMBD; i++) {
    //     x_out->buf[i] += residual_tmp->buf[i];
    // }
    addition(x_out, residual_tmp);
}


/* [Token Generation] */
void generate_tokens(int* input, int n_token) {
    
    int n_prompt = 1;
    /* Outer loop: select a single prompt */
    for (int p = 0; p < n_prompt * N_SEQ; p+=N_SEQ) {
        
        /* Initialize input prompt */
        vector<int> input_prompt(N_SEQ);
        memcpy(input_prompt.data(), input + p, N_SEQ * sizeof(int));

        /* Inner loop: generate next token */
        for (int t = 1; t <= n_token; t++) {

            /* Token + Positional Embedding */
            token_pos_embedding(input_prompt, wte, wpe, embd);

            /* Forward path of Transformer blocks */
            for (int i = 0; i < N_LAYER; i++) {
                transformer_block(embd, 
                    attn_b[i], attn_w[i], 
                    proj_b[i], proj_w[i], 
                    ln_1_b[i], ln_1_g[i], 
                    ln_2_b[i], ln_2_g[i], 
                    mlp1_b[i], mlp1_w[i], 
                    mlp2_b[i], mlp2_w[i], 
                    transformer_block_output);

                /* Copy output to embd for next block */
                copy(transformer_block_output, embd);
            }

            /* Final Layer Normalization */
            layer_norm(embd, ln_f_g, ln_f_b);

            /* Projection to vocab. dimension */
            transpose(wte, wte_transposed_tmp);
            linear(embd, wte_transposed_tmp, zero_vocab_tmp, logit_output);

            /* Greedy sampling (only last timestep is considered) */
            int next_token_id = greedy_sampling(logit_output);

            /* Print generated token ID */
            fprintf(stdout, " [DEBUG] Generated token ID: %d\n", next_token_id);

            /* Update input sequence and N_SEQ length */
            N_SEQ += 1;
            input_prompt.push_back(next_token_id);
            
            /* Re-initialize activations for next token generation */
            finalize_activations();
            initialize_activations();
        }  
    }
}


/* [Model Finalization] */
void finalize_parameters() {

    /* Freeing parameters */
    for (int i = 0; i < N_LAYER; i++) {
        delete attn_b[i];
        delete attn_w[i];
        delete proj_b[i];
        delete proj_w[i];
        delete ln_1_b[i];
        delete ln_1_g[i];
        delete ln_2_b[i];
        delete ln_2_g[i];
        delete mlp1_b[i];
        delete mlp1_w[i];
        delete mlp2_b[i];
        delete mlp2_w[i];
    }
    delete ln_f_b;
    delete ln_f_g;
    delete wpe;
    delete wte;
}

void finalize_activations() {

    /* Freeing activations */
    delete embd;
    delete ffn_proj_output;
    delete mha_qkv_output;
    delete mha_output;
    delete mha_qkv_split_tmp;
    delete mha_qkv_head_tmp;
    delete mha_mask_tmp;
    delete mha_merge_head_tmp;
    delete mha_q_tmp;
    delete mha_k_tmp;
    delete mha_v_tmp;
    delete mha_attn_output;
    delete mha_concat_head_tmp;
    delete attn_score_output;
    delete zero_seq_tmp;
    delete k_transposed_tmp;
    delete wte_transposed_tmp;
    delete zero_dv_tmp;
    delete zero_vocab_tmp;
    delete residual_tmp;
    delete logit_output;
    delete transformer_block_output;
}
