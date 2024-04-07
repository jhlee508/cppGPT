#include <cmath>
#include <cstdio>
#include <cmath>

#include "model.h"
#include "util.h"


/* [Parameters] */
/* 'blocks': 12 Transformer Blocks in Model */
Tensor *attn_b[N_LAYER], *attn_w[N_LAYER];
Tensor *proj_b[N_LAYER], *proj_w[N_LAYER];
Tensor *ln_1_b[N_LAYER], *ln_1_g[N_LAYER];
Tensor *ln_2_b[N_LAYER], *ln_2_g[N_LAYER];
Tensor *mlp1_b[N_LAYER], *mlp1_w[N_LAYER];
Tensor *mlp2_b[N_LAYER], *mlp2_w[N_LAYER];
/* 'ln_f': Final Layer Normalization */
Tensor *ln_f_b, *ln_f_g;
/* 'wpe' & 'wte': Word Positional and Token Embedding */
Tensor *wpe, *wte;

/* [Activations] */
Tensor *embd;
Tensor *ffn_proj_act;
Tensor *mha_Wqkv_out_act, *mha_out_act, *mha_qkv_act, *mha_qkv_head_act;
Tensor *mha_mask;
Tensor *mha_q_act, *mha_k_act, *mha_v_act;
Tensor *mha_out_head_act;
Tensor *mha_attn_out_act;
Tensor *mha_concat_head_act;
Tensor *attn_score_act, *zero_seq_act;
Tensor *wte_T_act, *zero_vocab_act;
Tensor *residual_act;
Tensor *tx_block_out_act;
Tensor *logits_out_act;


/* [Model Initialization] */
void initialize_model(const char* param_fname) {
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

    /* Initializing activations */
    // Embedding
    embd = new Tensor({N_SEQ, N_EMBD});

    // FFN
    ffn_proj_act = new Tensor({N_SEQ, 4*N_EMBD});

    // MHA
    mha_Wqkv_out_act = new Tensor({N_SEQ, 3*N_EMBD});
    mha_out_act = new Tensor({N_SEQ, N_EMBD});
    mha_qkv_act = new Tensor({3, N_SEQ, N_EMBD});
    mha_qkv_head_act = new Tensor({3, N_HEAD, N_SEQ, N_EMBD/N_HEAD});
    mha_mask = new Tensor({N_SEQ, N_SEQ});
    mha_out_head_act = new Tensor({N_HEAD, N_SEQ, N_EMBD/N_HEAD});
    mha_q_act = new Tensor({N_SEQ, N_EMBD/N_HEAD});
    mha_k_act = new Tensor({N_SEQ, N_EMBD/N_HEAD});
    mha_v_act = new Tensor({N_SEQ, N_EMBD/N_HEAD});
    mha_attn_out_act = new Tensor({N_SEQ, N_EMBD/N_HEAD});
    mha_concat_head_act = new Tensor({N_SEQ, N_EMBD});

    // Attention
    attn_score_act = new Tensor({N_SEQ, N_SEQ});
    zero_seq_act = new Tensor({N_SEQ});

    // Projection to vocab dimension 
    wte_T_act = new Tensor({N_EMBD, N_VOCAB});
    zero_vocab_act = new Tensor({N_VOCAB});

    // Transformer block
    residual_act = new Tensor({N_SEQ, N_EMBD});
    logits_out_act = new Tensor({N_SEQ, N_VOCAB}); 
    tx_block_out_act = new Tensor({N_SEQ, N_EMBD});
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
            wte->buf[x[i]*N_EMBD + j] + wpe->buf[i*N_EMBD + j];
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
    linear(x, mlp1_w, mlp1_b, ffn_proj_act);

    /* GELU */
    gelu(ffn_proj_act);

    /* Projection Down [N_SEQ, 4*N_EMBD] -> [N_SEQ, N_EMBD] */
    linear(ffn_proj_act, mlp2_w, mlp2_b, x_out);
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
    int D_V = v->shape[1];

    Tensor* k_T = new Tensor({D_K, N_K});

    /* q @ k.T */
    transpose(k, k_T);
    linear(q, k_T, zero_seq_act, attn_score_act);

    delete k_T;

    /* Elem-wise scaling */    
    for (int i = 0; i < N_Q; i++) {
        for (int j = 0; j < N_K; j++) {
            attn_score_act->buf[i*N_K + j] /= sqrt(D_K);
        }
    }

    /* Apply mask */
    for (int i = 0; i < N_Q; i++) {
        for (int j = 0; j < N_K; j++) {
            attn_score_act->buf[i*N_K + j] += 
                mask->buf[i*N_K + j];
        }
    }

    /* softmax */
    softmax(attn_score_act);

    /* attn_score_act @ v */
    Tensor* zero_bias = new Tensor({D_V});
    linear(attn_score_act, v, zero_bias, out);
    delete zero_bias;
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
    linear(x, attn_w, attn_b, mha_Wqkv_out_act);

    /* Split into qkv: [n_seq, 3*n_embd] -> [3, n_seq, n_embd] */
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < N_SEQ; j++) {
            for (int k = 0; k < N_EMBD; k++) {
                mha_qkv_act->buf[i*N_SEQ*N_EMBD + j*N_EMBD + k] = 
                    mha_Wqkv_out_act->buf[j*3*N_EMBD + i*N_EMBD + k];
            }
        }
    }

    /* Split into heads: [3, n_seq, n_embd] -> [3, n_head, n_seq, n_embd/n_head] */
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < N_HEAD; j++) {
            for (int k = 0; k < N_SEQ; k++) {
                for (int l = 0; l < N_EMBD/N_HEAD; l++) {
                    mha_qkv_head_act->buf[i*N_HEAD*N_SEQ*N_EMBD/N_HEAD + j*N_SEQ*N_EMBD/N_HEAD + k*N_EMBD/N_HEAD + l] = 
                        mha_qkv_act->buf[i*N_SEQ*N_EMBD + k*N_EMBD + j*N_EMBD/N_HEAD + l];
                }
            }
        }
    }

    /* Generate mask to hide future inputs */
    for (int i = 0; i < N_SEQ; i++) {
        for (int j = 0; j < N_SEQ; j++) {
            if (i >= j) {
                mha_mask->buf[i*N_SEQ + j] = 0;
            } else {
                mha_mask->buf[i*N_SEQ + j] = -1e10;
            }
        }
    }
    
    /* Perform Attention over each head [n_head, 3, n_seq, n_embd/n_head] -> [n_head, n_seq, n_embd/n_head] */
    for (int i = 0; i < N_HEAD; i++) {

        /* Extract q, k, v from qkv_head */
        for (int j = 0; j < N_SEQ; j++) {
            for (int k = 0; k < N_EMBD/N_HEAD; k++) {
                mha_q_act->buf[j*N_EMBD/N_HEAD + k] = 
                    mha_qkv_head_act->buf[0*N_HEAD*N_SEQ*N_EMBD/N_HEAD + i*N_SEQ*N_EMBD/N_HEAD + j*N_EMBD/N_HEAD + k];
                mha_k_act->buf[j*N_EMBD/N_HEAD + k] = 
                    mha_qkv_head_act->buf[1*N_HEAD*N_SEQ*N_EMBD/N_HEAD + i*N_SEQ*N_EMBD/N_HEAD + j*N_EMBD/N_HEAD + k];
                mha_v_act->buf[j*N_EMBD/N_HEAD + k] = 
                    mha_qkv_head_act->buf[2*N_HEAD*N_SEQ*N_EMBD/N_HEAD + i*N_SEQ*N_EMBD/N_HEAD + j*N_EMBD/N_HEAD + k];
            }
        }

        /* Attention */
        attention(mha_q_act, mha_k_act, mha_v_act, mha_mask, mha_attn_out_act);

        /* Merge each head's attn output [n_seq, n_embd/n_head] -> [n_head, n_seq, n_embd/n_head] */
        for (int j = 0; j < N_SEQ; j++) {
            for (int k = 0; k < N_EMBD/N_HEAD; k++) {
                mha_out_head_act->buf[i*N_SEQ*N_EMBD/N_HEAD + j*N_EMBD/N_HEAD + k] = 
                    mha_attn_out_act->buf[j*N_EMBD/N_HEAD + k];
            }
        }
    }

    /* Concat each heads [n_head, n_seq, n_embd/n_head] -> [n_seq, n_embd] */
    for (int i = 0; i < N_SEQ; i++) {
        for (int j = 0; j < N_HEAD; j++) {
            for (int k = 0; k < N_EMBD/N_HEAD; k++) {
                mha_concat_head_act->buf[i*N_EMBD + j*N_EMBD/N_HEAD + k] = 
                    mha_out_head_act->buf[j*N_SEQ*N_EMBD/N_HEAD + i*N_EMBD/N_HEAD + k];
            }
        }
    }

    /* OUT projection [n_seq, n_embd] -> [n_seq, n_embd] */
    linear(mha_concat_head_act, proj_w, proj_b, x_out);
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
    for (int i = 0; i < N_SEQ*N_EMBD; i++) {
        residual_act->buf[i] = x->buf[i];
    }

    /* Layer Normalization */
    layer_norm(x, ln_1_g, ln_1_b);

    /* Masked Multi-Head Self Attention */
    mha(x, attn_b, attn_w, proj_b, proj_w, mha_out_act);

    /* Add Residual */
    for (int i = 0; i < N_SEQ*N_EMBD; i++) {
        mha_out_act->buf[i] += residual_act->buf[i];
    }   

    /* Copy Residual */
    for (int i = 0; i < N_SEQ*N_EMBD; i++) {
        residual_act->buf[i] = mha_out_act->buf[i];
    }

    /* Layer Normalization */
    layer_norm(mha_out_act, ln_2_g, ln_2_b);

    /* Position-wise Feed-Forward Network */
    ffn(mha_out_act, mlp1_w, mlp1_b, mlp2_w, mlp2_b, x_out);

    /* Add Residual */
    for (int i = 0; i < N_SEQ*N_EMBD; i++) {
        x_out->buf[i] += residual_act->buf[i];
    }
}


/* [Token Generation] */
void generate_tokens(vector<int> input, int n_token) {

    /* Iteratively generate next token */
    for (int t = 0; t < n_token; t++) {

        /* Token + Positional Embedding */
        token_pos_embedding(input, wte, wpe, embd);

        /* Forward path through Transformer layers */
        for (int i = 0; i < N_LAYER; i++) {
            transformer_block(embd, 
                attn_b[i], attn_w[i], 
                proj_b[i], proj_w[i], 
                ln_1_b[i], ln_1_g[i], 
                ln_2_b[i], ln_2_g[i], 
                mlp1_b[i], mlp1_w[i], 
                mlp2_b[i], mlp2_w[i], 
                tx_block_out_act);

            /* Copy output to embd for next layer */
            for (int j = 0; j < N_SEQ*N_EMBD; j++) {
                embd->buf[j] = tx_block_out_act->buf[j];
            }
        }

        /* Final LayerNorm */
        layer_norm(embd, ln_f_g, ln_f_b);

        /* Projection to vocab dimension */
        transpose(wte, wte_T_act);
        linear(embd, wte_T_act, zero_vocab_act, logits_out_act);

        /* Greedy sampling (only last timestep is considered) */
        int next_token_id = -1;
        float max_val = -INFINITY;
        for (int i = 0; i < N_VOCAB; i++) {
            if (logits_out_act->buf[(N_SEQ-1)*N_VOCAB + i] > max_val) {
                max_val = logits_out_act->buf[(N_SEQ-1)*N_VOCAB + i];
                next_token_id = i;
            }
        }

        /* Print generated token */
        printf(" >>> Next token ID: %d\n", next_token_id);

        /* Update input for next iteration */
        input.push_back(next_token_id);
        input.erase(input.begin());
    }   
}


/* [Model Finalization] */
void finalize_model() {

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

    /* Freeing activations */
    delete embd;
    delete ffn_proj_act;
    delete mha_Wqkv_out_act;
    delete mha_out_act;
    delete mha_qkv_act;
    delete mha_qkv_head_act;
    delete mha_mask;
    delete mha_out_head_act;
    delete mha_q_act;
    delete mha_k_act;
    delete mha_v_act;
    delete mha_attn_out_act;
    delete mha_concat_head_act;
    delete attn_score_act;
    delete zero_seq_act;
    delete wte_T_act;
    delete zero_vocab_act;
    delete residual_act;
    delete logits_out_act;
    delete tx_block_out_act;
}

