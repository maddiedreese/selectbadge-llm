// TinyStories-260K inference adapted from maddiedreese/imac-llm (MIT) and
// karpathy/llama2.c (MIT). The checkpoint stays in XIP flash; activations and
// the KV cache stay in SRAM. Input files are the original little-endian ones.
#include "model.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include "pico/stdlib.h"

extern "C" {
extern const uint8_t model_start[], model_end[];
extern const uint8_t tokenizer_start[], tokenizer_end[];
}

namespace model {
namespace {
struct Config {
  int32_t dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, max_seq_len;
};
struct Weights {
  const float *embed, *rms_att, *wq, *wk, *wv, *wo, *rms_ffn;
  const float *w1, *w2, *w3, *rms_final, *freq_real, *freq_imag, *wcls;
};
Config cfg{};
Weights w{};
char fault[64] = "Not initialized";
char vocab[512][8]{};
float scores[512]{};
int token_ids[64]{};
int n_prompt = 0, pos = 0, token = 1, produced = 0;
bool running = false;
char result[kMaxOutput]{};
uint32_t rng = 0x9e3779b9;

float x[64], xb[64], xb2[64], hb[172], hb2[172];
float q[64], k[32], v[32], att[8 * kMaxContext], logits[512];
float key_cache[5 * kMaxContext * 32];
float value_cache[5 * kMaxContext * 32];

void rmsnorm(float *out, const float *in, const float *weight, int n) {
  float ss = 0;
  for (int i = 0; i < n; ++i) ss += in[i] * in[i];
  float scale = 1.0f / sqrtf(ss / n + 1e-5f);
  for (int i = 0; i < n; ++i) out[i] = weight[i] * scale * in[i];
}

void softmax(float *a, int n) {
  float peak = a[0], sum = 0;
  for (int i = 1; i < n; ++i) if (a[i] > peak) peak = a[i];
  for (int i = 0; i < n; ++i) { a[i] = expf(a[i] - peak); sum += a[i]; }
  for (int i = 0; i < n; ++i) a[i] /= sum;
}

void matmul(float *out, const float *in, const float *matrix, int cols, int rows) {
  for (int i = 0; i < rows; ++i) {
    float val = 0;
    const float *row = matrix + i * cols;
    for (int j = 0; j < cols; ++j) val += row[j] * in[j];
    out[i] = val;
  }
}

float *forward(int current, int position) {
  constexpr int dim = 64, kv_dim = 32, head_size = 8, hidden = 172;
  memcpy(x, w.embed + current * dim, dim * sizeof(float));
  for (int layer = 0; layer < cfg.n_layers; ++layer) {
    rmsnorm(xb, x, w.rms_att + layer * dim, dim);
    matmul(q, xb, w.wq + layer * dim * dim, dim, dim);
    matmul(k, xb, w.wk + layer * dim * kv_dim, dim, kv_dim);
    matmul(v, xb, w.wv + layer * dim * kv_dim, dim, kv_dim);

    for (int i = 0; i < dim; i += 2) {
      int j = (i % head_size) / 2;
      float real = w.freq_real[position * head_size / 2 + j];
      float imag = w.freq_imag[position * head_size / 2 + j];
      float a = q[i], b = q[i + 1];
      q[i] = a * real - b * imag;
      q[i + 1] = a * imag + b * real;
      if (i < kv_dim) {
        a = k[i]; b = k[i + 1];
        k[i] = a * real - b * imag;
        k[i + 1] = a * imag + b * real;
      }
    }

    int base = layer * kMaxContext * kv_dim;
    memcpy(key_cache + base + position * kv_dim, k, kv_dim * sizeof(float));
    memcpy(value_cache + base + position * kv_dim, v, kv_dim * sizeof(float));
    for (int head = 0; head < cfg.n_heads; ++head) {
      float *ah = att + head * kMaxContext;
      const float *qh = q + head * head_size;
      for (int t = 0; t <= position; ++t) {
        const float *kh = key_cache + base + t * kv_dim + (head / 2) * head_size;
        float score = 0;
        for (int i = 0; i < head_size; ++i) score += qh[i] * kh[i];
        ah[t] = score / sqrtf(float(head_size));
      }
      softmax(ah, position + 1);
      float *out = xb + head * head_size;
      memset(out, 0, head_size * sizeof(float));
      for (int t = 0; t <= position; ++t) {
        const float *vh = value_cache + base + t * kv_dim + (head / 2) * head_size;
        for (int i = 0; i < head_size; ++i) out[i] += ah[t] * vh[i];
      }
    }
    matmul(xb2, xb, w.wo + layer * dim * dim, dim, dim);
    for (int i = 0; i < dim; ++i) x[i] += xb2[i];
    rmsnorm(xb, x, w.rms_ffn + layer * dim, dim);
    matmul(hb, xb, w.w1 + layer * dim * hidden, dim, hidden);
    matmul(hb2, xb, w.w3 + layer * dim * hidden, dim, hidden);
    for (int i = 0; i < hidden; ++i) hb[i] = hb[i] / (1.0f + expf(-hb[i])) * hb2[i];
    matmul(xb, hb, w.w2 + layer * hidden * dim, hidden, dim);
    for (int i = 0; i < dim; ++i) x[i] += xb[i];
  }
  rmsnorm(x, x, w.rms_final, dim);
  matmul(logits, x, w.wcls, dim, cfg.vocab_size);
  return logits;
}

int lookup(const char *word) {
  for (int i = 0; i < cfg.vocab_size; ++i)
    if (strcmp(word, vocab[i]) == 0) return i;
  return -1;
}

int encode(const char *input) {
  int count = 0;
  while (*input && count < 64) {
    char ch[] = {*input++, 0};
    int id = lookup(ch);
    if (id >= 0) token_ids[count++] = id;
  }
  while (true) {
    float best = -1e10f;
    int index = -1, id = -1;
    for (int i = 0; i + 1 < count; ++i) {
      char merged[16];
      size_t n = strlen(vocab[token_ids[i]]) + strlen(vocab[token_ids[i + 1]]);
      if (n >= sizeof(merged)) continue;
      strcpy(merged, vocab[token_ids[i]]);
      strcat(merged, vocab[token_ids[i + 1]]);
      int candidate = lookup(merged);
      if (candidate >= 0 && scores[candidate] > best) {
        best = scores[candidate]; index = i; id = candidate;
      }
    }
    if (index < 0) break;
    token_ids[index] = id;
    for (int j = index + 1; j + 1 < count; ++j) token_ids[j] = token_ids[j + 1];
    --count;
  }
  return count;
}

int sample(float *values) {
  for (int i = 0; i < cfg.vocab_size; ++i) values[i] /= 0.9f;
  softmax(values, cfg.vocab_size);
  rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
  float target = float(rng & 0xffffff) / 16777216.0f;
  float cumulative = 0;
  for (int i = 0; i < cfg.vocab_size; ++i) {
    cumulative += values[i];
    if (target < cumulative) return i;
  }
  return cfg.vocab_size - 1;
}

void append(int id) {
  if (id < 3) return;
  char single[2] = {0, 0};
  const char *part = vocab[id];
  if (id <= 258) {
    int byte = id - 3;
    if (byte < 32 || byte > 126) return;
    single[0] = char(byte);
    part = single;
  }
  size_t have = strlen(result), add = strlen(part);
  if (have + add + 1 < sizeof(result)) strcat(result, part);
}
} // namespace

bool init() {
  if (size_t(model_end - model_start) != 1056540 ||
      size_t(tokenizer_end - tokenizer_start) != 6227) {
    strcpy(fault, "Model asset size mismatch"); return false;
  }
  memcpy(&cfg, model_start, sizeof(cfg));
  if (cfg.dim != 64 || cfg.hidden_dim != 172 || cfg.n_layers != 5 ||
      cfg.n_heads != 8 || cfg.n_kv_heads != 4 || cfg.vocab_size != 512 ||
      cfg.max_seq_len != 512) {
    strcpy(fault, "Unexpected checkpoint shape"); return false;
  }
  const float *p = reinterpret_cast<const float *>(model_start + sizeof(cfg));
  w.embed = p; p += cfg.vocab_size * cfg.dim;
  w.rms_att = p; p += cfg.n_layers * cfg.dim;
  w.wq = p; p += cfg.n_layers * cfg.dim * cfg.dim;
  w.wk = p; p += cfg.n_layers * cfg.dim * 32;
  w.wv = p; p += cfg.n_layers * cfg.dim * 32;
  w.wo = p; p += cfg.n_layers * cfg.dim * cfg.dim;
  w.rms_ffn = p; p += cfg.n_layers * cfg.dim;
  w.w1 = p; p += cfg.n_layers * cfg.dim * cfg.hidden_dim;
  w.w2 = p; p += cfg.n_layers * cfg.hidden_dim * cfg.dim;
  w.w3 = p; p += cfg.n_layers * cfg.dim * cfg.hidden_dim;
  w.rms_final = p; p += cfg.dim;
  w.freq_real = p; p += cfg.max_seq_len * 4;
  w.freq_imag = p; p += cfg.max_seq_len * 4;
  w.wcls = w.embed; // positive vocab_size means shared classifier weights
  if (reinterpret_cast<const uint8_t *>(p) != model_end) {
    strcpy(fault, "Checkpoint layout mismatch"); return false;
  }
  const uint8_t *t = tokenizer_start;
  int32_t max_len; memcpy(&max_len, t, 4); t += 4;
  if (max_len > 7) { strcpy(fault, "Tokenizer token too long"); return false; }
  for (int i = 0; i < 512; ++i) {
    int32_t n;
    memcpy(&scores[i], t, 4); t += 4;
    memcpy(&n, t, 4); t += 4;
    if (n < 0 || n > 7 || t + n > tokenizer_end) {
      strcpy(fault, "Tokenizer layout mismatch"); return false;
    }
    memcpy(vocab[i], t, n); vocab[i][n] = 0; t += n;
  }
  if (t != tokenizer_end) { strcpy(fault, "Tokenizer length mismatch"); return false; }
  fault[0] = 0;
  return true;
}

const char *error() { return fault; }

void begin(const char *prompt) {
  result[0] = 0;
  n_prompt = encode(prompt);
  produced = 0; pos = 0; token = 1;
  memset(key_cache, 0, sizeof(key_cache));
  memset(value_cache, 0, sizeof(value_cache));
  rng ^= time_us_32();
  if (n_prompt > 32) {
    strcpy(result, "Prompt exceeds 32 model tokens. Shorten it and retry.");
    running = false;
  } else {
    running = true;
  }
}

bool step() {
  if (!running) return false;
  float *scores = forward(token, pos);
  if (pos < n_prompt) {
    token = token_ids[pos++];
  } else {
    token = sample(scores);
    append(token);
    ++produced; ++pos;
    if (token == 2 || produced >= kMaxGenerated || pos >= kMaxContext ||
        strlen(result) > kMaxOutput - 16) running = false;
  }
  return true;
}

bool busy() { return running; }
int generated_tokens() { return produced; }
int prompt_tokens() { return n_prompt; }
const char *output() { return result; }
} // namespace model
