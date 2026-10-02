#include "badge.hpp"
#include "model.hpp"
#include "pico/stdlib.h"

#include <cstdio>
#include <cstring>

using namespace badge;

namespace {
const rgb_color_t BG(24, 24, 24, 255);
const rgb_color_t PANEL(36, 36, 36, 255);
const rgb_color_t KEY(46, 46, 46, 255);
const rgb_color_t BORDER(67, 67, 67, 255);
const rgb_color_t GREEN(62, 207, 142, 255); // Supabase brand primary
const rgb_color_t WHITE(238, 238, 238, 255);
const rgb_color_t MUTED(160, 160, 160, 255);

char prompt[model::kMaxPrompt + 1] = {};
int length = 0, row = 0, col = 0;
bool shift = false, result_view = false;
bool model_ready = false;

const char *alpha_rows[] = {"QWERTYUIOP", "ASDFGHJKL", "ZXCVBNM,.?"};
const char *actions[] = {"SPACE", "SHIFT", "DEL", "SEND"};

void box(int x, int y, int w, int h, const rgb_color_t &color) {
  pen(color);
  screen.rectangle(rect_t(x, y, w, h));
}

void label(const char *s, int x, int y, const rgb_color_t &color) {
  pen(color);
  text(s, x, y);
}

void centered(const char *s, int x, int y, int w, int h, const rgb_color_t &color) {
  int tx = x + (w - text_width(s)) / 2;
  int ty = y + (h - line_height()) / 2;
  label(s, tx, ty, color);
}

void wrapped(const char *s, int x, int y, int w, int bottom, const rgb_color_t &color) {
  char line[80]{};
  int n = 0;
  while (*s && y + line_height() <= bottom) {
    char ch = *s++;
    if (ch == '\r') continue;
    if (ch == '\n') {
      label(line, x, y, color); n = 0; line[0] = 0; y += line_height() + 4;
      continue;
    }
    if (n >= int(sizeof(line)) - 2) continue;
    line[n] = ch; line[n + 1] = 0;
    if (text_width(line) > w && n > 0) {
      line[n] = 0;
      label(line, x, y, color);
      y += line_height() + 4;
      n = 0; line[n++] = ch; line[n] = 0;
    } else {
      ++n;
    }
  }
  if (n && y + line_height() <= bottom) label(line, x, y, color);
}

void prompt_tail(int x, int y, int w) {
  const char *visible = prompt;
  while (*visible && text_width(visible) > w) ++visible;
  if (visible != prompt) {
    while (*visible && text_width(visible) + text_width("...") > w) ++visible;
    label("...", x, y, MUTED);
    x += text_width("...");
  }
  label(visible, x, y, WHITE);
}

int columns() { return row == 3 ? 4 : int(strlen(alpha_rows[row])); }

void draw_header(const char *state) {
  box(0, 0, 320, 240, BG);
  box(0, 0, 320, 3, GREEN);
  label("SUPABASE / LOCAL", 12, 11, GREEN);
  label("TinyStories 260K", 12, 27, WHITE);
  label(state, 257 - text_width(state), 14, MUTED);
  box(12, 43, 296, 1, BORDER);
}

void draw_keyboard() {
  draw_header(model_ready ? "READY" : "ERROR");
  box(12, 51, 296, 37, PANEL);
  label("PROMPT", 20, 56, MUTED);
  if (length) prompt_tail(20, 70, 280);
  else label("Type a story opening...", 20, 70, MUTED);

  for (int r = 0; r < 4; ++r) {
    int n = r == 3 ? 4 : int(strlen(alpha_rows[r]));
    int key_w = r == 3 ? 70 : 26;
    int gap = r == 3 ? 5 : 3;
    int total = n * key_w + (n - 1) * gap;
    int x0 = (320 - total) / 2;
    int y = 95 + 34 * r;
    for (int c = 0; c < n; ++c) {
      int x = x0 + c * (key_w + gap);
      bool selected = row == r && col == c;
      bool shift_active = r == 3 && c == 1 && shift;
      box(x, y, key_w, 29, selected ? GREEN : (shift_active ? BORDER : KEY));
      if (!selected) {
        box(x, y, key_w, 1, BORDER);
        box(x, y + 28, key_w, 1, BORDER);
      }
      char key[2] = {0, 0};
      const char *name;
      if (r == 3) name = actions[c];
      else { key[0] = alpha_rows[r][c]; name = key; }
      centered(name, x, y, key_w, 29, selected ? BG : WHITE);
    }
  }
  label("A/B move  UP/DOWN row  C select", 12, 232, MUTED);
}

void draw_result() {
  draw_header(model::busy() ? "RUNNING" : "DONE");
  box(12, 52, 296, 39, PANEL);
  label("PROMPT", 20, 58, MUTED);
  prompt_tail(20, 73, 280);
  label("CONTINUATION", 12, 104, GREEN);
  box(12, 120, 296, 90, PANEL);
  wrapped(model::output(), 20, 130, 280, 204, WHITE);

  box(0, 218, 320, 22, PANEL);
  char stat[56];
  std::snprintf(stat, sizeof(stat), "%d / %d tokens", model::generated_tokens(), model::kMaxGenerated);
  label(stat, 12, 225, MUTED);
  label(model::busy() ? "HOME stop" : "C edit  B clear", 183, 225, GREEN);
}

void start_generation() {
  if (!model_ready) return;
  model::begin(prompt);
  result_view = true;
}

void select_key() {
  if (row == 3) {
    if (col == 0 && length < model::kMaxPrompt) prompt[length++] = ' ';
    if (col == 1) shift = !shift;
    if (col == 2 && length) --length;
    if (col == 3) start_generation();
  } else if (length < model::kMaxPrompt) {
    char ch = alpha_rows[row][col];
    if (!shift && ch >= 'A' && ch <= 'Z') ch += 'a' - 'A';
    prompt[length++] = ch;
    shift = false;
  }
  prompt[length] = 0;
}
} // namespace

int main() {
  badge::init(badge::HIRES);
  model_ready = model::init();
  while (true) {
    poll();
    if (!result_view) {
      if (buttons::pressed(buttons::A)) col = (col + columns() - 1) % columns();
      if (buttons::pressed(buttons::B)) col = (col + 1) % columns();
      if (buttons::pressed(buttons::UP)) { row = (row + 3) % 4; if (col >= columns()) col = columns() - 1; }
      if (buttons::pressed(buttons::DOWN)) { row = (row + 1) % 4; if (col >= columns()) col = columns() - 1; }
      if (buttons::pressed(buttons::C)) select_key();
      draw_keyboard();
      if (!model_ready) label(model::error(), 15, 73, GREEN);
    } else {
      if (buttons::pressed(buttons::HOME) && model::busy()) {
        // Finishes the current token, then returns to editing.
        result_view = false;
      } else if (!model::busy() && buttons::pressed(buttons::C)) {
        result_view = false;
      } else if (!model::busy() && buttons::pressed(buttons::B)) {
        prompt[0] = 0; length = 0; result_view = false;
      }
      if (result_view && model::busy()) model::step();
      if (result_view) draw_result();
      else draw_keyboard();
    }
    present();
    sleep_ms(25);
  }
}
