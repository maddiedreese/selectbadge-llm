#pragma once

namespace model {
  constexpr int kMaxContext = 64;
  constexpr int kMaxGenerated = 24;
  constexpr int kMaxPrompt = 64;
  constexpr int kMaxOutput = 512;

  bool init();
  const char *error();
  void begin(const char *prompt);
  // Runs one transformer forward pass. Returns true when display state changed.
  bool step();
  bool busy();
  int generated_tokens();
  int prompt_tokens();
  const char *output();
}
