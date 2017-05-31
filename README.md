## Neural network models for natural language inference

This repository contains the models I used for my masters thesis that investigate a dependency-based sentence encoder, implemented in PyTorch. Included are a simple reimplementation of SPINN by [Bowman et al. (2016)](http://aclweb.org/anthology/P16-1139), a modified variant that handle dependency structures, as well as an ordinary LSTM. The dependency variant replace the shift-reduce parser for binary constituency structures with a transition-based parser to handle dependency structures, and use a composition function with a slightly different parametrization.

## TODO

- [ ] Merge the composition function with separate child matrices (need to handle its usage in a clean way)
- [ ] Add support for prediciting transitions
