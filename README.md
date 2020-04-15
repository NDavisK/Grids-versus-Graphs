# Grids-versus-Graphs

The widely employed ConvLSTM for modeling spatio-temporal data 
can model only data in fixed-sized partitions. The standard convolutional filters
cannot be applied on variable-sized partitions. The search for a model that works well with 
irregular spaced partitions, we delve into Graph-based LSTM models.
A comparison of ConvLSTM and GraphLSTM reveals the competitive performance of GraphLSTM, 
at a lower computational complexity, across three real-world large-scale taxi demand-supply data sets,
with different performance metrics.

For graph-based modeling, we borrow heavily from Cui, Zhiyong, et al. "Traffic graph convolutional recurrent neural network: 
A deep learning framework for network-scale traffic learning and forecasting." and their code at https://github.com/zhiyongc/Graph_Convolutional_LSTM 
