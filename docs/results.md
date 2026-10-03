# Evaluation results

Scored where the sensor returned a real reading and t >= 0.2 s. Heights are the height of the top of the ball in metres. Timing errors are model minus measured.

| Dataset | Model | Points | RMSE (m) | MAE (m) | Fall to 20 cm (s) | First apex time (s) | First apex height (m) |
|---|---|---|---|---|---|---|---|
| data.csv | physics (fitted on data.csv) | 183 | 0.042 | 0.036 | -0.009 | -0.030 | +0.044 |
| data.csv | LSTM (not trained on this file) | 183 | 0.234 | 0.204 | -0.005 | -0.040 | -0.368 |
| data.csv | constant = mean of measured | 183 | 0.128 | 0.105 | n/a | -0.270 | -0.242 |
| data2.csv | physics (fitted on data.csv) | 51 | 0.176 | 0.131 | +0.002 | -0.110 | +0.109 |
| data2.csv | LSTM (not trained on this file) | 51 | 0.202 | 0.149 | +0.002 | -0.200 | -0.245 |
| data2.csv | constant = mean of measured | 51 | 0.116 | 0.100 | n/a | -0.370 | -0.144 |
| data3.csv | physics (fitted on data.csv) | 108 | 0.162 | 0.127 | +0.001 | -0.110 | +0.092 |
| data3.csv | LSTM (not trained on this file) | 108 | 0.166 | 0.132 | -0.002 | -0.170 | -0.220 |
| data3.csv | constant = mean of measured | 108 | 0.114 | 0.100 | n/a | -0.380 | -0.196 |

Physics parameters fitted on data.csv: restitution 0.87, per-bounce decay 0.98 (rubber preset: 59 g, 3 cm radius, Cd 0.47). Initial downward speed estimated from the first 0.2 s of each dataset: data.csv 0.83 m/s, data2.csv 0.53 m/s, data3.csv 0.45 m/s.

## Physics cross-fit (RMSE in m; rows = fitted on, columns = evaluated on)

| Fitted on (e, decay) | data.csv | data2.csv | data3.csv |
|---|---|---|---|
| data.csv (0.87, 0.98) | 0.042 | 0.176 | 0.162 |
| data2.csv (0.74, 1.0) | 0.126 | 0.136 | 0.122 |
| data3.csv (0.75, 0.98) | 0.124 | 0.136 | 0.121 |

LSTM: trained once per held-out dataset on synthetic data plus augmented copies of the other two recordings, 30 epochs max, seed 42.
