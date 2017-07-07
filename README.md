# EigenPro

## Untar preprocessed MNIST data
The data is preprocessed by z-score and transformed into mat format.
```
cd data
unzip mnist.zip
```

## Run experiments (in MATLAB console)
The experiments will compare Pegasos, Kernel EigenPro, Random Fourier Feature with linear SGD, and Random Fourier Feature with EigenPro on MNIST.
```
run('run_expr.m')
```

## Set CPU/GPU flag
User can change the first line in 'run_expr.m' to switch between GPU and CPU. When set
```
use_gpu = true;
```
the script will use MATLAB gpuarray to store data and weights.

## Experimental results

### Classification Error (MNIST)
In these experiments, EigenPro (Primal) achieves classification error 1.23%, after only 10 epochs. In comparison, Pegasos takes 160 epochs to reach the same error. Although the number of random features used by EigenPro (Random) and RF/DSGD is 6 * 10^4, same as the number of training points, methods using random features deliver slighly worse performance. Specifically, RF/DSGD has error rate 1.71% after 40 epochs and Pegasos reaches error rate 1.65% after the same number of epochs.

<table>
  <tr>
    <th rowspan="2">#Epochs</th>
    <th colspan="4">Primal</th>
    <th colspan="4">Random Fourier Feature</th>
  </tr>
  <tr>
    <td colspan="2">EigenPro</td>
    <td colspan="2">Pegasos</td>
    <td colspan="2">EigenPro</td>
    <td colspan="2">RF/DSGD</td>
  </tr>
  <tr>
    <td></td>
    <td>train</td>
    <td>test</td>
    <td>train</td>
    <td>test</td>
    <td>train</td>
    <td>test</td>
    <td>train</td>
    <td>test</td>
  </tr>
  <tr>
    <td>1</td>
    <td>0.92%</td>
    <td>2.03%</td>
    <td>5.12%</td>
    <td>5.21%</td>
    <td>0.80%</td>
    <td>1.93%</td>
    <td>5.21%</td>
    <td>5.33%</td>
  </tr>
  <tr>
    <td>5</td>
    <td>0.10%</td>
    <td>1.44%</td>
    <td>2.36%</td>
    <td>2.84%</td>
    <td>0.12%</td>
    <td>1.49%</td>
    <td>2.48%</td>
    <td>2.98%</td>
  </tr>
  <tr>
    <td>10</td>
    <td>0.01%</td>
    <td>1.23%</td>
    <td>1.58%</td>
    <td>2.32%</td>
    <td>0.03%</td>
    <td><b>1.44%</b></td>
    <td>1.66%</td>
    <td>2.37%</td>
  </tr>
  <tr>
    <td>20</td>
    <td>0.0%</td>
    <td><b>1.20%</b></td>
    <td>0.90%</td>
    <td>1.93%</td>
    <td>0.01%</td>
    <td>1.45%</td>
    <td>0.98%</td>
    <td>2.03%</td>
  </tr>
  <tr>
    <td>40</td>
    <td>0.0%</td>
    <td>1.20%</td>
    <td>0.39%</td>
    <td>1.65%</td>
    <td>0.0%</td>
    <td>1.46%</td>
    <td>0.49%</td>
    <td>1.71%</td>
  </tr>
</table>


### Training Time per Epoch

<table>
  <tr>
    <th rowspan="2">Computing<br>Resource</th>
    <th colspan="2">Primal</th>
    <th colspan="2">Random Fourier Feature</th>
  </tr>
  <tr>
    <td>EigenPro</td>
    <td>Pegasos</td>
    <td>EigenPro</td>
    <td>RF/DSGD</td>
  </tr>
  <tr>
    <td>One GTX Titan X</td>
    <td>4.8s</td>
    <td>4.6s</td>
    <td>2.2s</td>
    <td>2.0s</td>
  </tr>
  <tr>
    <td>Two Xeon E5-2620</td>
    <td>72s</td>
    <td>70s</td>
    <td>78s</td>
    <td>72s</td>
  </tr>
</table>

### EigenPro Preprocessing Time
In our experiments we construct the EigenPro preconditioner by computing the top 160 approximate eigenvectors for a subsample matrix with 4800 points using Randomized SVD (RSVD).

<table>
  <tr>
    <th>Computing<br>Resource</th>
    <th>RSVD Time<br>(k = 160, m = 4800)</th>
  </tr>
  <tr>
    <td>One GTX Titan X</td>
    <td>7.6s</td>
  </tr>
  <tr>
    <td>Two Xeon E5-2620</td>
    <td>17.7s</td>
  </tr>
</table>
